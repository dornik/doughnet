import torch
import numpy as np
import mpm as us
from mpm.utils.geom import xyzw_to_wxyz, xyzw_from_wxyz
from scipy.spatial.transform import Rotation
from simple_pid import PID


def transform_by_quat(v, q):
    uv = torch.cross(q[1:], v)
    uuv = torch.cross(q[1:], uv)
    return v + 2 * (q[0] * uv + uuv)

def quat_to_R(q):
    d = q.dot(q)
    w, x, y, z = q
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    return torch.tensor([
            [1 - (yy + zz), xy - wz, xz + wy], 
            [xy + wz, 1 - (xx + zz), yz - wx], 
            [xz - wy, yz + wx, 1 - (xx + yy)]]).to(q.device).to(us.FTYPE_TC)

def get_ee_state(ee, state=None):
    if state is None:
        state = ee.get_state()
    ee_state = {
        'pos': state[:3].detach().cpu().numpy(),
        'quat': state[3:7].detach().cpu().numpy(),  # wxyz
        'v': ee.v.detach().cpu().numpy() if ee.v is not None else np.zeros(3),
        'w': ee.w.detach().cpu().numpy() if ee.w is not None else np.zeros(3),
        'is_gripper': ee.is_gripper,
    }
    if ee.is_gripper:
        pos_left, pos_right = EndEffector.get_left_right_pos(state)
        ee_state['pos_left'] = pos_left.detach().cpu().numpy()
        ee_state['pos_right'] = pos_right.detach().cpu().numpy()
        ee_state['v_left'] = (ee.v - ee.v_close).detach().cpu().numpy() if ee.v is not None else np.zeros(3)
        ee_state['v_right'] = (ee.v + ee.v_close).detach().cpu().numpy() if ee.v is not None else np.zeros(3)
        # each finger has the same quat and w as the overall ee (rigid body assumption)
        ee_state['open'] = float(state[7])  # = distance between left and right finger, wrt their inner faces
        ee_state['v_close'] = ee.v_close.detach().cpu().numpy() if ee.v_close is not None else np.zeros(3)
        # = closing speed, relative speed of left and right is double
    return ee_state


class EndEffector:

    def __init__(self, config, entities) -> None:
        self.config = config
        self.entities = entities
        assert len(entities) in [1, 2]
        self.is_gripper = len(entities) == 2
        # controller
        self.controller = EndEffector.NdimPID(config.controller, self.is_gripper)
        # state
        self.state = None
        self.v = None
        self.w = None
        self.v_close = None
        self.v_from_w = None

    def reset(self):
        self.controller.reset()
        self.init_state()

    def get_goal(self):
        return self.controller.get_goal()

    def set_goal(self, goal_state):
        self.controller.set_goal(goal_state)

    def init_state(self):
        init_pos = self.config.state.pos
        init_quat = self.config.state.quat
        if self.is_gripper:
            assert hasattr(self.config.state, 'open')
            init_open = self.config.state.open
            init_state = torch.tensor([*init_pos, *init_quat, init_open]).cuda().to(us.FTYPE_TC)
        else:
            init_state = torch.tensor([*init_pos, *init_quat]).cuda().to(us.FTYPE_TC)
        self.set_state(init_state)

    def get_state(self):
        if self.is_gripper:
            left_pos = self.entities[0].latest_pos[0]
            right_pos = self.entities[1].latest_pos[0]
            pos = (left_pos + right_pos) / 2
            quat = self.entities[0].latest_quat[0]
            open = (left_pos - right_pos).norm()
            is_negative_open = torch.tensor([*(left_pos - right_pos)]).cuda().to(us.FTYPE_TC).dot(
                transform_by_quat(torch.tensor([1.0, 0, 0]).cuda().to(us.FTYPE_TC),
                                  torch.tensor([*quat]).cuda().to(us.FTYPE_TC)))
            open *= -1 if is_negative_open > 0 else 1
            state = torch.tensor([*pos, *quat, open]).cuda().to(us.FTYPE_TC)
        else:
            pos = self.entities[0].latest_pos[0]
            quat = self.entities[0].latest_quat[0]
            open = None
            state = torch.tensor([*pos, *quat]).cuda().to(us.FTYPE_TC)
        return state
    
    def set_state(self, state):
        if self.is_gripper:
            if len(state) == 7:
                state = torch.tensor([*state, self.state[7]]).cuda().to(us.FTYPE_TC)  # use current as default
            # assert len(state) == 8  # xyz, wxyz, open            
            left_pos, right_pos = EndEffector.get_left_right_pos(state)
            self.entities[0].set_position(left_pos)
            self.entities[0].set_quaternion(state[3:7])
            self.entities[0].set_velocity(vel=torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC),
                                          ang=torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC))
            self.entities[1].set_position(right_pos)
            self.entities[1].set_quaternion(state[3:7])
            self.entities[1].set_velocity(vel=torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC),
                                          ang=torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC))
        else:
            self.entities[0].set_position(state[:3])
            self.entities[0].set_quaternion(state[3:7])
            self.entities[0].set_velocity(vel=torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC),
                                          ang=torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC))

    def step(self):
        # observe state
        self.state = self.get_state()
        # update velocities to reach goal
        velocities = self.controller.step(self.state, dt=self.config.controller.dt)
        self._move(v=velocities[:3], w=velocities[3:6])
        if self.is_gripper:
            self._grip(v_close=velocities[6])

        if any(torch.abs(velocities) > self.config.controller.tol):
            return False
        else:
            return True

    def _move(self, v, w):
        self.v = v
        self.w = w
        if self.is_gripper:
            r = torch.tensor([self.state[7]*0.5, 0.0, 0.0]).cuda().to(us.FTYPE_TC)
            r = transform_by_quat(r, self.state[3:7])
            self.v_from_w = torch.cross(self.w, r)
        else:
            self.v_from_w = torch.tensor([0.0, 0.0, 0.0]).cuda().to(us.FTYPE_TC)

        self.entities[0].set_velocity(vel=self.v - self.v_from_w, ang=w)
        if self.is_gripper:
            w_inv = w.clone()
            w_inv[0] *= -1  # [-w, x, y, z] -- opposite angle around same axis
            self.entities[1].set_velocity(vel=self.v + self.v_from_w, ang=w_inv)
        return v, w

    def _grip(self, v_close):
        if not self.is_gripper:
            raise RuntimeError("cannot grip with single finger entity")
        self.v_close = transform_by_quat(torch.tensor([v_close, 0.0, 0.0]).cuda().to(us.FTYPE_TC),
                                         self.state[3:7])
        self.entities[0].set_velocity(vel=self.v - self.v_from_w - self.v_close)
        self.entities[1].set_velocity(vel=self.v + self.v_from_w + self.v_close)

    @staticmethod
    def get_left_right_pos(state):
        pos, quat, open = state[:3], state[3:7], torch.tensor([state[7], 0.0, 0.0]).cuda().to(us.FTYPE_TC)
        left_pos = pos - transform_by_quat(open*0.5, quat)
        right_pos = pos + transform_by_quat(open*0.5, quat)
        return left_pos, right_pos

    class NdimPID:

        def __init__(self, config, is_gripper):
            self.config = config
            self.is_gripper = is_gripper
            # parse config
            n = 6
            p = [config.lin.p]*3 + [config.ang.p]*3
            i = [config.lin.i]*3 + [config.ang.i]*3
            d = [config.lin.d]*3 + [config.ang.d]*3
            v_max = [config.lin.vmax]*3 + [config.ang.vmax]*3
            if self.is_gripper:
                n += 1
                p += [config.grip.p]
                i += [config.grip.i]
                d += [config.grip.d]
                v_max += [config.grip.vmax]
            # create controllers
            self.controllers = []
            for dim in range(n):
                controller = PID(p[dim], i[dim], d[dim], sample_time=config.dt)
                controller.output_limits = (-v_max[dim], v_max[dim])
                self.controllers.append(controller)
            self.goal_euler = None
        
        def reset(self):
            for controller in self.controllers:
                controller.reset()
            self.goal_euler = None

        def parse_from_quat(self, state):
            if state is None:
                return None
            else:  # convert quat to euler
                euler = Rotation.from_quat(xyzw_from_wxyz(state[3:7].cpu())).as_euler('xyz')
                return torch.tensor([*state[:3], *euler, *state[7:]]).cuda().to(us.FTYPE_TC)
        
        def parse_to_quat(self, state):
            if state is None:
                return None
            else:  # convert euler to quat
                quat = Rotation.from_euler('xyz', state[3:6].cpu()).as_quat()
                return torch.tensor([*state[:3], *xyzw_to_wxyz(quat), *state[6:]]).cuda().to(us.FTYPE_TC)
            
        def get_goal(self):
            return self.parse_to_quat(self.goal_euler)

        def set_goal(self, goal):
            # in goal: [x, y, z, qw, qx, qy, qz{, open}]
            self.goal_euler = self.parse_from_quat(goal)
            if self.goal_euler is not None:
                assert len(self.goal_euler) == len(self.controllers)
                for i, g in enumerate(self.goal_euler):
                    self.controllers[i].reset()  # clears pid terms
                    self.controllers[i].setpoint = g
        
        def step(self, state, dt):
            # in state: [x, y, z, qw, qx, qy, qz{, open}]
            # out velocities: [x, y, z, roll, pitch, yaw{, open}]
            state_euler = self.parse_from_quat(state)
            if self.goal_euler is None:
                return torch.tensor([0.0] * len(self.controllers)).cuda().to(us.FTYPE_TC)
            else:
                assert len(state_euler) == len(self.controllers)
                return torch.tensor([controller(state_euler[i], dt)
                                     for i, controller in enumerate(self.controllers)]).cuda().to(us.FTYPE_TC)
