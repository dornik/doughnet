import uuid
import numpy as np
import torch
import taichi as ti
import mpm as us
from .mesh import Mesh
from mpm.utils.repr import _repr, _repr_uuid
from mpm.engine.states.entities import ToolEntityState
from mpm.utils.geom import quat_mul, axis_angle_to_quat, euler_to_quat


@ti.data_oriented
class ToolEntity:
    # Mesh-based tool body entity
    def __init__(
        self,
        scene,
        solver,
        material,
        geom,
        surface_options,
    ):
        self.scene           = scene
        self.sim             = scene.sim
        self.solver          = solver
        self.id              = str(uuid.uuid4())
        self.material        = material
        self.geom            = geom
        self.surface_options = surface_options
        self.name            = geom.name

        self.pos  = ti.Vector.field(3, us.FTYPE_TI, needs_grad=False, needs_dual=False) # positon
        self.quat = ti.Vector.field(4, us.FTYPE_TI, needs_grad=False, needs_dual=False) # quaternion wxyz
        self.vel  = ti.Vector.field(3, us.FTYPE_TI, needs_grad=False, needs_dual=False) # velocity
        self.ang  = ti.Vector.field(3, us.FTYPE_TI, needs_grad=False, needs_dual=False) # angular velocity

        ti.root.dense(ti.i, (self.sim.max_substeps_local+1,)).place(
            self.pos, self.pos.grad, self.quat, self.quat.grad,
            self.vel, self.vel.grad, self.ang, self.ang.grad
        )

        self.init_pos = np.array(geom.pos).astype(us.FTYPE_NP)
        self.init_quat = np.array(euler_to_quat(geom.euler), dtype=us.FTYPE_NP)

        self.mesh = Mesh(
            container = self,
            material  = material,
            geom      = geom,
        )
        self.init_tgt_vars()
        self.latest_pos = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.latest_quat = ti.Vector.field(4, dtype=ti.f32, shape=(1))

        us.logger.info(f'Entity {_repr_uuid(self.id)} ("{self.name}") added. class: {self.__class__.__name__}, geom: {self.geom.__class__.__name__}, material: {self.material.name}.')

    def init_tgt_vars(self):
        # temp variable to store targets for next step
        self._tgt = {
            'pos'  : torch.tensor(self.init_pos, requires_grad=False).cuda(),
            'quat' : torch.tensor(self.init_quat, requires_grad=False).cuda(),
            'vel'  : None,
            'ang'  : None,
        }

    def save_ckpt(self):
        # restart from frame 0 in memory
        self.copy_frame(self.sim.max_substeps_local, 0)

    def substep_pre_coupling(self, f):
        self.advect(f)

    def substep_coupling(self, f):
        pass

    def substep_post_coupling(self, f):
        self.update_latest_pos(f)
        self.update_latest_quat(f)
        self.update_mesh_pose(f)

    def update_mesh_pose(self, f):
        # For visualization only. No need to compute grad.
        self.mesh.update_vertices(f)

    @ti.func
    def collide(self, f, pos_world, vel_mat, dt):
        return self.mesh.collide(f, pos_world, vel_mat, dt)
    
    @ti.func
    def collision_info(self, f, pos_world, info):
        return self.mesh.collision_info(f, pos_world, info)

    @ti.kernel
    def update_latest_pos(self, f: ti.i32):
        self.latest_pos[0] = ti.cast(self.pos[f], ti.f32)

    @ti.kernel
    def update_latest_quat(self, f: ti.i32):
        self.latest_quat[0] = ti.cast(self.quat[f], ti.f32)

    @ti.kernel
    def advect(self, f: ti.i32):
        self.pos[f+1] = self.pos[f] + self.vel[f] * self.solver.substep_dt
        # rotate in world coordinates about itself.
        self.quat[f+1] = quat_mul(axis_angle_to_quat(self.ang[f] * self.solver.substep_dt), self.quat[f])

    # state set and copy ...
    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        self.pos[target]  = self.pos[source]
        self.quat[target] = self.quat[source]
        self.vel[target]  = self.vel[source]
        self.ang[target]  = self.ang[source]

    @ti.kernel
    def get_frame(self, f: ti.i32, pos: ti.types.ndarray(), quat: ti.types.ndarray(), vel: ti.types.ndarray(), ang: ti.types.ndarray()):
        for i in ti.static(range(3)):
            pos[i] = self.pos[f][i]
        for i in ti.static(range(4)):
            quat[i] = self.quat[f][i]
        for i in ti.static(range(3)):
            vel[i] = self.vel[f][i]
        for i in ti.static(range(3)):
            ang[i] = self.ang[f][i]

    @ti.kernel
    def set_frame(self, f: ti.i32, pos: ti.types.ndarray(), quat: ti.types.ndarray(), vel: ti.types.ndarray(), ang: ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.pos[f][i] = pos[i]
        for i in ti.static(range(4)):
            self.quat[f][i] = quat[i]
        for i in ti.static(range(3)):
            self.vel[f][i] = vel[i]
        for i in ti.static(range(3)):
            self.ang[f][i] = ang[i]

    def get_state(self, f=None):
        state = ToolEntityState(self, self.sim.cur_step_global)

        if f is None:
            f = self.sim.cur_substep_local
        self.get_frame(f, state.pos, state.quat, state.vel, state.ang)

        return state

    def set_state(self, f, state):
        f = self.sim.cur_substep_local
        self.set_frame(f, state.pos, state.quat, state.vel, state.ang)

    def build(self):
        self.init_state = ToolEntityState(self, 0)
        self.set_init_state(self.init_pos, self.init_quat)

    @ti.kernel
    def set_init_state(self, pos: ti.types.ndarray(), quat: ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.pos[0][i] = pos[i]
        for i in ti.static(range(4)):
            self.quat[0][i] = quat[i]

    @ti.kernel
    def set_vel(self, s: ti.i32, vel: ti.types.ndarray()):
        for j in range(s*self.sim.n_substeps, (s+1)*self.sim.n_substeps):
            for k in ti.static(range(3)):
                self.vel[j][k] = vel[k]

    @ti.kernel
    def set_ang(self, s: ti.i32, ang: ti.types.ndarray()):
        for j in range(s*self.sim.n_substeps, (s+1)*self.sim.n_substeps):
            for k in ti.static(range(3)):
                self.ang[j][k] = ang[k]

    @ti.kernel
    def set_pos(self, s: ti.i32, pos: ti.types.ndarray()):
        j = s*self.sim.n_substeps
        for k in ti.static(range(3)):
            self.pos[j][k] = pos[k]

    @ti.kernel
    def set_quat(self, s: ti.i32, quat: ti.types.ndarray()):
        j = s*self.sim.n_substeps
        for k in ti.static(range(4)):
            self.quat[j][k] = quat[k]

    def set_velocity(self, vel=None, ang=None):
        if vel is not None:
            self._tgt['vel'] = vel.clone()

        if ang is not None:
            self._tgt['ang'] = ang.clone()

    def set_position(self, pos):
        self._tgt['pos'] = pos.clone()

    def set_quaternion(self, quat):
        self._tgt['quat'] = quat.clone()

    def process_input(self):
        if self._tgt['pos'] is not None:
            us.assert_contiguous(self._tgt['pos'])
            self.set_pos(self.sim.cur_step_local, self._tgt['pos'])

        if self._tgt['quat'] is not None:
            us.assert_contiguous(self._tgt['quat'])
            self.set_quat(self.sim.cur_step_local, self._tgt['quat'])

        if self._tgt['vel'] is not None:
            us.assert_contiguous(self._tgt['vel'])
            self.set_vel(self.sim.cur_step_local, self._tgt['vel'])

        if self._tgt['ang'] is not None:
            us.assert_contiguous(self._tgt['ang'])
            self.set_ang(self.sim.cur_step_local, self._tgt['ang'])

        self._tgt['pos']  = None
        self._tgt['quat'] = None
        self._tgt['vel']  = None
        self._tgt['ang']  = None
