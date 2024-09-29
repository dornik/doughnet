import numpy as np
import mpm as us
import taichi as ti
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------------------------
# ------------------------------------- taichi ---------------------------------------
# ------------------------------------------------------------------------------------

@ti.func
def quat_mul(u, v):
    terms = v.outer_product(u)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    return ti.Vector([w, x, y, z]).normalized()

@ti.func
def axis_angle_to_quat(axis_angle):
    w = axis_angle.norm(us.EPS)
    v = (axis_angle/w) * ti.sin(w/2)
    return ti.Vector([ti.cos(w/2), v[0], v[1], v[2]]).normalized()

@ti.func
def inv_quat(quat):
    return ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]]).normalized()

@ti.func
def inv_trans_ti(pos_A, trans_B_to_A, rot_B_to_A):
    return rot_B_to_A.inverse() @ (pos_A - trans_B_to_A)

@ti.func
def trans_ti(pos_B, trans_B_to_A, rot_B_to_A):
    return rot_B_to_A.inverse() @ pos_B + trans_B_to_A

@ti.func
def normalize(v, eps):
    return v / (v.norm(eps))

@ti.func
def rotate(vec, quat):
  s, u = quat[0], quat[1:]
  r = 2 * (u.dot(vec)* u) + (s * s - u.dot(u)) * vec
  r = r + 2 * s * u.cross(vec)
  return r
  
@ti.func
def transform_by_quat_ti(v, quat):
    qvec = ti.Vector([quat[1], quat[2], quat[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (quat[0] * uv + uuv)

@ti.func
def inv_transform_by_quat_ti(v, quat):
    return transform_by_quat_ti(v, inv_quat(quat))

@ti.func
def transform_by_T_ti(pos, T, dtype):
    new_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0], dt=dtype)
    new_pos = T @ new_pos
    return new_pos[:3]

@ti.func
def transform_by_trans_quat_ti(pos, trans, quat):
    return transform_by_quat_ti(pos, quat) + trans

@ti.func
def inv_transform_by_trans_quat_ti(pos, trans, quat):
    return transform_by_quat_ti(pos - trans, inv_quat(quat))

@ti.func
def quat_to_R(q):
    """Converts quaternion to 3x3 rotation matrix."""
    d = q.dot(q)
    w, x, y, z = q
    s = 2 / d
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs

    return ti.Matrix([
            [1 - (yy + zz), xy - wz, xz + wy], 
            [xy + wz, 1 - (xx + zz), yz - wx], 
            [xz - wy, yz + wx, 1 - (xx + yy)]])

@ti.func
def transform_motion(pos, rot, m_ang, m_vel):
    rot_t = inv_quat(rot)
    ang = transform_by_quat_ti(m_ang, rot_t)
    vel = transform_by_quat_ti(m_vel - pos.cross(m_ang), rot_t)
    return ang, vel

@ti.func
def transform_transform(pos, rot, t_pos, t_rot):
    o_pos = pos + transform_by_quat_ti(t_pos, rot)
    o_rot = quat_mul(rot, t_rot)
    return o_pos, o_rot

@ti.func
def transform_inertia(pos, rot, i_inertial, i_trans_pos, i_trans_rot, i_mass):
    h = ti.Matrix.rows([
        pos.cross(ti.Vector([-1.0, 0.0, 0.0])), 
        pos.cross(ti.Vector([0.0, -1.0, 0.0])), 
        pos.cross(ti.Vector([0.0, 0.0, -1.0]))])

    rotm = quat_to_R(rot)
    i = rotm @ i_inertial @ rotm.transpose() + h @ h.transpose() * i_mass
    trans_pos = pos * i_mass
    trans_rot = rot
    return i, trans_pos, trans_rot, i_mass

@ti.func
def transform_inv_motion(t_pos, t_rot, m_ang, m_vel):
    ang = transform_by_quat_ti(m_ang, t_rot)
    vel = transform_by_quat_ti(m_vel, t_rot) + t_pos.cross(m_ang)
    return ang, vel

@ti.func
def inertial_mul(trans_pos, i, mass, vel, ang):
    _ang = i @ ang + trans_pos.cross(vel)
    _vel = mass * vel - trans_pos.cross(ang)
    return _ang, _vel

@ti.func
def motion_cross_force(m_ang, m_vel, f_ang, f_vel):
    vel = m_ang.cross(f_vel)
    ang = m_ang.cross(f_ang) + m_vel.cross(f_vel)
    return ang, vel

@ti.func
def motion_cross_motion(s_ang, s_vel, m_ang, m_vel):
    vel = s_ang.cross(m_vel) + s_vel.cross(m_ang)
    ang = s_ang.cross(m_ang)
    return ang, vel

@ti.func
def orthogonals(a):
    """Returns orthogonal vectors `b` and `c`, given a normal vector `a`."""
    y, z = ti.Vector([0., 1., 0.]), ti.Vector([0., 0., 1.])
    b = z
    if -0.5 < a[1] and a[1] < 0.5:
        b = y
    b = b - a * a.dot(b)
    # make b a normal vector. however if a is a zero vector, zero b as well.
    b = b.normalized()
    if a.norm() < 1e-5:
        b = b * 0.
    return b, a.cross(b)

@ti.func
def imp_aref(params, pos, vel):
    timeconst, dampratio, dmin, dmax, width, mid, power = params

    imp_x = ti.abs(pos) / width
    imp_a = (1.0 / mid ** (power - 1)) * imp_x ** power
    imp_b = 1 - (1.0 / (1 - mid) ** (power - 1)) * (1 - imp_x) ** power
    imp_y = imp_a if imp_x < mid else imp_b

    imp = dmin + imp_y * (dmax - dmin)
    imp = ti.math.clamp(imp, dmin, dmax)
    imp = dmax if imp_x > 1.0 else imp

    b = 2 / (dmax * timeconst)
    k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)

    aref = -b * vel - k * imp * pos

    return imp, aref

@ti.func
def closest_segment_point(a, b, pt):
    ab = b - a
    t = (pt - a).dot(ab) / (ab.dot(ab) + 1e-6)
    return a + ti.math.clamp(t, 0.0, 1.0) * ab

@ti.func
def get_face_norm(v0, v1, v2):
    edge0 = v1 - v0
    edge1 = v2 - v0
    face_norm = edge0.cross(edge1)
    face_norm = face_norm.normalized()
    return face_norm


# ------------------------------------------------------------------------------------
# ------------------------------------- numpy ----------------------------------------
# ------------------------------------------------------------------------------------

def xyzw_to_wxyz(xyzw):
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

def xyzw_from_wxyz(wxyz):
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

def euler_to_quat(euler_xyz):
    quat_xyzw = Rotation.from_euler('zyx', euler_xyz[::-1], degrees=True).as_quat()
    return xyzw_to_wxyz(quat_xyzw)
    
def R_to_quat(R):
    quat_xyzw = Rotation.from_matrix(R).as_quat()
    return xyzw_to_wxyz(quat_xyzw)
    
def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))
    
    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])

def scale_to_T(scale):
    T_scale = np.eye(4, dtype=scale.dtype)
    T_scale[[0, 1, 2], [0, 1, 2]] = scale
    return T_scale

def trans_quat_to_T(trans=None, quat=None):
    if trans is not None:
        dtype = trans.dtype
    else:
        dtype = quat.dtype

    T = np.eye(4, dtype=dtype)
    if trans is not None:
        T[:3, 3] = trans
    if quat is not None:
        T[:3, :3] = Rotation.from_quat(xyzw_from_wxyz(quat)).as_matrix()
        
    return T

def transform_by_T_np(pos, T):
    if len(pos.shape) == 2:
        assert pos.shape[1] == 3
        new_pos = np.hstack([pos, np.ones_like(pos[:, :1])]).T
        new_pos = (T @ new_pos).T
        new_pos = new_pos[:, :3]

    elif len(pos.shape) == 1:
        assert pos.shape[0] == 3
        new_pos = np.append(pos, 1)
        new_pos = T @ new_pos
        new_pos = new_pos[:3]

    else:
        assert False

    return new_pos

def transform_by_trans_quat_np(pos, trans=None, quat=None):
    return transform_by_quat_np(pos, quat) + trans

def transform_by_quat_np(v, quat):
    qvec = quat[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (quat[0] * uv + uuv)

def default_pos():
    return np.zeros(3)

def default_quat():
    return np.array([1., 0., 0., 0.])

def default_dofs_limit(n=6):
    return np.tile([[-np.inf,  np.inf]], [n, 1])

def default_dofs_invweight(n=6):
    return np.ones([n])


def default_dofs_motion_ang(n=6):
    if n == 6:
        return np.array([[0., 0., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.],
                         [1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
    else:
        assert False

def default_dofs_motion_vel(n=6):
    if n == 6:
        return np.array([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.],
                         [0., 0., 0.],
                         [0., 0., 0.],
                         [0., 0., 0.]])
    else:
        return False

def default_dofs_stiffness(n=6):
    return np.zeros(n)

def default_dofs_armature(n=6):
    return np.zeros(n)

@ti.data_oriented
class SpatialHasher:
    def __init__(self, grid_size, grid_res, n_hash_slots=None):
        self.grid_size = grid_size
        self.grid_res = np.array(grid_res).astype(us.ITYPE_NP)

        if n_hash_slots is None:
            self.n_hash_slots = np.prod(grid_res)
        else:
            self.n_hash_slots = n_hash_slots

    @ti.func
    def pos_to_grid(self, pos):
        return ti.floor(pos / self.grid_size, ti.i32)

    @ti.func
    def grid_to_slot(self, grid_id):
        return (grid_id[0] * self.grid_res[1] * self.grid_res[2] + grid_id[1] * self.grid_res[2] + grid_id[2]) % self.n_hash_slots
    
    @ti.func
    def pos_to_slot(self, pos):
        return self.grid_to_slot(self.pos_to_grid(pos))
    