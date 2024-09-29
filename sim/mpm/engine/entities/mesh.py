import trimesh
import numpy as np
import taichi as ti
import pickle as pkl
import mpm as us
from mpm.utils.misc import *
from mpm.utils.mesh import *
import mpm.utils.geom as geom_utils
from mpm.engine.materials import Tool


@ti.data_oriented
class Mesh:
    def __init__(self, container, material, geom):
        self.container      = container
        self.material       = material
        self.collision      = material.collision
        self.sdf_res        = material.sdf_res
        self.pos            = geom.offset_pos
        self.euler          = geom.offset_euler
        self.scale          = geom.scale
        self.raw_file       = geom.file
        self.raw_file_vis   = geom.file if geom.file_vis is None else geom.file_vis
        self.gl_renderer_id = None

        if not isinstance(self.material, Tool):
            us.raise_exception('Material not supported by mesh')

        self.load_file()
        self.init_transform()

    def load_file(self):
        # mesh
        self.raw_file_path       = get_raw_mesh_path(self.raw_file)
        self.raw_file_vis_path   = get_raw_mesh_path(self.raw_file_vis)
        self.processed_file_path = get_processed_mesh_path(self.raw_file, self.raw_file_vis)
        if self.collision:
            self.processed_sdf_path = get_processed_sdf_path(self.raw_file, self.sdf_res)
        self.mesh = trimesh.load(self.processed_file_path)
        self.raw_vertices = np.ascontiguousarray(np.array(self.mesh.vertices, dtype=np.float32))
        self.raw_vertex_normals_np = np.ascontiguousarray(np.array(self.mesh.vertex_normals, dtype=np.float32))
        self.faces_np = np.ascontiguousarray(np.array(self.mesh.faces, dtype=np.int32)).flatten()

        self.n_vertices = len(self.raw_vertices)
        self.n_faces = len(self.faces_np)

        # load color
        self.colors_np = np.tile([self.container.surface_options.color], [self.n_vertices, 1]).astype(np.float32)

        if self.collision:
            # sdf
            self.friction = self.material.friction
            sdf_data = pkl.load(open(self.processed_sdf_path, 'rb'))
            self.sdf_voxels_np = sdf_data['voxels'].astype(us.FTYPE_NP)
            self.sdf_voxels_res = self.sdf_voxels_np.shape[0]
            self.T_mesh_to_voxels_np = sdf_data['T_mesh_to_voxels'].astype(us.FTYPE_NP)

    def init_transform(self):
        scale = np.array(self.scale, dtype=us.FTYPE_NP)
        pos = np.array(self.pos, dtype=us.FTYPE_NP)
        quat = np.array(geom_utils.euler_to_quat(self.euler), dtype=us.FTYPE_NP)

        # apply initial transforms (scale then quat then pos)
        T_init = geom_utils.trans_quat_to_T(pos, quat) @ geom_utils.scale_to_T(scale)
        self.init_vertices_np = np.ascontiguousarray(geom_utils.transform_by_T_np(self.raw_vertices, T_init).astype(np.float32))

        R_init = geom_utils.trans_quat_to_T(None, quat)
        self.init_vertex_normals_np = np.ascontiguousarray(geom_utils.transform_by_T_np(self.raw_vertex_normals_np, R_init).astype(np.float32))

        # init ti fields
        self.init_vertices       = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.init_vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.faces               = ti.field(dtype=ti.i32, shape=(self.n_faces))
        self.colors              = ti.Vector.field(self.colors_np.shape[1], dtype=ti.f32, shape=(self.n_vertices))

        self.init_vertices.from_numpy(self.init_vertices_np)
        self.init_vertex_normals.from_numpy(self.init_vertex_normals_np)
        self.faces.from_numpy(self.faces_np)
        self.colors.from_numpy(self.colors_np)

        if self.collision:
            self.T_mesh_to_voxels_np = self.T_mesh_to_voxels_np @ np.linalg.inv(T_init)
            self.sdf_voxels          = ti.field(dtype=us.FTYPE_TI, shape=self.sdf_voxels_np.shape)
            self.T_mesh_to_voxels    = ti.Matrix.field(4, 4, dtype=us.FTYPE_TI, shape=())

            self.sdf_voxels.from_numpy(self.sdf_voxels_np)
            self.T_mesh_to_voxels.from_numpy(self.T_mesh_to_voxels_np)

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))

    @ti.kernel
    def update_vertices(self, f: ti.i32):
        for i in self.vertices:
            self.vertices[i] = ti.cast(geom_utils.transform_by_trans_quat_ti(self.init_vertices[i], self.container.pos[f], self.container.quat[f]), self.vertices.dtype)
            self.vertex_normals[i] = ti.cast(geom_utils.transform_by_quat_ti(self.init_vertex_normals[i], self.container.quat[f]), self.vertices.dtype)

    @ti.func
    def sdf(self, f, pos_world):
        # sdf value from world coordinate
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_voxels = geom_utils.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], us.FTYPE_TI)

        return self.sdf_(pos_voxels)

    @ti.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, ti.i32)
        signed_dist = ti.cast(0.0, us.FTYPE_TI)
        if (base >= self.sdf_voxels_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.sdf_voxels[voxel_pos]

        return signed_dist

    @ti.func
    def normal(self, f, pos_world):
        # compute normal with finite difference
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_voxels = geom_utils.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], us.FTYPE_TI)
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_voxels[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = geom_utils.transform_by_quat_ti(normal_vec_mesh, self.container.quat[f])
        normal_vec_world = geom_utils.normalize(normal_vec_world, us.EPS)

        return normal_vec_world

    @ti.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, us.FTYPE_TI)
        normal_vec = ti.Vector([0, 0, 0], dt=us.FTYPE_TI)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = geom_utils.normalize(normal_vec, us.EPS)

        return normal_vec

    @ti.func
    def vel_collider(self, f, pos_world, dt):
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_world_new = geom_utils.transform_by_trans_quat_ti(pos_mesh, self.container.pos[f+1], self.container.quat[f+1])
        vel_collider = (pos_world_new - pos_world) / dt
        return vel_collider

    @ti.func
    def collide(self, f, pos_world, vel_mat, dt):
        if ti.static(self.collision):
            signed_dist = self.sdf(f, pos_world)
            influence = ti.min(ti.exp(-signed_dist * self.material.contact_softness), 1)
            if signed_dist <= 0 or (self.material.contact_softness > 0 and influence > 0.1):
                vel_collider = self.vel_collider(f, pos_world, dt)

                if ti.static(self.friction > 10.0):
                    vel_mat = vel_collider
                else:
                    # v w.r.t collider
                    rel_v = vel_mat - vel_collider
                    normal_vec = self.normal(f, pos_world)
                    normal_component = rel_v.dot(normal_vec)

                    # remove inward velocity, if any
                    rel_v_t = rel_v - ti.min(normal_component, 0) * normal_vec
                    rel_v_t_norm = rel_v_t.norm()

                    # tangential component after friction (if friction exists)
                    rel_v_t_friction = rel_v_t / rel_v_t_norm * ti.max(0, rel_v_t_norm + normal_component * self.friction)

                    # tangential component after friction
                    flag = ti.cast(normal_component < 0 and rel_v_t_norm > us.EPS, us.FTYPE_TI)
                    rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
                    vel_mat = vel_collider + rel_v_t * influence + rel_v * (1-influence)

        return vel_mat

    @ti.func
    def is_collide(self, f, pos_world):
        flag = 0
        if ti.static(self.collision):
            signed_dist = self.sdf(f, pos_world)
            if signed_dist <= 0:
                flag = 1

        return flag
    
    @ti.func
    def collision_info(self, f, pos_world, info):
        if ti.static(self.collision):
            signed_dist = self.sdf(f, pos_world)
            influence = ti.min(ti.exp(-signed_dist * self.material.contact_softness), 1)
            info[0] = signed_dist
            info[1] = influence
            if signed_dist <= 0 or (self.material.contact_softness > 0 and influence > 0.1):
                info[2] = ti.cast(1.0, us.FTYPE_TI)
        return info
