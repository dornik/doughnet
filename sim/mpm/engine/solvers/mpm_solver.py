import numpy as np
import mpm as us
import taichi as ti
from mpm.utils.misc import *
from mpm.engine.entities import MPMEntity
from mpm.engine.boundaries import CubeBoundary
from mpm.engine.states.solvers import MPMSolverState


@ti.data_oriented
class MPMSolver:
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        self.scene   = scene
        self.sim     = sim
        self.options = options

        # options
        self.gravity                        = ti.Vector(options.gravity)
        self.step_dt                        = options.step_dt
        self.substep_dt                     = options.substep_dt
        self.grid_density                   = options.grid_density
        self.particle_diameter              = options.particle_diameter
        self.upper_bound                    = np.array(options.upper_bound)
        self.lower_bound                    = np.array(options.lower_bound)
        self.inv_dx                         = options.grid_density
        self.leaf_block_size                = options.leaf_block_size
        self.use_sparse_grid                = options.use_sparse_grid

        # ensure solver n_substeps is compatible with simulator
        assert int(self.step_dt / self.substep_dt) == self.sim.n_substeps

        # dependent parameters 
        self.dx               = float(1.0 / self.grid_density)
        self.p_vol            = float((self.dx * 0.5) ** 2)
        self.lower_bound_cell = np.round(self.grid_density * self.lower_bound).astype(us.ITYPE_NP)
        self.upper_bound_cell = np.round(self.grid_density * self.upper_bound).astype(us.ITYPE_NP)
        self.grid_res         = self.upper_bound_cell - self.lower_bound_cell + 1 # +1 to include both corner
        self.grid_offset      = ti.Vector(self.lower_bound_cell)
        if self.use_sparse_grid:
            self.grid_res = (np.ceil(self.grid_res / self.leaf_block_size) * self.leaf_block_size).astype(us.ITYPE_NP)


        # materials
        self.mat_count = 0
        self.mat_idxs = []
        self.mat_models = dict()

        # entities
        self.entities = list()

        # boundary
        self.setup_boundary()

    def setup_boundary(self):
        # safety padding
        self.boundary_padding = 3 * self.dx
        self.boundary = CubeBoundary(
            lower = self.lower_bound + self.boundary_padding,
            upper = self.upper_bound - self.boundary_padding,
        )

    def init_particle_fields(self):
        # dynamic particle state
        particle_state = ti.types.struct(
            pos   = ti.types.vector(3, us.FTYPE_TI),  # position
            vel   = ti.types.vector(3, us.FTYPE_TI),  # velocity
            C     = ti.types.matrix(3, 3, us.FTYPE_TI),  # affine velocity field
            F     = ti.types.matrix(3, 3, us.FTYPE_TI),  # deformation gradient
            F_tmp = ti.types.matrix(3, 3, us.FTYPE_TI),  # temp deformation gradient
            U     = ti.types.matrix(3, 3, us.FTYPE_TI),  # SVD
            V     = ti.types.matrix(3, 3, us.FTYPE_TI),  # SVD
            S     = ti.types.matrix(3, 3, us.FTYPE_TI),  # SVD
            collision_info = ti.types.vector(3, us.FTYPE_TI),  # collision info
            collision_info_gripper = ti.types.vector(6, us.FTYPE_TI),  # 2xcollision info (left and right finger)
        )

        # dynamic particle state without gradient
        particle_state_ng = ti.types.struct(
            active = ti.i32,
        )

        # static particle info
        particle_info = ti.types.struct(
            mat_idx = ti.i32,
            mu      = us.FTYPE_TI,
            lam     = us.FTYPE_TI,
            mass    = us.FTYPE_TI,
            color   = ti.types.vector(4, ti.f32),
        )

        # single frame particle state for rendering
        particle_state_render = ti.types.struct(
            pos      = ti.types.vector(3, ti.f32),
            active = ti.i32,
            color   = ti.types.vector(4, ti.f32),
        )

        # construct fields
        self.particles        = particle_state.field(shape=(self.sim.max_substeps_local+1, self.n_particles_max), needs_grad=False, needs_dual=False, layout=ti.Layout.SOA)
        self.particles_ng     = particle_state_ng.field(shape=(self.sim.max_substeps_local+1, self.n_particles_max), needs_grad=False, needs_dual=False, layout=ti.Layout.SOA)
        self.particles_info   = particle_info.field(shape=self.n_particles_max, needs_grad=False, needs_dual=False, layout=ti.Layout.SOA)
        self.particles_render = particle_state_render.field(shape=self.n_particles_max, needs_grad=False, needs_dual=False, layout=ti.Layout.SOA)

    def init_grid_fields(self):
        grid_cell_state = ti.types.struct(
            vel_in  = ti.types.vector(3, us.FTYPE_TI), # input momentum/velocity
            mass  = us.FTYPE_TI,                     # mass
            vel_out = ti.types.vector(3, us.FTYPE_TI), # output momentum/velocity
        )

        if self.use_sparse_grid:
            # temporal block -> coarse block -> fine block
            self.grid_block_0 = ti.root.dense(ti.axes(0), self.sim.max_substeps_local+1)
            self.grid_block_1 = self.grid_block_0.pointer(ti.axes(1, 2, 3), self.grid_res // self.leaf_block_size)
            self.grid_block_2 = self.grid_block_1.dense(ti.axes(1, 2, 3), self.leaf_block_size)

            self.grid = grid_cell_state.field(needs_grad=False, needs_dual=False)
            self.grid_block_2.place(self.grid, self.grid.grad)

            self.deactivate_grid_block()

        else:
            self.grid = grid_cell_state.field(shape=(self.sim.max_substeps_local+1, *self.grid_res), needs_grad=False, needs_dual=False, layout=ti.Layout.SOA)

    def deactivate_grid_block(self):
        self.grid_block_1.deactivate_all()

    def build(self):
        # particles and entities
        self.n_particles_max = self.n_particles

        if self.is_active():
            self.init_particle_fields()
            self.init_grid_fields()

            for entity in self.entities:
                entity.add_to_solver()

    # ------------------------------------------------------------------------------------
    # -------------------------------------- misc ----------------------------------------
    # ------------------------------------------------------------------------------------
    
    def add_entity(self, material, geom, surface_options):  
        entity = MPMEntity(
            scene             = self.scene,
            solver            = self,
            material          = material,
            geom              = geom,
            surface_options   = surface_options,
            particle_diameter = self.particle_diameter,
            idx_offset        = self.n_particles,
        )

        # entities added before building will be added to solver during build()
        if self.scene.is_built:
            entity.add_to_solver()
            
        self.entities.append(entity)
        return entity

    def is_active(self):
        return self.n_particles_max > 0

    @property
    def n_particles(self):
        return sum([entity.n for entity in self.entities])

    # ------------------------------------------------------------------------------------
    # ----------------------------------- simulation -------------------------------------
    # ------------------------------------------------------------------------------------
    
    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                self.particles[f, i].F_tmp = (ti.Matrix.identity(us.FTYPE_TI, 3) + self.substep_dt * self.particles[f, i].C) @ self.particles[f, i].F

    @ti.kernel
    def svd(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                self.particles[f, i].U, self.particles[f, i].S, self.particles[f, i].V = ti.svd(self.particles[f, i].F_tmp, us.FTYPE_TI)

    @ti.func
    def clamp(self, a):
        if a>=0:
            a = ti.max(a, 1e-8)
        else:
            a = ti.min(a, -1e-8)
        return a

    @ti.kernel
    def advect_active(self, f: ti.i32):
        for i in range(self.n_particles_max):
            self.particles_ng[f+1, i].active = self.particles_ng[f, i].active

    @ti.kernel
    def advect_inactive_particles(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active == 0:
                self.particles[f+1, i].vel = self.particles[f, i].vel
                self.particles[f+1, i].pos = self.particles[f, i].pos
                self.particles[f+1, i].C   = self.particles[f, i].C
                self.particles[f+1, i].F   = self.particles[f, i].F

    @ti.func
    def stencil_range(self):
        return ti.ndrange(3, 3, 3)

    @ti.kernel
    def p2g(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                base = ti.floor(self.particles[f, i].pos * self.inv_dx - 0.5).cast(ti.i32)
                fx   = self.particles[f, i].pos * self.inv_dx - base.cast(us.FTYPE_TI)
                w    = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

                J = self.particles[f, i].S.determinant()

                stress = ti.Matrix.zero(us.FTYPE_TI, 3, 3)
                for mat_idx in ti.static(self.mat_idxs):
                    if self.particles_info[i].mat_idx == mat_idx:
                        stress = self.mat_models[mat_idx][0](
                            U     = self.particles[f, i].U,
                            S     = self.particles[f, i].S,
                            V     = self.particles[f, i].V,
                            F_tmp = self.particles[f, i].F_tmp,
                            mu    = self.particles_info[i].mu,
                            lam   = self.particles_info[i].lam,
                            J     = J,
                        )

                stress = (-self.substep_dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
                affine = stress + self.particles_info[i].mass * self.particles[f, i].C

                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(us.FTYPE_TI) - fx) * self.dx
                    weight = ti.cast(1.0, us.FTYPE_TI)
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]

                    self.grid[f, base - self.grid_offset + offset].vel_in += weight * (self.particles_info[i].mass * self.particles[f, i].vel + affine @ dpos)
                    self.grid[f, base - self.grid_offset + offset].mass += weight * self.particles_info[i].mass

                # update deformation gradient based on material type
                F_new = ti.Matrix.zero(us.FTYPE_TI, 3, 3)
                for mat_idx in ti.static(self.mat_idxs):
                    if self.particles_info[i].mat_idx == mat_idx:
                        F_new = self.mat_models[mat_idx][1](
                            J     = J,
                            F_tmp = self.particles[f, i].F_tmp,
                            U     = self.particles[f, i].U,
                            S     = self.particles[f, i].S,
                            V     = self.particles[f, i].V,
                        )

                self.particles[f+1, i].F = F_new

    @ti.kernel
    def grid_op(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.grid_res)):
            if self.grid[f, I].mass > us.EPS:
                vel_out = (1 / self.grid[f, I].mass) * self.grid[f, I].vel_in  # Momentum to velocity
                vel_out += self.substep_dt * self.gravity # gravity

                # collide with tool entities
                vel_out = self.sim.tool_solver.grid_collide(f, (I+self.grid_offset)*self.dx, vel_out, self.substep_dt)

                # impose boundary
                _, self.grid[f, I].vel_out = self.boundary.impose_pos_vel((I+self.grid_offset)*self.dx, vel_out)

    @ti.kernel
    def g2p(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                base = ti.floor(self.particles[f, i].pos * self.inv_dx - 0.5).cast(ti.i32)
                fx = self.particles[f, i].pos * self.inv_dx - base.cast(us.FTYPE_TI)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_vel = ti.Vector.zero(us.FTYPE_TI, 3)
                new_C = ti.Matrix.zero(us.FTYPE_TI, 3, 3)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(us.FTYPE_TI) - fx
                    grid_vel = self.grid[f, base - self.grid_offset + offset].vel_out
                    weight = ti.cast(1.0, us.FTYPE_TI)
                    for d in ti.static(range(3)):
                        weight *= w[offset[d]][d]
                    new_vel += weight * grid_vel
                    new_C += 4 * self.inv_dx * weight * grid_vel.outer_product(dpos)

                # collide with tool entities
                new_vel = self.sim.tool_solver.particle_collide(f, self.particles[f, i].pos, new_vel, self.substep_dt)

                # compute actual new_pos with new_vel
                new_pos = self.particles[f, i].pos + self.substep_dt * new_vel

                # impose boundary (for safety only)
                new_pos, new_vel = self.boundary.impose_pos_vel(new_pos, new_vel)

                # advect to next frame    
                self.particles[f+1, i].vel = new_vel
                self.particles[f+1, i].C   = new_C
                self.particles[f+1, i].pos = new_pos
    
    @ti.kernel
    def collision_info(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                info = ti.Vector.zero(us.FTYPE_TI, 3)
                info = self.sim.tool_solver.collision_info(f, self.particles[f, i].pos, info)
                self.particles[f, i].collision_info = info

    @ti.kernel
    def collision_info_gripper(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                # check both fingers
                info = ti.Vector.zero(us.FTYPE_TI, 6)
                info = self.sim.tool_solver.collision_info_gripper(f, self.particles[f, i].pos, info)
                self.particles[f, i].collision_info_gripper = info

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def process_input(self):
        for entity in self.entities:
            entity.process_input()

    def substep_pre_coupling(self, f):
        if self.is_active():
            if not self.use_sparse_grid:
                self.reset_grid(f)
            self.advect_active(f)
            self.compute_F_tmp(f)
            self.svd(f)
            self.p2g(f)

    def substep_coupling(self, f):
        if self.is_active():
            self.grid_op(f)
            self.g2p(f)

    def substep_post_coupling(self, f):
        pass

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles_max):
            self.particles[target, i].pos = self.particles[source, i].pos
            self.particles[target, i].vel = self.particles[source, i].vel
            self.particles[target, i].F   = self.particles[source, i].F
            self.particles[target, i].C   = self.particles[source, i].C
            self.particles_ng[target, i].active = self.particles_ng[source, i].active

    @ti.kernel
    def reset_grid(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.grid_res)):
            self.grid[f, I].vel_in  = 0
            self.grid[f, I].mass    = 0
            self.grid[f, I].vel_out = 0

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------
    
    def save_ckpt(self):
        if self.is_active():
            # restart from frame 0 in memory
            if self.use_sparse_grid:
                self.deactivate_grid_block()
            self.copy_frame(self.sim.max_substeps_local, 0)

    @ti.kernel
    def get_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), active: ti.types.ndarray()):
        for i in range(self.n_particles_max):
            for j in ti.static(range(3)):
                pos[i, j] = self.particles[f, i].pos[j]
                vel[i, j] = self.particles[f, i].vel[j]
                for k in ti.static(range(3)):
                    C[i, j, k] = self.particles[f, i].C[j, k]
                    F[i, j, k] = self.particles[f, i].F[j, k]
            active[i] = self.particles_ng[f, i].active

    @ti.kernel
    def set_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), active: ti.types.ndarray()):
        for i in range(self.n_particles_max):
            for j in ti.static(range(3)):
                self.particles[f, i].pos[j] = pos[i, j]
                self.particles[f, i].vel[j] = vel[i, j]
                for k in ti.static(range(3)):
                    self.particles[f, i].C[j, k] = C[i, j, k]
                    self.particles[f, i].F[j, k] = F[i, j, k]
            self.particles_ng[f, i].active = active[i]

    def get_state(self, f):
        state = MPMSolverState(self.scene)
        if self.is_active():
            self.get_frame(f, state.pos, state.vel, state.C, state.F, state.active)
        return state

    def set_state(self, f, state):
        if self.is_active():
            self.set_frame(f, state.pos, state.vel, state.C, state.F, state.active)

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        for i in range(self.n_particles_max):
            if self.particles_ng[f, i].active:
                for j in ti.static(range(3)):
                    self.particles_render[i].pos[j] = ti.cast(self.particles[f, i].pos[j], ti.f32)
            else:
                # let's inject a bit of humor here
                self.particles_render[i].pos = ti.Vector([2333333, 6666666, 5201314])
            self.particles_render[i].active = ti.cast(self.particles_ng[f, i].active, ti.i32)
            self.particles_render[i].color = self.particles_info[i].color

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        return self.particles_render
