import mpm as us
import taichi as ti
from .particle_entity import ParticleEntity
from mpm.engine.states.entities import MPMEntityState


@ti.data_oriented
class MPMEntity(ParticleEntity):
    '''
    MPM-based particle entity.
    '''
    def init_tgt_keys(self):
        self._tgt_keys = ['vel', 'pos', 'act', 'vel_masked', 'col']

    @ti.kernel
    def add_to_solver_kernel(
            self,
            f     : ti.i32,
            pos   : ti.types.ndarray(),
            color : ti.types.ndarray(),
        ):
        for i in range(self.n):
            i_global = i + self.idx_offset
            for j in ti.static(range(3)):
                self.solver.particles[f, i_global].pos[j] = pos[i, j]
            self.solver.particles[f, i_global].vel  = ti.Vector.zero(us.FTYPE_TI, 3)
            self.solver.particles[f, i_global].F    = ti.Matrix.identity(us.FTYPE_TI, 3)
            self.solver.particles[f, i_global].C    = ti.Matrix.zero(us.FTYPE_TI, 3, 3)
            
            self.solver.particles_ng[f, i_global].active = 1

            for j in ti.static(range(4)):
                self.solver.particles_info[i_global].color[j] = color[j]
            self.solver.particles_info[i_global].mat_idx = self.material.idx
            self.solver.particles_info[i_global].mu      = self.material.mu
            self.solver.particles_info[i_global].lam     = self.material.lam
            self.solver.particles_info[i_global].mass    = self.solver.p_vol * self.material.rho

    @ti.kernel
    def set_vel(self, f: ti.i32, vel: ti.types.ndarray()):
        for i in range(self.n):
            i_global = i + self.idx_offset
            for k in ti.static(range(3)):
                self.solver.particles[f, i_global].vel[k] = vel[i, k]

    @ti.kernel
    def set_vel_masked(self, f: ti.i32, vel_masked: ti.types.ndarray()):
        for i in range(self.n):
            i_global = i + self.idx_offset
            if vel_masked[i, 3] == 1:  # last entry is mask -- only update if 1
                for k in ti.static(range(3)):
                    self.solver.particles[f, i_global].vel[k] = vel_masked[i, k]
            else:  # attenuate current velocity by factor in mask
                for k in ti.static(range(3)):
                    self.solver.particles[f, i_global].vel[k] = self.solver.particles[f, i_global].vel[k] * vel_masked[i, k]

    @ti.kernel
    def set_pos(self, f: ti.i32, pos: ti.types.ndarray()):
        for i in range(self.n):
            i_global = i + self.idx_offset
            for k in ti.static(range(3)):
                self.solver.particles[f, i_global].pos[k] = pos[i, k]

            # we restore these whenever directly setting positions
            self.solver.particles[f, i_global].vel = ti.Vector.zero(us.FTYPE_TI, 3)
            self.solver.particles[f, i_global].F   = ti.Matrix.identity(us.FTYPE_TI, 3)
            self.solver.particles[f, i_global].C   = ti.Matrix.zero(us.FTYPE_TI, 3, 3)

    @ti.kernel
    def set_active(self, f: ti.i32, active: ti.i32):
        for i in range(self.n):
            i_global = i + self.idx_offset
            self.solver.particles_ng[f, i_global].active = active

    @ti.kernel
    def set_col(self, f: ti.i32, colors: ti.types.ndarray()):
        for i in range(self.n):
            i_global = i + self.idx_offset
            for k in ti.static(range(4)):
                self.solver.particles_info[i_global].color[k] = colors[i, k]

    def process_input(self):

        # set_pos followed by set_vel, because set_pos resets velocity.
        if self._tgt['pos'] is not None:
            us.assert_contiguous(self._tgt['pos'])
            self.set_pos(self.sim.cur_substep_local, self._tgt['pos'])

        if self._tgt['vel'] is not None:
            us.assert_contiguous(self._tgt['vel'])
            self.set_vel(self.sim.cur_substep_local, self._tgt['vel'])
        if self._tgt['vel_masked'] is not None:
            self.set_vel_masked(self.sim.cur_substep_local, self._tgt['vel_masked'])

        if self._tgt['act'] is not None:
            assert self._tgt['act'] in [us.ACTIVE, us.INACTIVE]
            self.set_active(self.sim.cur_substep_local, self._tgt['act'])

        if self._tgt['col'] is not None:
            us.assert_contiguous(self._tgt['col'])
            self.set_col(self.sim.cur_substep_local, self._tgt['col'])
            
        for key in self._tgt_keys:
            self._tgt[key] = None

    @ti.kernel
    def get_frame(self, f: ti.i32, pos: ti.types.ndarray(), vel: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray()):
        for i in range(self.n):
            i_global = i + self.idx_offset
            for j in ti.static(range(3)):
                pos[i, j] = self.solver.particles[f, i_global].pos[j]
                vel[i, j] = self.solver.particles[f, i_global].vel[j]
                for k in ti.static(range(3)):
                    C[i, j, k] = self.solver.particles[f, i_global].C[j, k]
                    F[i, j, k] = self.solver.particles[f, i_global].F[j, k]

    def get_state(self):
        state = MPMEntityState(self, self.sim.cur_step_global)
        self.get_frame(self.sim.cur_substep_local, state.pos, state.vel, state.C, state.F)

        return state