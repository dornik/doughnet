import taichi as ti
import mpm as us
from mpm.utils.misc import *
from .states.solvers import SimState
from .solvers import *
from collections import OrderedDict


@ti.data_oriented
class Simulator:
    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(
        self,
        scene,
        options,
        tool_options,
        mpm_options,
    ):
        self.scene = scene

        # options
        self.options       = options
        self.tool_options  = tool_options
        self.mpm_options   = mpm_options

        self.gravity            = ti.Vector(options.gravity)
        self.step_dt            = options.step_dt
        self.substep_dt         = options.substep_dt
        self.max_substeps_local = options.max_substeps_local

        # dependent parameters 
        self.n_substeps      = int(self.step_dt / self.substep_dt)
        self.max_steps_local = int(self.max_substeps_local / self.n_substeps)
        assert self.max_substeps_local % self.n_substeps == 0

        # solvers
        self.solvers = OrderedDict()
        self.solvers['tool_solver']  = self.tool_solver  = ToolSolver(self.scene, self, self.tool_options)
        self.solvers['mpm_solver']   = self.mpm_solver   = MPMSolver(self.scene, self, self.mpm_options)

    def build(self):
        # step
        self.cur_substep_global = 0

        # solvers
        for solver in self.solvers.values():
            solver.build()

    def reset(self, state):
        for solver_name, solver in self.solvers.items():
            solver.set_state(0, state[solver_name])

        self.cur_substep_global = 0

    # ------------------------------------------------------------------------------------
    # -------------------------------- step computation ----------------------------------
    # ------------------------------------------------------------------------------------
    '''
    We use f to represent substep, and s to represent step.
    '''
    def f_global_to_f_local(self, f_global):
        f_local = f_global % self.max_substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self.n_substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self.n_substeps
        return s_global

    @property
    def cur_substep_local(self):
        return self.f_global_to_f_local(self.cur_substep_global)

    @property
    def cur_step_local(self):
        return self.f_global_to_s_local(self.cur_substep_global)

    @property
    def cur_step_global(self):
        return self.f_global_to_s_global(self.cur_substep_global)

    # ------------------------------------------------------------------------------------
    # ------------------------------------ stepping --------------------------------------
    # ------------------------------------------------------------------------------------

    def step(self):
        self.step_()

        if self.cur_substep_local == 0:
            self.save_ckpt()

    def step_(self):
        self.process_input()

        for i in range(0, self.n_substeps):
            self.substep(self.cur_substep_local)
            self.cur_substep_global += 1

    def process_input(self):
        ''' 
        setting _tgt state using external commands
        note that external inputs are given at step level, not substep
        '''
        for solver in self.solvers.values():
            solver.process_input()

    def substep(self, f):
        self.substep_pre_coupling(f)
        self.substep_coupling(f)
        self.substep_post_coupling(f)

    #-------------- pre coupling --------------
    def substep_pre_coupling(self, f):
        for solver in self.solvers.values():
            solver.substep_pre_coupling(f)

    #-------------- coupling --------------
    def substep_coupling(self, f):
        for solver in self.solvers.values():
            solver.substep_coupling(f)

    #-------------- post coupling --------------
    def substep_post_coupling(self, f):
        for solver in self.solvers.values():
            solver.substep_post_coupling(f)

    # ------------------------------------------------------------------------------------
    # --------------------------------------- io -----------------------------------------
    # ------------------------------------------------------------------------------------

    def save_ckpt(self):
        ckpt_start_step = self.cur_substep_global - self.max_substeps_local
        ckpt_end_step = self.cur_substep_global - 1

        for solver in self.solvers.values():
            solver.save_ckpt()

        us.logger.debug(f'Forward: Saved checkpoint for global substep {ckpt_start_step} to {ckpt_end_step}. Now starts from substep {self.cur_substep_global}.')

    def get_state(self):
        state = SimState(
            scene        = self.scene,
            s_global     = self.cur_step_global,
            tool_solver  = self.tool_solver.get_state(self.cur_substep_local),
            mpm_solver   = self.mpm_solver.get_state(self.cur_substep_local),
        )

        return state
    
    def set_state(self, state):
        self.cur_substep_global = 0
        for solver_name, solver in self.solvers.items():
            solver.set_state(0, state[solver_name])

    def collision_info(self):
        self.mpm_solver.collision_info(self.cur_substep_local)
        return self.mpm_solver.particles.collision_info.to_numpy()[self.cur_substep_local]
    
    def collision_info_gripper(self):
        self.mpm_solver.collision_info_gripper(self.cur_substep_local)
        return self.mpm_solver.particles.collision_info_gripper.to_numpy()[self.cur_substep_local]

    @property
    def entities(self):
        entities = []
        for solver in self.solvers.values():
            entities += solver.entities
        return entities
