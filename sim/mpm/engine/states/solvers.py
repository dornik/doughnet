import torch
import mpm as us
from mpm.utils.repr import _repr


class SimState:
    '''
    Dynamic state queried from a unisim Scene's Simulator.
    '''
    def __init__(self, scene, s_global, tool_solver, mpm_solver,):
        self.scene    = scene
        self.s_global = s_global
        
        self.tool_solver  = tool_solver
        self.mpm_solver   = mpm_solver

    def __getitem__(self, key):
        return eval(f'self.{key}')

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'scene        : {_repr(self.scene)}\n' \
               f's_global     : {_repr(self.s_global)}\n' \
               f'tool_solver  : {_repr(self.tool_solver)}\n' \
               f'mpm_solver   : {_repr(self.mpm_solver)}\n' \


class ToolSolverState:
    '''
    Dynamic state queried from a unisim RigidSolver.
    '''
    def __init__(self, scene):
        self.scene = scene
        self.data  = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'entity_states : {_repr(self.data)}'


class MPMSolverState:
    '''
    Dynamic state queried from a unisim MPMSolver.
    '''
    def __init__(self, scene):
        self.scene = scene

        self.pos    = None
        self.vel    = None
        self.C      = None
        self.F      = None
        self.active = None
        self.mat_id = None

        if scene.sim.mpm_solver.is_active():
            self.pos    = torch.zeros((scene.sim.mpm_solver.n_particles_max, 3), dtype=us.FTYPE_TC)
            self.vel    = torch.zeros((scene.sim.mpm_solver.n_particles_max, 3), dtype=us.FTYPE_TC)
            self.C      = torch.zeros((scene.sim.mpm_solver.n_particles_max, 3, 3), dtype=us.FTYPE_TC)
            self.F      = torch.zeros((scene.sim.mpm_solver.n_particles_max, 3, 3), dtype=us.FTYPE_TC)
            self.active = torch.zeros((scene.sim.mpm_solver.n_particles_max,), dtype=us.ITYPE_TC)
            self.mat_id = torch.zeros((scene.sim.mpm_solver.n_particles_max,), dtype=us.ITYPE_TC)

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'scene  : {_repr(self.scene)}\n' \
               f'pos    : {_repr(self.pos)}\n' \
               f'vel    : {_repr(self.v)}\n' \
               f'C      : {_repr(self.C)}\n' \
               f'F      : {_repr(self.F)}\n' \
               f'active : {_repr(self.active)}\n' \
               f'mat_id : {_repr(self.mat_id)}'
