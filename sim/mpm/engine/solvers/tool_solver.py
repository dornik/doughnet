import taichi as ti
import mpm as us
from mpm.utils.misc import *
from mpm.engine.entities import ToolEntity
from mpm.engine.boundaries import FloorBoundary
from mpm.engine.states.solvers import ToolSolverState


@ti.data_oriented
class ToolSolver:

    # ------------------------------------------------------------------------------------
    # --------------------------------- Initialization -----------------------------------
    # ------------------------------------------------------------------------------------

    def __init__(self, scene, sim, options):
        self.scene   = scene
        self.sim     = sim
        self.options = options

        # options
        self.step_dt      = options.step_dt
        self.substep_dt   = options.substep_dt
        self.floor_height = options.floor_height

        # ensure solver n_substeps is compatible with simulator
        assert int(self.step_dt / self.substep_dt) == self.sim.n_substeps

        self.entities = list()

        # boundary
        self.setup_boundary()

    def build(self):
        for entity in self.entities:
            entity.build()

    def setup_boundary(self):
        self.boundary = FloorBoundary(height=self.floor_height)

    def add_entity(self, material, geom, surface_options):
        entity = ToolEntity(
            scene           = self.scene,
            solver          = self,
            material        = material,
            geom            = geom,
            surface_options = surface_options,
        )
        self.entities.append(entity)
        return entity

    def get_state(self, f):
        state = ToolSolverState(self.scene)
        for entity in self.entities:
            state.data.append(entity.get_state(f))
        return state

    def set_state(self, f, state):
        assert len(state) == len(self.entities)
        for i, entity in enumerate(self.entities):
            entity.set_state(f, state[i])

    def process_input(self):
        for entity in self.entities:
            entity.process_input()

    def substep_pre_coupling(self, f):
        for entity in self.entities:
            entity.substep_pre_coupling(f)

    def substep_coupling(self, f):
        for entity in self.entities:
            entity.substep_coupling(f)

    def substep_post_coupling(self, f):
        for entity in self.entities:
            entity.substep_post_coupling(f)

    def save_ckpt(self):
        for entity in self.entities:
            entity.save_ckpt()

    def is_active(self):
        return len(self.entities) > 0

    @ti.func
    def grid_collide(self, f, pos_world, vel_mat, dt):
        for entity in ti.static(self.entities):
            if ti.static(entity.material.collision_type in ['grid', 'both']):
                vel_mat = entity.collide(f, pos_world, vel_mat, dt)
        return vel_mat

    @ti.func
    def particle_collide(self, f, pos_world, vel_mat, dt):
        for entity in ti.static(self.entities):
            if ti.static(entity.material.collision_type in ['particle', 'both']):
                new_pos_tmp = pos_world + dt * vel_mat
                vel_mat = entity.collide(f, new_pos_tmp, vel_mat, dt)
        return vel_mat

    @ti.func
    def collision_info(self, f, pos_world, info):
        for entity in ti.static(self.entities):
            if ti.static(entity.material.collision_type in ['particle', 'both']) and entity.name != 'ground':
                # info = [truncated signed distance, influence, {0... not in contact, 1... in contact}]
                info = entity.collision_info(f, pos_world, info)
        return info

    @ti.func
    def collision_info_gripper(self, f, pos_world, info_gripper):
        for entity in ti.static(self.entities):
            # info = [truncated signed distance, influence, {0... not in contact, 1... in contact}]
            info = ti.Vector.zero(us.FTYPE_TI, 3)
            if entity.name == 'finger_left':
                info_gripper[0:3] = entity.collision_info(f, pos_world, info)
            elif entity.name == 'finger_right':
                info_gripper[3:6] = entity.collision_info(f, pos_world, info)
        return info_gripper
