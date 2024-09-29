import uuid
import numpy as np
import taichi as ti
import mpm as us
from time import time
from mpm.vis import Visualizer
from mpm.engine import materials
from mpm.engine.simulator import Simulator
from mpm.utils.repr import _repr, _repr_uuid
from mpm.options.renderers import RendererOptions
from mpm.options.vis import ViewerOptions
from mpm.options.solvers import *


@ti.data_oriented
class Scene:
    '''
    Scene wraps all components in a simulation environment.
    '''    
    def __init__(
            self,
            sim_options      = SimOptions(),
            tool_options     = ToolOptions(),
            mpm_options      = MPMOptions(),
            viewer_options   = ViewerOptions(),
            renderer_options = RendererOptions(),
        ):

        self.id            = str(uuid.uuid4())
        self.t             = 0
        self.is_built      = False

        self.sim_options   = sim_options
        self.tool_options  = tool_options
        self.mpm_options   = mpm_options

        # merge options
        self.tool_options.copy_attributes_from(self.sim_options)
        self.mpm_options.copy_attributes_from(self.sim_options)

        # simulator
        self.sim = Simulator(
            scene         = self,
            options       = self.sim_options,
            tool_options  = self.tool_options,
            mpm_options   = self.mpm_options,
        )

        # materials
        self.mats = []

        # visualizer
        self.visualizer = Visualizer(
            viewer_options   = viewer_options,
            renderer_options = renderer_options,
        )

        # track FPS
        self.step_ts = []
        self.last_step_t = time()

        self._forward_ready   = False

        us.logger.info(f'Scene {_repr_uuid(self.id)} created.')

    def add_entity(self, material, geom, surface_options=us.options.SurfaceOptions()):
        if material.scene is not self:
            us.raise_exception(f'Scene mismatch: material belongs to Scene {_repr_uuid(material.scene.id)}; this is Scene {_repr_uuid(self.id)}.')

        if isinstance(material, materials.Tool):
            entity = self.sim.tool_solver.add_entity(material, geom, surface_options)

        elif isinstance(material, materials.ElastoPlastic):
            entity = self.sim.mpm_solver.add_entity(material, geom, surface_options)

        else:
            us.raise_exception('Material type not supported.')

        return entity

    def add_material(self, material):
        self.mats.append(material)
        material.scene = self
        return material

    def add_camera(
        self,
        res=(320, 320),
        pos=(0.5, 2.5, 3.5),
        lookat=(0.5, 0.5, 0.5),
        up=(0.0, 0.0, 1.0),
        fov=30,
    ):
        return self.visualizer.add_camera(res, pos, lookat, up, fov)

    def build(self):
        # simulator
        self.sim.build()

        # visualizer
        self.visualizer.build(self)

        # reset state
        self._reset()

        self.is_built = True
        us.logger.info(f'Scene {_repr_uuid(self.id)} built.')

    def _reset(self, state=None):
        if self.is_built:
            if state is None:
                state = self._init_state
            self.sim.reset(state)

        self.t = 0
        self._forward_ready = True

        # update _init_state
        self._init_state = self.get_state()

    def reset(self, state=None):
        us.logger.debug(f'Resetting Scene {_repr_uuid(self.id)}.')
        self._reset(state)

    def get_state(self):
        state = self.sim.get_state()
        return state
    
    def set_state(self, state):
        self.sim.set_state(state)

    def step(self, show_FPS=False):
        if not self._forward_ready:
            us.raise_exception('Forward simulation not allowed after backward pass. Please reset scene state.')

        if not self.is_built:
            us.raise_exception('Scene is not built yet.')

        self.sim.step()
        
        self.t += 1
        cur_step_t = time()
        self.step_ts.append(cur_step_t - self.last_step_t)
        self.last_step_t = cur_step_t
        self.step_ts = self.step_ts[-100:]
        if show_FPS:
            # drop first few frames
            us.logger.debug(f'Step: {self.t}, Realtime FPS: {1.0 / np.mean(self.step_ts[max(0, min(3, len(self.step_ts) - 3)):]):.2f}')

    def get_viewer_image(self):
        return self.visualizer.viewer.get_image()

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'id: {_repr_uuid(self.id)}'
