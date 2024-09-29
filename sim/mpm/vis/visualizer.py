import taichi as ti
import mpm as us
from .viewer import Viewer
from .renderer import Renderer


@ti.data_oriented
class Visualizer:

    def __init__(self, viewer_options, renderer_options):

        self.viewer = Viewer(viewer_options)

        if isinstance(renderer_options, us.options.renderers.RendererOptions):
            self.renderer = Renderer(renderer_options)
        else:
            raise NotImplementedError

    def add_camera(self, res, pos, lookat, up, fov):
        return self.renderer.add_camera(res, pos, lookat, up, fov)

    def build(self, scene):
        self.viewer.build(scene)
        self.renderer.build(scene)

