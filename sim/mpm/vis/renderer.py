import numpy as np
import taichi as ti


@ti.data_oriented
class Renderer:
    def __init__(self, options):
        self.particle_radius = options.particle_radius
        self.lights = []
        for light in options.lights:
            self.add_light(light['pos'], light['color'])

        self.cameras = []

    def add_camera(self, res, pos, lookat, up, fov):
        camera = Camera(self, len(self.cameras), res, pos, lookat, up, fov)
        self.cameras.append(camera)
        return camera

    def add_light(self, pos, color=(0.5, 0.5, 0.5)):
        light = {
            'pos': pos,
            'color': color
        }
        self.lights.append(light)

    def build(self, scene):
        self.sim = scene.sim
        for camera in self.cameras:
            camera.build(self.sim)


@ti.data_oriented
class Camera:
    def __init__(
        self, 
        renderer,
        id     = 0,
        res    = (320, 320),
        pos    = (0.5, 2.5, 3.5),
        lookat = (0.5, 0.5, 0.5),
        up     = (0.0, 0.0, 1.0),
        fov    = 30,
    ):
        self.renderer = renderer
        self.id       = id
        self.res      = res
        self.pos      = pos
        self.lookat   = lookat
        self.up       = up
        self.fov      = fov
        
        self.init_window()

    def init_window(self):
        self.window = ti.ui.Window(f'Camera [{self.id}]', self.res, vsync=False, show_window=False)

        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(*self.pos)
        self.camera.lookat(*self.lookat)
        self.camera.up(*self.up)
        self.camera.fov(self.fov)

    def build(self, sim):
        self.sim = sim
        
    def update_scene(self):
        if self.sim.tool_solver.is_active():
            for tool_entity in self.sim.tool_solver.entities:
                if tool_entity.mesh is not None:
                    self.scene.mesh(tool_entity.mesh.vertices, tool_entity.mesh.faces, per_vertex_color=tool_entity.mesh.colors)
        if self.sim.mpm_solver.is_active():
            state = self.sim.mpm_solver.get_state_render(self.sim.cur_substep_local)
            self.scene.particles(state.pos, per_vertex_color=state.color, radius=self.renderer.particle_radius)
        for light in self.renderer.lights:
            self.scene.point_light(pos=light['pos'], color=light['color'])

    def update_camera(self):
        self.scene.set_camera(self.camera)

    def render(self):
        self.update_scene()
        self.update_camera()
        self.canvas.scene(self.scene)
        img = np.rot90(self.window.get_image_buffer_as_numpy())
        img = (img * 255).astype(np.uint8)
        return img
