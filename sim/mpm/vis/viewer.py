import numpy as np
import mpm as us
import taichi as ti


@ti.data_oriented
class Viewer:
    def __init__(self, options):
        self.res             = options.res
        self.camera_pos      = options.camera_pos
        self.camera_lookat   = options.camera_lookat
        self.camera_up       = options.camera_up
        self.camera_fov      = options.camera_fov
        self.vsync           = options.vsync
        self.particle_radius = options.particle_radius
        
        self.lights = []
        for light in options.lights:
            self.add_light(light['pos'], light['color'])

        self.init_window()

    def add_light(self, pos, color=(0.5, 0.5, 0.5)):
        light = {
            'pos': pos,
            'color': color
        }
        self.lights.append(light)

    def build(self, scene):
        self.sim = scene.sim

        # xyz frame
        self.axis_length = 0.15
        self.axis_n_particles = 50
        self.frames = [ti.Vector.field(3, dtype=ti.f32, shape=self.axis_n_particles),
                        ti.Vector.field(3, dtype=ti.f32, shape=self.axis_n_particles),
                        ti.Vector.field(3, dtype=ti.f32, shape=self.axis_n_particles)]
        for i in range(self.axis_n_particles):
            self.frames[0][i] = ti.Vector([0., 0., 0.]) + i/self.axis_n_particles * ti.Vector([self.axis_length, 0., 0.])
            self.frames[1][i] = ti.Vector([0., 0., 0.]) + i/self.axis_n_particles * ti.Vector([0., self.axis_length, 0.])
            self.frames[2][i] = ti.Vector([0., 0., 0.]) + i/self.axis_n_particles * ti.Vector([0., 0., self.axis_length])

    def init_window(self):
        self.window = ti.ui.Window('unisim', self.res, vsync=self.vsync, show_window=False)

        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(*self.camera_pos)
        self.camera.lookat(*self.camera_lookat)
        self.camera.up(*self.camera_up)
        self.camera.fov(self.camera_fov)

    def update_scene(self):
        if self.sim.tool_solver.is_active():
            for tool_entity in self.sim.tool_solver.entities:
                if tool_entity.mesh is not None:
                    self.scene.mesh(tool_entity.mesh.vertices, tool_entity.mesh.faces, per_vertex_color=tool_entity.mesh.colors)
        if self.sim.mpm_solver.is_active():
            state = self.sim.mpm_solver.get_state_render(self.sim.cur_substep_local)
            self.scene.particles(state.pos, per_vertex_color=state.color, radius=self.particle_radius)
        for light in self.lights:
            self.scene.point_light(pos=light['pos'], color=light['color'])

    def update_camera(self):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

    def get_image(self):
        self.update_scene()
        self.update_camera()
        self.canvas.scene(self.scene)
        img = np.rot90(self.window.get_image_buffer_as_numpy())
        img = (img * 255).astype(np.uint8)
        return img
