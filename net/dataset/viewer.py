import taichi as ti
import numpy as np
import torch
import matplotlib as mpl


@ti.data_oriented
class Viewer:

    def __init__(self,
                res    = (640, 640),
                pos    = (0.0, 0.0, 1.0),
                lookat = (0.0, 0.0, 0.0),
                up     = (0.0, 1.0, 0.0),
                fov    = 60,
                particle_radius=0.0025,
                lights=[{'pos': (0.5, 2.5, 3.5), 'color': (0.5, 0.5, 0.5)}],
                show_window=False,
                ):
        self.res = res
        self.pos = pos
        self.lookat = lookat
        self.up = up
        self.fov = fov
        self.particle_radius = particle_radius
        self.lights = lights
        self.show_window = show_window
        self.cm = mpl.colormaps['Set2']
        self.init_window()

    def init_window(self):
        self.window = ti.ui.Window('Viewer', self.res, vsync=False,
                                   show_window=self.show_window, fps_limit=30)

        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((0.8, 0.8, 0.8))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(*self.pos)
        self.camera.lookat(*self.lookat)
        self.camera.up(*self.up)
        self.camera.fov(self.fov)

    def add_particles(self, pos, color=(0.5, 0.5, 0.5), labels=None):
        ti_pos = ti.Vector.field(3, ti.f32, (len(pos),))
        ti_pos.from_torch(pos)
        if labels is not None:
            ti_colors = ti.Vector.field(3, ti.f32, len(labels))
            ti_colors.from_numpy(self.cm(labels.cpu()).astype(np.float32)[..., :3].reshape(-1, 3))
        else:
            ti_colors = None
        self.scene.particles(ti_pos, color=color, per_vertex_color=ti_colors, radius=self.particle_radius)
    
    def add_mesh(self, vertices, faces, color=(0.5, 0.5, 0.5), vert_labels=None, wireframe=False):
        ti_vertices = ti.Vector.field(3, ti.f32, (len(vertices),))
        ti_vertices.from_torch(vertices)
        ti_faces = ti.field(ti.i32, faces.flatten().shape)
        ti_faces.from_torch(faces.flatten())
        if vert_labels is not None:
            ti_colors = ti.Vector.field(3, ti.f32, len(vert_labels))
            ti_colors.from_numpy(self.cm(vert_labels.cpu()).astype(np.float32)[..., :3].reshape(-1, 3))
        else:
            ti_colors = None
        self.scene.mesh(ti_vertices, ti_faces, color=color, per_vertex_color=ti_colors, show_wireframe=wireframe)

    def update_scene(self):
        for light in self.lights:
            self.scene.point_light(pos=light['pos'], color=light['color'])
    
    def update_camera(self):
        self.scene.set_camera(self.camera)

    def render(self):
        self.update_scene()
        self.update_camera()
        self.canvas.scene(self.scene)
    
    def get_image(self):
        img = np.rot90(self.window.get_image_buffer_as_numpy())
        img = (img * 255).astype(np.uint8)
        return img


if __name__ == '__main__':
    SHOW = True

    ti.init(arch=ti.gpu, debug=False)
    viewer = Viewer(show_window=SHOW)

    import hdf5plugin
    import h5py
    with h5py.File('/home/dominik/papers/por/data/dataset.h5', 'r', libver='latest') as f:
        # data = {k: torch.tensor(f['val_test'][k][:2, ::5]) for k in f['val_test'].keys()}
        data = {k: torch.tensor(f['real'][k][30:32, ::]) for k in f['real'].keys()}
    # with h5py.File('/home/dominik/papers/por/data/camera_ready/data.h5', 'r', libver='latest') as f:
    #     data = {k: torch.tensor(f[k][:2, ::5]) for k in f.keys()}

    scene, frame = 0, 0
    for frame in range(data['scene'][scene].shape[0]):
        # viewer.add_particles(data['ee_observed'][scene, frame], color=(0.8, 0.0, 0.8))
        viewer.add_mesh(
            vertices=data['ee_verts'][scene, frame],
            faces=data['ee_faces'][scene, frame],
            color=(0.8, 0.0, 0.8),
            wireframe=True,
        )
        if 'obj_observed' in data:  # note: labels for real do not show topology changes
            viewer.add_particles(data['obj_observed'][scene, frame, :, :3],
                                labels=data['obj_observed'][scene, frame, :, 3].int())
        else:
            viewer.add_mesh(
                vertices=data['obj_verts'][scene, frame],
                faces=data['obj_faces'][scene, frame],
                vert_labels=data['obj_vert_labels'][scene, frame],
                wireframe=False,
            )
        viewer.render()

        if not SHOW:
            import matplotlib.pyplot as plt
            plt.imshow(viewer.get_image())
            plt.show()
        else:
            viewer.window.show()
            # plt.pause(0.1)
