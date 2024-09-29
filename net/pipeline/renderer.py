import numpy as np
import torch
import torch.nn as nn
import kaolin as kal
# import ninja
import nvdiffrast.torch as dr
from kornia.geometry.transform import remap, rescale
from kornia import morphology as morph
from kornia.utils import create_meshgrid
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
import math


def uniform_2_sphere_torch(shape, z_min=-1.0, z_max=1.0, radius_min=1.0, radius_max=1.0, device='cuda'):
    """Uniform sampling on a 2-sphere
    via https://gist.github.com/andrewbolster/10274979
    """

    phi = torch.rand(shape, device=device) * 2 * np.pi
    cos_theta = torch.rand(shape, device=device) * (z_max - z_min) + z_min
    radius = torch.rand(shape + (1,), device=device) * (radius_max - radius_min) + radius_min

    theta = torch.acos(cos_theta)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack([x, y, z], dim=-1) * radius


class Renderer(nn.Module):

    _device2glctx = {}

    def __init__(self, cfg, device) -> None:
        super().__init__()
        if cfg.render_downsample > 1:
            assert cfg.width % cfg.render_downsample == 0 and cfg.height % cfg.render_downsample == 0
            assert (cfg.width / cfg.render_downsample) % 8 == 0 and (cfg.height / cfg.render_downsample) % 8 == 0  # requirement of nvdiffrast
            # downsampled render resolution
            cfg.width = int(cfg.width / cfg.render_downsample)
            cfg.height = int(cfg.height / cfg.render_downsample)
            cfg.fx = cfg.fx / cfg.render_downsample
            cfg.fy = cfg.fy / cfg.render_downsample
            cfg.cx = cfg.cx / cfg.render_downsample
            cfg.cy = cfg.cy / cfg.render_downsample
        self.cfg = cfg

        self.glctx = Renderer._get_nvdiff_glctx(device)
        max_dist = cfg.max_dist * cfg.meter_to_unit
        self.camera_intrinsics = kal.render.camera.PinholeIntrinsics.from_focal(width=cfg.width, height=cfg.height,
                                                                                focal_x=cfg.fx, focal_y=cfg.fy,
                                                                                x0=cfg.cx - cfg.width/2, y0=cfg.cy - cfg.height/2,
                                                                                near=max_dist*0.5, far=max_dist*1.5,
                                                                                device=device)
        # get intrinsics in GL-style (for rendering --> operates on downsampled resolution)
        self.projection_matrix = self.camera_intrinsics.projection_matrix()
        # get intrinsics in CV-style (for depth to pcd --> operates on upsampled resolution ==> use original intrinsics (if downsampled))
        K = self.camera_intrinsics.perspective_matrix()[:, :3, :3]
        K[:, :2, :] *= cfg.render_downsample  # note: nop if not downsampled
        K[:, 0, 2] = -K[:, 0, 2] + (self.cfg.width/2)*self.cfg.render_downsample
        K[:, 1, 2] = -K[:, 1, 2] + (self.cfg.height/2)*self.cfg.render_downsample
        K[:, 2, 2] = 1
        self.inv_K = torch.inverse(K[:, :3, :3])
        # image coordinates (again, on upsampled resolution)
        y, x = torch.meshgrid(torch.arange(self.cfg.height*self.cfg.render_downsample, device=device),
                              torch.arange(cfg.width*self.cfg.render_downsample, device=device),
                              indexing='ij')
        x = x[..., None].float()
        y = y[..., None].float()
        self.xy1 = torch.cat([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)

    def forward(self, data, postfix, num_samples):
        verts, faces = data[f'obj_verts{postfix}'], data[f'obj_faces{postfix}']
        vert_labels = data[f'obj_vert_labels{postfix}']

        # # camera extrinsics
        # origins, camera_transform = self.get_random_extrinsics(len(verts), verts.device)

        # camera extrinsics
        camera_transform = np.eye(4, dtype=np.float32)
        camera_transform[:3, :3] = (Rotation.from_euler('zyx', [0, 0, 180], degrees=True) * Rotation.from_quat(self.cfg.cam_q_mean)).as_matrix()
        camera_transform[:3, 3] = np.array(self.cfg.cam_t_mean) * self.cfg.meter_to_unit
        camera_transform[:3, 3] = camera_transform[:3, :3] @ -camera_transform[:3, 3]
        camera_transform = torch.from_numpy(camera_transform).to(verts.device).float()[None, ...].repeat(len(verts), 1, 1)
        # add noise: translational offset from Gaussian, rotational offset from random rotvec and magnitude from Gaussian
        camera_transform[:, :3, 3] += torch.randn_like(camera_transform[:, :3, 3]) * self.cfg.cam_t_std * self.cfg.meter_to_unit
        rand_rot = special_ortho_group.rvs(3, size=len(verts)).astype(np.float32).reshape(-1, 3, 3)
        axis_angle = Rotation.as_rotvec(Rotation.from_matrix(rand_rot))
        axis_angle /= np.linalg.norm(axis_angle, axis=1)[:, None]
        axis_angle *= np.deg2rad(np.random.normal(0, self.cfg.cam_r_std, size=len(verts)))[:, None]
        rand_rot = Rotation.from_rotvec(axis_angle).as_matrix()
        camera_transform[:, :3, :3] = torch.from_numpy(rand_rot).to(verts.device).float() @ camera_transform[:, :3, :3]
        # to expected format
        camera_transform = camera_transform.transpose(2, 1)[:, :, :3]  # B x 4 x 3

        # also render plane (and ee fingers) to allow more realistic blurring/occlusion/etc
        verts_plane, faces_plane = torch.from_numpy(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)).to(verts.device), torch.from_numpy(np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32) + verts.shape[1]).to(verts.device)
        verts = torch.cat([verts, verts_plane.unsqueeze(0).expand(verts.shape[0], -1, -1)], dim=1)
        faces = torch.cat([faces, faces_plane.unsqueeze(0).expand(faces.shape[0], -1, -1)], dim=1)
        vert_labels = torch.cat([vert_labels, torch.zeros_like(vert_labels[:, :4])], dim=1)  # -> excluded from seg

        # render
        depth, seg = self.render(verts, faces, vert_labels, camera_transform)
        # add noise
        if self.cfg.perlin_noise:  # low frequency
            # noise at most [-1, 1]
            noise = torch.stack([Renderer.rand_perlin_2d(depth.shape[1:], (self.cfg.perlin_noise_res, self.cfg.perlin_noise_res), depth.device, fade=lambda t: 6*t**5 - 15*t**4 + 10*t**3)
                                 for _ in range(depth.shape[0])])
            noise = noise * self.cfg.perlin_noise_scale * self.cfg.meter_to_unit
            depth[seg>0] = depth[seg>0] + noise[seg>0]
        if self.cfg.image_noise:  # high frequency
            depth = self.add_kinect_azure_noise(depth, self.cfg.unit_to_meter)
        if self.cfg.render_downsample > 1:  # blur
            depth = rescale(depth.unsqueeze(1), float(self.cfg.render_downsample), interpolation='bilinear').squeeze(1)
            seg = rescale(seg.unsqueeze(1).float(), float(self.cfg.render_downsample), interpolation='nearest').squeeze(1).long()
        # from depth to pcd
        xyz_world = self.depth_to_pcd(depth, camera_transform)
        # from pcd to observation (augment/sample points)
        observation, empty = self.pcd_to_observation(xyz_world, seg, num_samples, len(verts), verts.device, self.cfg.voxel_size)
        return observation

    def get_random_extrinsics(self, count, device):
        origins = uniform_2_sphere_torch((count, 1),
                                         z_min=self.cfg.z_range[0], z_max=self.cfg.z_range[1],
                                         radius_min=self.cfg.distance_range[0], radius_max=self.cfg.distance_range[1],
                                         device=device).reshape(-1, 3)
        lookat = torch.zeros_like(origins)
        up = torch.zeros_like(origins)
        up[..., 0] = 1  # top-down camera, looking in -z direction towards origin
        return origins, kal.render.camera.generate_transformation_matrix(origins, lookat, up)
    
    def render(self, verts, faces, vert_labels, camera_transform):
        vertices_camera = kal.render.camera.up_to_homogeneous(verts) @ camera_transform
        vertices_clip = kal.render.camera.up_to_homogeneous(vertices_camera) @ self.projection_matrix.transpose(-2, -1)
        rast_out, _ = dr.rasterize(self.glctx, vertices_clip, faces[0].int(), resolution=[self.cfg.height, self.cfg.width])
        out, _ = dr.interpolate(torch.cat([vertices_camera[..., 2, None], vert_labels], dim=-1).contiguous(), rast_out, faces[0].int())
        depth = -out[..., 0]
        seg = torch.round(out[..., 1]).int()
        return depth, seg
    
    def depth_to_pcd(self, depth, camera_transform):
        # from depth to camera space
        b, h, w = depth.shape
        xyz_cam = self.xy1[None].repeat(b, 1, 1) @ self.inv_K.transpose(2, 1) * depth.view(b, h*w, 1)
        xyz_cam[..., 2] *= -1  # flip z

        # from camera to world space
        to_cam = torch.eye(4)[None].repeat(len(depth), 1, 1).to(depth.device)
        to_cam[..., :3] = camera_transform
        to_world = torch.inverse(to_cam)
        to_world = to_world[..., :3]
        xyz_world = kal.render.camera.up_to_homogeneous(xyz_cam) @ to_world
        return xyz_world

    @staticmethod
    def pcd_to_observation(xyz_world, seg, num_samples, batch_size, device, voxel_size):
        empty = []
        observation = torch.zeros((batch_size, num_samples, 4), device=device)  # B x N x 4 (xyz + label)
        for bi, (xyz_b, seg_b) in enumerate(zip(xyz_world, seg.view(batch_size, -1, 1))):
            pcd = xyz_b[seg_b.view(-1) > 0]  # remove invalid / non-object points
            # voxel downsample to same density
            pcd[:, :3] = torch.round(pcd[:, :3] / voxel_size) * voxel_size
            unique_xyz, unique_indices = torch.unique(pcd[:, :3], dim=0, return_inverse=True)
            unique_l = torch.zeros_like(unique_xyz[:, 0].int()).scatter_reduce(index=unique_indices,
                                                                               src=torch.round(seg_b[seg_b>0]).int(),
                                                                               reduce='amax', dim=0, include_self=False)
            pcd = torch.cat([unique_xyz, unique_l[..., None]], dim=-1)

            if pcd.shape[0] > num_samples:
                # random subsample to same number of points
                pcd = pcd[torch.randperm(pcd.shape[0])[:num_samples], :]
            elif pcd.shape[0] > 0:
                while pcd.shape[0] < num_samples:
                    # pad with random points from the pcd
                    pcd = torch.cat([pcd, pcd[torch.randperm(pcd.shape[0])[:(num_samples - pcd.shape[0])]]], dim=0)
            else:
                print(f'  No points in pcd for batch item {bi} -- setting observation to zeros')
                observation[bi] = torch.zeros((num_samples, 4), device=device)
                observation[bi, :, -1] = 1  # set to object
                empty += [True]
                continue
            empty += [False]
            observation[bi] = pcd
        return observation, empty

    @staticmethod
    def _get_nvdiff_glctx(device):
        # import ninja  # some bug where ninja is not imported/visible at first
        if device not in Renderer._device2glctx:
            Renderer._device2glctx[device] = dr.RasterizeCudaContext(device=device)
        return Renderer._device2glctx[device]

    @staticmethod
    def add_kinect_azure_noise(depth, unit_to_meter):
        """ adapted from https://github.com/DLR-RM/BlenderProc/blob/b6bc84b1d49b45ac2aa3dac8e831ad0fd0e00fca/blenderproc/python/postprocessing/PostProcessingUtility.py#L183
        Add noise, holes and smooth depth maps according to the noise characteristics of the Kinect Azure sensor.
        https://www.mdpi.com/1424-8220/21/2/413

        For further realism, consider to use the projection from depth to color image in the Azure Kinect SDK:
        https://docs.microsoft.com/de-de/azure/kinect-dk/use-image-transformation

        :param depth: Input depth image(s) in meters
        :param missing_depth_darkness_thres: uint8 gray value threshold at which depth becomes invalid, i.e. 0
        :return: Noisy depth image(s)
        """

        # smoothing at borders
        depth = Renderer.add_gaussian_shifts(depth, 0.25)

        # 0.5mm base noise, 1mm std noise @ 1m, 3.6mm std noise @ 3m
        depth += ((5/10000 + torch.clip((depth-0.5) * 1/1000, max=0)) * torch.randn(size=depth.shape, device=depth.device)) * unit_to_meter
        # Creates the shape of the kernel (equivalent to cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        kernel = torch.ones((3, 3), device=depth.device)

        # Applies the minimum filter with kernel NxN
        min_depth = morph.erosion(depth.unsqueeze(1), kernel)
        max_depth = morph.dilation(depth.unsqueeze(1), kernel)

        # missing depth at 0.8m min/max difference
        depth[torch.abs(min_depth-max_depth).squeeze(1) > 0.8] = 0

        return depth

    @staticmethod
    def add_gaussian_shifts(image, std: float = 0.5):
        """ adapted from https://github.com/DLR-RM/BlenderProc/blob/b6bc84b1d49b45ac2aa3dac8e831ad0fd0e00fca/blenderproc/python/postprocessing/PostProcessingUtility.py#L229
        Randomly shifts the pixels of the input depth image in x and y direction.

        :param image: Input depth image(s)
        :param std: Standard deviation of pixel shifts, defaults to 0.5
        :return: Augmented images
        """

        batches, rows, cols = image.shape
        gaussian_shifts = torch.randn(size=(batches, rows, cols, 2), device=image.device) * std

        grid = create_meshgrid(rows, cols, normalized_coordinates=False, device=image.device).expand(len(image), -1, -1, -1)
        grid_interp = grid + gaussian_shifts

        xp_interp = torch.clip(grid_interp[..., 0], 0.0, cols)
        yp_interp = torch.clip(grid_interp[..., 1], 0.0, rows)
        depth_interp = remap(image.unsqueeze(1), xp_interp, yp_interp, mode='bilinear').squeeze(1)

        return depth_interp

    @staticmethod
    def rand_perlin_2d(shape, res, device='cuda', fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        # via https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1).to(device)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
