import torch
import torch.nn as nn
from kaolin.ops.mesh import sample_points as sample_mesh
from kaolin.ops.mesh import check_sign, index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import numpy as np

from net.model.model import Predictor
from net.pipeline.renderer import Renderer
from net.pipeline.evaluater import Evaluater


class Pipeline(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.renderer = Renderer(config.augmentation.render, device='cuda')
        self.model = Predictor(config).cuda()
        self.evaluator = Evaluater(config.loss).cuda()

    def forward(self, data, z_rec_cur=None, val_test=False):

        # == OBSERVATION ==
        input_dim = self.config.dimensions.input_dim
        num_samples = self.config.dataset.n_points
        for postfix in ['', '_nxt', '_nxtnxt'][:(self.config.dataset.next_frames+1)]:
            if f'obj_verts{postfix}' not in data and f'obj_observed{postfix}' not in data:  # only prepare what is needed
                continue
            # obj: always partial, noisy (from single camera view, either real or synthetic)
            if f'obj_observed{postfix}' in data:  # already prepared = real (or state=points ablation)
                obs = data[f'obj_observed{postfix}']
                obs_voxelized, _ = self.renderer.pcd_to_observation(obs[..., :3], obs[..., 3], num_samples,
                                                                        obs.shape[0], obs.device, self.config.augmentation.render.voxel_size)
                data[f'obj_observed{postfix}'] = obs_voxelized[..., :input_dim]
            else:  # synthetic
                # obj: create incomplete and noisy observation by rendering GT mesh to get a noisy, incomplete observation
                obs = self.renderer(data, postfix=postfix, num_samples=num_samples)
                data[f'obj_observed{postfix}'] = obs[..., :input_dim]
            
            # ee: always complete, noise free ("we know what we want to do")
            if f'ee_observed{postfix}' in data:  # already prepared = real
                pass
            else:
                data[f'ee_observed{postfix}'] = sample_points(data, 'ee', postfix, num_samples)

        # == QUERY POINTS ==
        # -- same query (i.e., where we evaluate the latent shape) for all steps --

        if val_test:  # use a dense grid
            resolution = 0.01  # 0.02 = 7mb, 0.01 = 11mb for visualization of one scene (with about 70 frames)
            bounds = torch.from_numpy(np.array([[-0.30, 0.30], [-0.52, 0.52], [-resolution/2, 0.11 + resolution/2]])).to(data['obj_observed'].device).float()
            steps = (bounds[:, 1] - bounds[:, 0])/resolution
            steps = steps.int() + 1
            # num_query = np.prod(steps)

            coords = [torch.linspace(*bounds[i], steps[i], device=data['obj_observed'].device) for i in range(3)]
            grid_coords = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)  # -> X x Y x Z x 3

            query = grid_coords.reshape(1, -1, 3).expand(data['obj_observed'].shape[0], -1, -1)  # -> N x 3
        else:
            bounds = torch.from_numpy(np.array([[-0.35, 0.35], [-0.55, 0.55], [-0.05, 0.15]])).to(data['obj_observed'].device).float() * self.config.augmentation.scale[1]
            num_query_box = self.config.dimensions.num_query_out // 2
            num_query_vox = self.config.dimensions.num_query_out - num_query_box

            # box queries
            query_box = torch.rand((data['obj_observed'].shape[0], num_query_box, 3), device=data['obj_observed'].device) * 2 - 1  # [-1,1]^3
            query_box = query_box * (bounds[:, 1] - bounds[:, 0]) / 2 + (bounds[:, 1] + bounds[:, 0]) / 2  # -> N x num_query_box x 3

            # add vox queries (get better coverage of ROI)
            voxel_size = self.config.augmentation.render.voxel_size * 2
            padding_size = self.config.augmentation.render.voxel_size * 10
            query_vox = voxelize_to_ground(data['obj_observed'], voxel_size, padding_size, num_samples=num_query_vox)
            query = torch.cat([query_box, query_vox], dim=1)
        data['query'] = query

        # == PREDICTION: Reconstruction (and Dynamics Prediction) ==
        loss, loss_dict = 0.0, {}
        acc, acc_dict = 0.0, {}
        log = [data, loss_dict, acc_dict]

        do_train_rec = not self.config.dataset.next_frames > 1 
        with torch.set_grad_enabled(do_train_rec and not val_test):
            # assumes that we always do pretrain -> resume, hence don't need gradients for reconstruct
            # - reconstructions
            if self.config.loss.get_rec_cur:
                # reconstruct current (note: if z_rec_cur is given, i.e., this is a continuous step, we don't recompute z -- just decode)
                z_rec_cur, log = self.reconstruct(z_rec_cur, *log, '', decode=do_train_rec)
            if self.config.loss.get_rec_nxt:
                # reconstruct next (from nxt observation)
                z_rec_nxt, log = self.reconstruct(None, *log, '_nxt', decode=do_train_rec)
            if self.config.loss.get_rec_nxtnxt:
                # reconstruct next-next (from nxtnxt observation)
                z_rec_nxtnxt, log = self.reconstruct(None, *log, '_nxtnxt', decode=do_train_rec)
        # - predictions: fwd
        if self.config.loss.get_pre_nxt:
            # predict next (from reconstruct current)
            z_pre_nxt, log = self.predict(z_rec_cur, *log, '_nxt')
        if self.config.loss.get_pre_nxtnxt:
            # predict next-next (from predict next = autoregressive)
            z_pre_nxtnxt, log = self.predict(z_pre_nxt, *log, '_nxtnxt')
        data, loss_dict, acc_dict = log
        # - predictions: pairwise loss between reconstruction and prediction for corresponding time step
        if self.config.loss.w_lat_nxt > 0:
            loss_dict['lat_loss_nxt'] = loss_tuple(z_pre_nxt, z_rec_nxt, metric=self.config.loss.lat_metric)
        if self.config.loss.w_lat_nxtnxt > 0:
            loss_dict['lat_loss_nxtnxt'] = loss_tuple(z_pre_nxtnxt, z_rec_nxtnxt, metric=self.config.loss.lat_metric)

        # - aggregate losses
        for prefix in ['rec', 'pre', 'lat']:
            for postfix in ['', '_nxt', '_nxtnxt'][:(self.config.dataset.next_frames+1)]:
                if f'{prefix}_loss{postfix}' not in loss_dict:
                    continue
                w_postfix = '_cur' if postfix == '' else postfix
                loss += loss_dict[f'{prefix}_loss{postfix}'] * getattr(self.config.loss, f'w_{prefix}{w_postfix}')
        # - aggregate accuracies
        for k in self.config.loss.acc_keys:
            if k not in acc_dict:
                raise ValueError(f'requested accuracy key {k} not computed')
            acc += acc_dict[k]

        return (loss, acc, loss_dict, acc_dict), data
    
    # @profile
    def reconstruct(self, z, data, loss_dict=None, acc_dict=None, postfix='', decode=True):
        # reconstruction
        logits_rec, logits_g, z_rec = self.model.reconstruct(data[f'obj_observed{postfix}'], data['query'], z=z, decode=decode)
        if decode:
            if f'obj_verts{postfix}' in data:
                # true label
                label_rec = get_label(data['query'], data[f'obj_verts{postfix}'], data[f'obj_faces{postfix}'], data[f'obj_face_labels{postfix}'])
                # loss/accuracy
                loss_dict, loss_best_perm = self.evaluator.loss(logits_rec, label_rec, logits_g, data[f'genus{postfix}'],
                                                                'rec', postfix, loss_dict)

                # bookkeeping
                data[f'obj_true_part{postfix}'] = label_rec
                data[f'best_perm_loss{postfix}'] = loss_best_perm
            # else:  # real data -- no GT mesh, so no true label and hence no loss/accuracy -- not permutated
            data[f'obj_predicted_part{postfix}'] = logits_rec
            # shift such that class 0->-1 indicates an "empty" component -- not permutated
            data[f'predicted_genus{postfix}'] = torch.argmax(logits_g, dim=-1) - 1
            # accuracy
            data[f'ee_true_part{postfix}'] = get_label(data['query'], data[f'ee_verts{postfix}'], data[f'ee_faces{postfix}'],
                                                       torch.ones_like(data[f'ee_faces{postfix}'][..., 0, None]))
            acc_dict, acc_best_perm = self.evaluator.accuracy(data, 'rec', postfix, acc_dict)
            data[f'best_perm_acc{postfix}'] = acc_best_perm
        data[f'z_rec{postfix}'] = z_rec.detach()

        return z_rec, (data, loss_dict, acc_dict)

    def predict(self, z, data, loss_dict=None, acc_dict=None, postfix='_nxt', decode=True):       
        # get ee
        ee_postfix = '' if postfix == '_nxt' else '_nxt'
        if self.config.model.condition.ee_shape == 'both':
            ee_observed = data[f'ee_observed{ee_postfix}'][..., :self.config.dimensions.input_dim]
            ee_target = data[f'ee_target{ee_postfix}'][..., :self.config.dimensions.input_dim]
            ee_observed = torch.stack([ee_observed, ee_target])
        else:
            ee_observed = data[f'ee_{self.config.model.condition.ee_shape}{ee_postfix}'][..., :self.config.dimensions.input_dim]

        # input_dim (= zero label, same mlp as observation) or with 3 (= no label, same mlp as query)
        if ee_observed.shape[-1] == 3:
            ee_label = torch.zeros_like(ee_observed[..., 0, None])  # B x N x 1
            ee_observed = torch.cat([ee_observed, ee_label], dim=-1)  # B x N x 4

        # prediction
        logits_pre, logits_g, z_pre = self.model.predict(ee_observed, data['query'], z, decode=decode)

        if decode:
            if f'obj_verts{postfix}' in data:
                # true label
                label_pre = get_label(data['query'], data[f'obj_verts{postfix}'], data[f'obj_faces{postfix}'], data[f'obj_face_labels{postfix}'])
                # loss
                loss_dict, loss_best_perm = self.evaluator.loss(logits_pre, label_pre, logits_g, data[f'genus{postfix}'],
                                                                'pre', postfix, loss_dict)
                data[f'obj_true_part{postfix}'] = label_pre
                data[f'best_perm_loss{postfix}'] = loss_best_perm
            # else:  # real data -- no GT mesh, so no true label and hence no loss/accuracy -- not permutated
            data[f'obj_predicted_part{postfix}'] = logits_pre
            # shift such that class 0->-1 indicates an "empty" component -- not permutated
            data[f'predicted_genus{postfix}'] = torch.argmax(logits_g, dim=-1) - 1
            # accuracy
            if postfix == '_nxt':  # one-step ahead
                data[f'ee_true_part{postfix}'] = get_label(data['query'], data[f'ee_verts{postfix}'], data[f'ee_faces{postfix}'],
                                                           torch.ones_like(data[f'ee_faces{postfix}'][..., 0, None]))
                acc_dict, acc_best_perm = self.evaluator.accuracy(data, 'pre', postfix, acc_dict)
                data[f'best_perm_acc{postfix}'] = acc_best_perm
        data[f'z_pre{postfix}'] = z_pre.detach()

        return z_pre, (data, loss_dict, acc_dict)


def loss_tuple(z, z_ref, metric='smooth'):
    if metric == 'l1':
        loss_latent = torch.nn.functional.l1_loss(z, z_ref)
    elif metric == 'smooth':
        loss_latent = torch.nn.functional.smooth_l1_loss(z, z_ref, beta=1.0)
    elif metric == 'l2':
        loss_latent = torch.nn.functional.mse_loss(z, z_ref)
    elif metric == 'cos':
        loss_latent = 1-torch.nn.functional.cosine_similarity(z, z_ref, dim=-1).mean()
    return loss_latent

def voxelize_to_ground(observed, voxel_size, padding_size, num_samples):    
    B, _, _ = observed.shape
    
    # voxelize observation
    observed_voxels = observed.clone()
    observed_voxels[..., :3] = torch.round(observed_voxels[..., :3] / voxel_size) * voxel_size
    xy_coordinates = observed_voxels[..., :2]
    z_coordinates = observed_voxels[..., 2]
    
    samples = torch.zeros((B, num_samples, 3), device=observed_voxels.device)
    for b in range(B):
        # get unique xy-coordinates (and their corresponding labels)
        unique_xy, unique_indices = torch.unique(xy_coordinates[b], dim=0, return_inverse=True)
        # get the maximal z-coordinate per unique xy-coordinate
        max_z_per_xy = torch.zeros(unique_xy.shape[0], dtype=z_coordinates.dtype, device=samples.device)
        max_z_per_xy = max_z_per_xy.scatter_reduce(index=unique_indices,
                                                   src=z_coordinates[b],
                                                   reduce='amax', dim=0, include_self=False)
        # for all unique xy-coordinates, fill from the overall max(max_z_per_xy) to the ground...
        z_values = torch.arange(0, max_z_per_xy.max() + voxel_size, voxel_size, device=samples.device)[None, :, None].repeat(unique_xy.shape[0], 1, 1)
        xy_values = unique_xy[:, None, :].repeat(1, z_values.shape[1], 1)
        xyz_values = torch.cat([xy_values, z_values], dim=-1)
        # ... and prune those resulting voxels whose z-coordinate > max_z_per_xy
        max_z_values = max_z_per_xy[:, None, None].repeat(1, z_values.shape[1], 1)
        mask = z_values <= max_z_values
        xyz_lower = xyz_values[mask.squeeze(2)]

        # sample uniformly within the remaining voxels
        # note: oversample each voxel first and then randomly subsample over all points to get exactly {num_samples} samples
        oversample_factor = int(np.ceil(num_samples / xyz_lower.shape[0]))
        xyz_upper = xyz_lower + voxel_size
        xyz_lower, xyz_upper = xyz_lower.repeat(oversample_factor, 1), xyz_upper.repeat(oversample_factor, 1)
        voxel_scale = (xyz_upper - xyz_lower) + padding_size  # note: all same size actually
        voxel_mins = xyz_lower - padding_size/2
        voxel_samples = torch.rand_like(xyz_lower) * voxel_scale + voxel_mins
        # subsample to get desired number of samples (note: shuffles the order of the samples too -- gets rid of xy-coordinates' order)
        samples[b] = voxel_samples[torch.randperm(voxel_samples.shape[0])][:num_samples]
    
    return samples


def sample_points(data, prefix, postfix, num_samples):
    points, face_indices = sample_mesh(data[f'{prefix}_verts{postfix}'], data[f'{prefix}_faces{postfix}'][0],  # same faces for all
                                       num_samples=num_samples,
                                       )  # B x N x 3
    # add part label (for visualization)
    if prefix == 'obj':
        points_label = torch.gather(data[f'{prefix}_face_labels{postfix}'].squeeze(-1), 1, face_indices).unsqueeze(-1)  # B x N x 1
    else:
        points_label = torch.zeros_like(points[..., 0, None])  # B x N x 1
    points = torch.cat([points, points_label], dim=-1)  # B x N x 4

    return points


def get_label(query, verts, faces, face_labels, robust=True):
    if robust:  # compute everything per mesh (~ 1.5x slower)
        true_part = torch.zeros((query.shape[0], query.shape[1]), device=query.device, dtype=torch.int)
        for b in range(query.shape[0]):
            labels = face_labels[b]
            unique_labels = torch.unique(labels)  # consecutive per label but padded with first label to constant size
            for part_label in unique_labels:
                part_mask = (labels == part_label).squeeze()
                part_faces = faces[b][part_mask]
                is_inside = check_sign(verts[b, None], part_faces.long(), query[b, None]).squeeze()
                true_part[b][is_inside] = part_label
    else:  # batched
        # get in/out and closest face label
        face_verts = index_vertices_by_faces(verts, faces[0].long())  # same faces for all meshes
        pd, face_indices, distance_type = point_to_mesh_distance(query, face_verts)
        is_inside = check_sign(verts, faces[0].long(), query)
        # sdf = torch.where(is_inside, -pd, pd)

        # label of closest face
        closest_part = torch.gather(face_labels.squeeze(-1), 1, face_indices)  # >=1
        true_part = torch.where(is_inside, closest_part, torch.zeros_like(closest_part))  # 0 if outside

    return true_part[..., None]  # B x Q x 1  = labels for predicted_part probabilities
