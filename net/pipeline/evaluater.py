import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, one_hot
from torchvision.ops import sigmoid_focal_loss
from itertools import permutations


def fn_perm(loss_fn, est, gt, part_permutations, kwargs_fn=lambda x: {}):
    # per batch, compute loss (mean over parts and samples) for each permutation -> B x permutation
    return torch.stack([loss_fn(est[:, :, perm], gt, reduction='none', **kwargs_fn).mean(dim=[-2, -1]) for perm in part_permutations]).T

def agg_perm(losses):
    # minimum over permutations (best match), mean over batches
    return torch.min(losses, dim=1)[0].mean()

def arg_perm(losses):
    # index of best permutation per batch
    return torch.min(losses, dim=1)[1]


class Evaluater(nn.Module):

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        # get permutations of predictions
        if self.config.no_outlier_perm:
            self.part_permutations = list(permutations(range(1, self.config.num_parts)))
            self.part_permutations = [[0] + list(perm) for perm in self.part_permutations]  # prepend 0 to all permutations (outlier)
        else:
            self.part_permutations = list(permutations(range(self.config.num_parts)))
    
    def cuda(self, device='cuda'):
        self.part_permutations = torch.LongTensor(self.part_permutations).to(device)
        return super().cuda(device)

    def _get_losses(self, part_est, part_gt_one_hot, loss_dict=None):
        if loss_dict is not None:  # assumes that part_est is only the best one
            loss_focal = sigmoid_focal_loss(part_est, part_gt_one_hot, reduction='mean',
                                            alpha=self.config.focal_alpha, gamma=self.config.focal_gamma)
            loss_dict['focal'] = loss_focal
            loss = loss_focal * self.config.w_focal
            return loss, loss_dict
        else:
            losses_focal = fn_perm(sigmoid_focal_loss, part_est, part_gt_one_hot, self.part_permutations,
                                    {'alpha': self.config.focal_alpha, 'gamma': self.config.focal_gamma})
            losses = losses_focal * self.config.w_focal
            return losses

    def loss(self,
             part_est, part_gt, genus_est, genus_gt,
             prefix, postfix, loss_dict=None):

        # labels to binary masks: outliers at 0, labels start from 1 -- outliers' one-hot is [1, 0, ..., 0]
        part_gt_one_hot = one_hot(part_gt.squeeze(2).long(), self.config.num_parts).float()

        # == LOSS ==
        # -- part labels --
        with torch.no_grad():
            # compute loss for all permutations without gradient
            losses = self._get_losses(part_est, part_gt_one_hot)
            # get best permutation per batch
            best_perm_idx = arg_perm(losses)
            best_perm = self.part_permutations[best_perm_idx]
        # fetch part_est for best permutation
        best_part_est = torch.gather(part_est, 2, best_perm.unsqueeze(1).expand(-1, part_est.shape[1], -1))
        # compute loss for best permutation with gradient
        loss_dict = {} if loss_dict is None else loss_dict
        loss, new_loss_dict = self._get_losses(best_part_est, part_gt_one_hot, {})

        if self.config.w_top > 0:
            # -- topology --
            # apply permutation from model itself
            g_logit = torch.gather(genus_est, 1, best_perm.unsqueeze(2).expand(-1, -1, genus_est.shape[2]))  # B x max parts x (num_genus+1)
            # outliers are -1 in annotation, shift everthing +1 to match cross_entropy
            loss_top = cross_entropy(g_logit.reshape(-1, 4), genus_gt.reshape(-1).long() + 1)
            new_loss_dict['top'] = loss_top
            loss += loss_top * self.config.w_top
        
        # add pre/postfix to loss names
        new_loss_dict = {f'{prefix}_{k}{postfix}': v for k, v in new_loss_dict.items()}
        loss_dict.update(new_loss_dict)
        loss_dict[f'{prefix}_loss{postfix}'] = loss

        return loss_dict, best_perm

    def accuracy(self, data, prefix, postfix, acc_dict=None):
        acc_dict = {} if acc_dict is None else acc_dict

        batch_viou, batch_ciou, batch_accc, batch_accg = [], [], [], []
        for b in range(data['query'].shape[0]):

            # = prepare data
            y_true = data[f'obj_true_part{postfix}'][b]
            pred_probs = data[f'obj_predicted_part{postfix}'][b]
            if len(pred_probs.shape) == 1:
                pred_probs = pred_probs.reshape(-1, 1)
            if pred_probs.shape[-1] == 1:
                pred_probs = one_hot(pred_probs.long(), 5).float().reshape(-1, 5)
            # ignore estimates below plane
            above_plane = data['query'][b, :, 2] >= 0
            y_true[~above_plane] = 0  # outlier
            pred_probs[~above_plane] = torch.tensor([1.0, 0, 0, 0, 0], device=y_true.device)  # outlier
            # ignore estimates within ee
            in_ee = data[f'ee_true_part{postfix}'][b]
            y_true[in_ee] = 0
            pred_probs[in_ee] = torch.tensor([1.0, 0, 0, 0, 0], device=y_true.device)  # outlier
            # = compute iou
            vious, cious = ious_from_prob(pred_probs, y_true, self.part_permutations)
            best_perm, viou, ciou = best_from_ious(vious, cious, self.part_permutations)
            batch_viou += [viou]
            batch_ciou += [ciou]
            if self.config.w_top > 0:
                # = genus accuracy
                g_true = data[f'genus{postfix}'][b].float()
                g_pred = data[f'predicted_genus{postfix}'][b].float()
                g_pred = g_pred[best_perm]  # permuted to match best components according to ciou
                # only for GT inliers
                g_true_inliers = g_true[g_true >= 0]
                g_pred_inliers = g_pred[g_true >= 0]
                batch_accg += [float((g_pred_inliers == g_true_inliers).float().mean())]  # in [0, 1] -> frame average
                # = component accuracy
                c_true = (g_true >= 0).sum()
                c_pred = (g_pred >= 0).sum()
                batch_accc += [float(c_pred == c_true)]  # 0 or 1
        
        for k, values in zip(['viou', 'ciou', 'accc', 'accg'], [batch_viou, batch_ciou, batch_accc, batch_accg]):
            acc_dict[f'{prefix}_{k}{postfix}'] = torch.tensor(values).mean()

        return acc_dict, best_perm


def iou(y_pred, y_true):  # per sample in batch
    y_inter = (y_pred * y_true).sum(dim=-1)
    y_union = y_pred.sum(dim=-1) + y_true.sum(dim=-1) - y_inter
    return torch.where(y_union > 0, y_inter / y_union, torch.ones_like(y_inter))

def ious(y_pred, y_true, num_parts, mode='macro-valid', ignore_outliers=True, keep_batch=False):
    start_idx = 1 if ignore_outliers else 0

    voxel_iou = iou(y_pred > 0, y_true > 0).float()
    label_iou = torch.cat([iou(y_pred == li, y_true == li)[:, None] for li in range(start_idx, num_parts)], dim=1).float()

    weights = torch.stack([torch.stack([(y_tr == li).sum() for li in range(start_idx, num_parts)])
                           for y_tr in y_true]).float()
    weights = weights / weights.sum(dim=-1, keepdim=True)  # normalize by number of points per sample
        
    if mode == 'macro':
        pass  # label_iou = label_iou
    elif mode == 'macro-valid':
        # macro but ignore where true mask is empty
        label_iou = label_iou[weights > 0]
    else:  # weighted
        label_iou = (label_iou * weights).sum(dim=-1)

    if not keep_batch:
        voxel_iou = voxel_iou.mean()
        label_iou = label_iou.mean()

    return voxel_iou, label_iou

def ious_from_prob(y_probs, y_true, part_permutations):
    vious, cious = [], []
    y_true = y_true.reshape(1, -1)
    for perm in part_permutations:
        pred_labels_perm = torch.argmax(y_probs[:, perm], dim=-1).reshape(1, -1)
        viou, ciou = ious(pred_labels_perm, y_true, num_parts=5)
        vious += [viou]
        cious += [ciou]
    return torch.stack(vious), torch.stack(cious)

def best_from_ious(vious, cious, part_permutations, match_mode='ciou'):
    if match_mode == 'ciou':
        best_idx = torch.argmax(cious)
    else:  # viou
        best_idx = torch.argmax(vious)
    best_perm = part_permutations[best_idx]
    best_viou = vious[best_idx]
    best_ciou = cious[best_idx]
    return best_perm, float(best_viou), float(best_ciou)
