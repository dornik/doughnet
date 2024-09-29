import os
import sys
DEBUG = hasattr(sys, 'gettrace') and (sys.gettrace() is not None)
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import hdf5plugin
import h5py

import net.dataset.augmentations as aug
from net.pipeline.random import manual_seed


class DoughDataset(Dataset):
    
    def __init__(self, config):
        self.config = config
        self.path = config.path
        self.subset = config.subset

        # load data
        print(f'Loading {self.subset} data...')
        #   optionally subsample keyframes
        scene_step = 1 if not self.config.use_small else 20
        keyframe_step = 1 if self.config.next_frames == 0 else self.config.next_frame_offset
        with h5py.File(os.path.join(self.path, f'dataset.h5'), 'r', libver='latest') as f:
            self.data = {k: f[self.subset][k][::scene_step, ::keyframe_step] for k in f[self.subset].keys()}
        print(f'  {self.subset}: {len(self.data["scene"])} scenes with {len(self.data["scene"][0])} frames each.')

        # define keys to process/keep
        if self.subset == 'real':
            self.ks_augment = ['obj_observed', 'ee_verts', 'ee_observed',]
            self.ks_process = self.ks_augment + ['ee_faces',]
        else:
            self.ks_augment = ['obj_verts', 'ee_verts', 'ee_observed',]
            self.ks_process = self.ks_augment + ['obj_faces', 'obj_vert_labels', 'obj_face_labels', 'ee_faces',  # geometry
                                                 'genus']  # topology (-1 indicates 'no component', >=0 the component's genus)

        # if we need information from next frame(s), they need to be from the same scene;
        #   thus, skip last frames as they don't have "next frames"
        self.next_frames = self.config.next_frames
        assert 0 <= self.next_frames <= 2
        num_scenes, num_frames = self.data['scene'].shape  # renames to consecutive frames in case of keyframe_step > 1
        self.consecutive_items = [[si, fi] for si in range(num_scenes)
                                  for fi in range(num_frames-self.next_frames)]

        # prepare augmentations
        self.transforms = Compose([
            aug.Mirror(self.config.augmentation.mirror),
            aug.Scale(self.config.augmentation.scale),
        ]) if self.subset == 'train' else aug.Identity()
        
    def __len__(self):
        return len(self.consecutive_items)
    
    def __getitem__(self, idx):
        if self.subset != 'train':
            manual_seed(idx)  # deterministic val/test (leads to same augmentation for same idx)
        scene, frame = self.consecutive_items[idx]
        # process frame(s)
        item = {}
        item['augmentation'] = self.transforms(np.eye(4, dtype=np.float32))  # same for all frames
        for i in range(self.next_frames+1):
            postfix = ('' if i == 0 else '_') + ''.join(['nxt']*i)  # '', '_nxt', '_nxtnxt'
            item_data = {f'{k}{postfix}': self.data[k][scene, frame+i] for k in self.ks_process}  # get frame data
            for k in self.ks_augment:  # apply augmentation
                item_data[f'{k}{postfix}'] = torch.from_numpy(item_data[f'{k}{postfix}'] @ item['augmentation'][:3, :3] + item['augmentation'][:3, 3])
            item.update(item_data)  # add to item
            if i > 0:
                prev_postfix = ('' if i <= 1 else '_') + ''.join(['nxt']*(i-1))  # '', '', '_nxt'
                item[f'ee_target{prev_postfix}'] = item[f'ee_observed{postfix}']  # set target of previous to current observed
        # add meta information
        item['idx'] = idx
        item['scene'] = scene
        item['frame'] = frame
        return item
