import os
import torch
import socket
import random
import colorsys
import numpy as np
import mpm as us


def set_random_seed(seed):
    # Note: we don't set seed for taichi, since taichi doesn't support stochastic operations in gradient computation. Therefore, we only allow deterministic taichi operations.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_src_dir():
    return os.path.dirname(us.__file__)

def get_cfg_path(file):
    return os.path.join(get_src_dir(), 'envs', 'configs', file)

def get_tgt_path(file):
    return os.path.join(get_src_dir(), 'assets', 'targets', file)

def eval_str(x):
    if type(x) is str:
        return eval(x)
    else:
        return x
        
def is_on_server():
    hostname = socket.gethostname()
    if 'matrix' in hostname:
        return True
    else:
        return False

def alpha_to_transparency(color):
    return np.array([color[0], color[1], color[2], 1.0 - color[3]])

def random_color(h_low=0.0, h_high=1.0, s=1.0, v=1.0, alpha=1.0):
    color = np.array([0.0, 0.0, 0.0, alpha])
    h_range = h_high - h_low
    color[:3] = colorsys.hsv_to_rgb(np.random.rand()*h_range+h_low, s, v)
    return color
