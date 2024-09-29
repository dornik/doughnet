import torch
import numpy as np


def manual_seed(seed):
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)
    np.random.seed(seed)

def get_state():
    torch_state = torch.get_rng_state()
    torch_cuda_state = torch.cuda.get_rng_state()
    np_state = np.random.get_state()
    return (torch_state, torch_cuda_state, np_state)

def set_state(states):
    torch.set_rng_state(states[0])
    torch.cuda.set_rng_state(states[1])
    np.random.set_state(states[2])
