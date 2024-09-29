import torch
import numpy as np
import mpm as us


def _repr_list_d1(x):
    if len(x) > 0:
        return f'{_repr_type(x)} of {_repr_type(x[0])}, len: {len(x)}'
    else:
        return f'{_repr_type(x)}, len: {len(x)}'


def _repr(x):
    if type(x) in [int, float, dict, bool, np.int32, np.int64, np.float32, np.float64]:
        return f'{_repr_type(x)}: {x}'

    elif type(x) is str:
        return f"'{x}'"

    elif type(x) in [np.ndarray, torch.Tensor]:
        if len(x.shape) > 1 or x.shape[0] > 10:
            return f'{_repr_type(x)}, shape: {x.shape}'
        else:
            return x.__repr__()

    elif type(x) in [list, tuple]:
        if len(x) < 8:
            return f'{x}'
        else:
            return _repr_list_d1(x)

    elif type(x) is type(None):
        return f'None'

    else:
        return _repr_type(x)

def _repr_type(x):
    return f"<{str(x.__class__)[8:-2]}>"

def _simple_repr(x):
    repr_str = f'{_repr(x)}\n'

    max_key_l = max([len(key) for key in x.__dict__])
    for key in x.__dict__:
        repr_str += f'{key:<{max_key_l+1}}: {_repr(x.__dict__[key])}\n'
    # remove last \n
    repr_str = repr_str[:-1]
    
    return repr_str

def _repr_uuid(x, length=7):
    return f'<{x[:length]}>'