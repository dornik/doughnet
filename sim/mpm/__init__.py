import sys
import torch
import atexit
import traceback
import numpy as np
import taichi as ti
from enum import IntEnum
from .utils import set_random_seed
from .logging import get_logger

########################## Macros ##########################

_initialized = False

# float type
FTYPE_TI = None
FTYPE_NP = None
FTYPE_TC = None

# int type
ITYPE_TI = None
ITYPE_NP = None
ITYPE_TC = None

# dynamic loading
ACTIVE   = 1
INACTIVE = 0

# misc
EPS      = None
SEED     = None

# logging
logger    = None
error_msg = None

# exit callbacks
exit_cbs = []

########################## Constants ##########################
class USIntEnum(IntEnum):
    def __repr__(self):
        return f'us.{self.__class__.__name__}.{self.name}: {self.value}'

# geom type in rigid solver
class GEOM_TYPE(USIntEnum):
    PLANE    = 0
    SPHERE   = 1
    CYLINDER = 2
    CAPSULE  = 3
    BOX      = 4
    MESH     = 5

########################## init ##########################

def init(seed=None, allocate_gpu_memory=8, precision='32', debug=False, eps=1e-12, logging_level=None):

    # init taichi
    ti.init(arch=ti.gpu, device_memory_GB=allocate_gpu_memory, debug=debug)
    # atexit.register(us_exit)

    # unisim.logger
    global logger
    logger = get_logger(logging_level, debug)
    logger._error_msg = None

    # dtype
    global FTYPE_TI
    global FTYPE_NP
    global FTYPE_TC
    if precision == '32':
        FTYPE_TI = ti.f32
        FTYPE_NP = np.float32
        FTYPE_TC = torch.float32
    elif precision == '64':
        FTYPE_TI = ti.f64
        FTYPE_NP = np.float64
        FTYPE_TC = torch.float64
    else:
        raise_exception('Unsupported precision type.')
        
    # All int uses 32-bit precision, unless under special circumstances.
    global ITYPE_TI
    global ITYPE_NP
    global ITYPE_TC
    ITYPE_TI = ti.i32
    ITYPE_NP = np.int32
    ITYPE_TC = torch.int32

    global EPS
    EPS = eps

    # seed
    if seed is not None:
        global SEED
        SEED = seed
        set_random_seed(SEED)

    global exit_cbs
    exit_cbs = []

    global _initialized
    _initialized = True

    logger.debug(f'Taichi initialized. GPU: {allocate_gpu_memory} GB, seed: {seed}, precision: {precision}, debug: {debug}.')


########################## Exception and exit handling ##########################
    
class UniSimException(Exception):
    pass

def custom_excepthook(exctype, value, tb):
    if issubclass(exctype, UniSimException):
        # We don't want the traceback info to trace till this __init__.py file.
        stack_trace = ''.join(traceback.format_exception(exctype, value, tb)[:-2])
        print(stack_trace)
    else:
        # Use the system's default excepthook for other exception types
        sys.__excepthook__(exctype, value, tb)

# Set the custom excepthook to handle UniSimException
sys.excepthook = custom_excepthook

def raise_exception(msg='Something went wrong.'):
    logger._error_msg = msg
    tb = traceback.format_exc()
    raise UniSimException

def assert_contiguous(tensor):
    if not tensor.is_contiguous():
        raise_exception('Tensor not contiguous.')

def us_exit():
    # display error if it exists
    if logger._error_msg is not None:
        logger.error(logger._error_msg)
    logger.debug('Caching kernels...')
    ti.reset()


########################## shortcut imports for users ##########################

from . import options
from .options import geoms

from .engine import states
from .engine import materials
from .engine.scene import Scene

