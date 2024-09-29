import mpm as us
from .options import Options
from typing import Optional, Union


############################ Shape Primitives ############################


class Cube(Options):
    lower   : tuple           = (0.0, 0.0, 0.0)
    upper   : Optional[tuple] = (0.1, 0.1, 0.1)
    size    : Optional[tuple] = None
    euler   : tuple           = (0.0, 0.0, 0.0)
    name    : str             = 'cube'

class Cylinder(Options):
    center  : tuple = (0.0, 0.0, 0.0)
    height  : float = 0.0
    radius  : float = 0.0
    euler   : tuple = (0.0, 0.0, 0.0)
    name    : str   = 'cylinder'
    
class Supertoroid(Options):
    center  : tuple = (0.0, 0.0, 0.0)
    size    : tuple = (1.0, 1.0, 1.0)
    hole    : float = 1.0
    e_lat   : float = 1.0
    e_lon   : float = 1.0
    euler   : tuple = (0.0, 0.0, 0.0)
    name    : str   = 'supertoroid'

class Particles(Options):
    centers : list  = []            # list of tuples
    euler   : tuple = (0.0, 0.0, 0.0)  # required by particle_entity; always set to 0s
    name    : str   = 'particles'

############################ Mesh ############################

class Mesh(Options):
    file         : str                 = '*.obj'
    file_vis     : Optional[str]       = None
    offset_pos   : tuple               = (0.0, 0.0, 0.0)
    offset_euler : tuple               = (0.0, 0.0, 0.0)
    pos          : tuple               = (0.0, 0.0, 0.0)
    euler        : tuple               = (0.0, 0.0, 0.0)
    scale        : Union[float, tuple] = 1.0
    name         : str                 = 'mesh'

    # MPM specific
    voxelize_res: int = 128

    # Rigid specific
    mass              : float = 1.0
    fixed             : bool  = False
    group_by_material : bool  = True
    fast_mode         : bool  = True
