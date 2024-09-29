import mpm as us
from .options import Options
from typing import Optional, Union


class ViewerOptions(Options):
    res             : tuple  = (1280, 1280)
    camera_pos      : tuple  = (3.5, 0.5, 2.5)
    camera_lookat   : tuple  = (0.5, 0.5, 0.5)
    camera_up       : tuple  = (0.0, 0.0, 1.0)
    camera_fov      : float  = 30
    vsync           : bool   = True
    particle_radius : float  = 0.0075
    lights          : list   = [{'pos': (0.5, 0.5, 1.5), 'color': (0.5, 0.5, 0.5)},
                              {'pos': (1.5, 0.5, 1.5), 'color': (0.5, 0.5, 0.5)}]


class SurfaceOptions(Options):
    '''
    if color or image is specified, they will override asset's own property.
    color has a higher priority then image.
    '''
    material    : str             = 'matte'
    roughness   : float           = 0.0
    color       : Optional[tuple] = None
    image       : Optional[str]   = None
    image_color : tuple           = (1.0, 1.0, 1.0, 1.0)

    def __init__(self, **data):
        super().__init__(**data)
        
        if self.material not in ['null', 'matte', 'substrate', 'glass', 'metal']:
            us.raise_exception('Invalid material.')

