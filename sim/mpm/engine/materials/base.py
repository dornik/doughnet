import uuid
import numpy as np
import taichi as ti
import mpm as us
from mpm.utils.repr import _simple_repr


@ti.data_oriented
class Base:
    def __init__(
        self, 
        scene,
        name,
    ):
        if scene.is_built:
            us.raise_exception('Material creation is not allowed after a scene is built.')
        scene.mats.append(self)

        self.id             = str(uuid.uuid4())
        self.scene          = scene
        self.name           = name
        
    def __repr__(self):
        return _simple_repr(self)