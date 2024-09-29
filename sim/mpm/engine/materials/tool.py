import taichi as ti
from .base import Base


@ti.data_oriented
class Tool(Base):
    def __init__(
        self, 
        scene,
        name             = 'tool',
        friction         = 0.0,
        contact_softness = 100.0,
        collision        = True,
        # only used when collision is True
        collision_type   = 'particle',
        sdf_res          = 128,
    ):
        super().__init__(scene, name)

        self.friction         = friction
        self.contact_softness = contact_softness
        self.collision        = collision
        self.collision_type   = collision_type
        self.sdf_res          = sdf_res

        assert self.collision_type in ['particle', 'grid', 'both']
