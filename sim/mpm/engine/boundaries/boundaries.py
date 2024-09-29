import numpy as np
import taichi as ti
import mpm as us
from mpm.utils.misc import *
from mpm.utils.repr import _repr


@ti.data_oriented
class CubeBoundary():
    def __init__(self, lower, upper, restitution=0.0):
        self.restitution = restitution

        self.upper = np.array(upper, dtype=us.FTYPE_NP)
        self.lower = np.array(lower, dtype=us.FTYPE_NP)
        assert (self.upper >= self.lower).all()

        self.upper_ti = ti.Vector(upper, dt=us.FTYPE_TI)
        self.lower_ti = ti.Vector(lower, dt=us.FTYPE_TI)

    @ti.func
    def impose_pos_vel(self, pos, vel):
        for i in ti.static(range(3)):
            if pos[i] >= self.upper_ti[i] and vel[i] >= 0:
                vel[i] *= -self.restitution
            elif pos[i] <= self.lower_ti[i] and vel[i] <= 0:
                vel[i] *= -self.restitution

        pos = ti.max(ti.min(pos, self.upper_ti), self.lower_ti)

        return pos, vel

    @ti.func
    def impose_pos(self, pos):
        pos = ti.max(ti.min(pos, self.upper_ti), self.lower_ti)
        return pos

    def is_inside(self, pos):
        return np.all(pos < self.upper) and np.all(pos > self.lower)

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'lower       : {_repr(self.lower)}\n' \
               f'upper       : {_repr(self.upper)}\n' \
               f'restitution : {_repr(self.restitution)}'


@ti.data_oriented
class FloorBoundary():
    def __init__(self, height, restitution=0.0):
        self.height = height
        self.restitution = restitution

    @ti.func
    def impose_pos_vel(self, pos, vel):
        if pos[2] <= self.height and vel[2] <=0:
            vel[2] *= -self.restitution

        pos[2] = ti.max(pos[2], self.height)

        return pos, vel

    @ti.func
    def impose_pos(self, pos):
        pos[2] = ti.max(pos[2], self.height)
        return pos

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'height      : {_repr(self.height)}\n' \
               f'restitution : {_repr(self.restitution)}'
