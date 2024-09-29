import torch
import mpm as us
from mpm.utils.repr import _repr


class ToolEntityState:
    '''
    Dynamic state queried from a unisim ToolEntity.
    '''
    def __init__(self, entity, s_global):
        self.entity   = entity
        self.s_global = s_global

        self.pos  = torch.zeros(3, dtype=us.FTYPE_TC).cuda()
        self.quat = torch.zeros(4, dtype=us.FTYPE_TC).cuda()
        self.vel  = torch.zeros(3, dtype=us.FTYPE_TC).cuda()
        self.ang  = torch.zeros(3, dtype=us.FTYPE_TC).cuda()

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'entity : {_repr(self.entity)}\n' \
               f'pos    : {_repr(self.pos)}\n' \
               f'quat   : {_repr(self.quat)}\n' \
               f'vel    : {_repr(self.vel)}\n' \
               f'ang    : {_repr(self.ang)}'


class MPMEntityState:
    '''
    Dynamic state queried from a unisim MPMEntity.
    '''
    def __init__(self, entity, s_global):
        self.entity   = entity
        self.s_global = s_global

        self.pos    = torch.zeros((self.entity.n, 3), dtype=us.FTYPE_TC).cuda()
        self.vel    = torch.zeros((self.entity.n, 3), dtype=us.FTYPE_TC).cuda()
        self.C      = torch.zeros((self.entity.n, 3, 3), dtype=us.FTYPE_TC).cuda()
        self.F      = torch.zeros((self.entity.n, 3, 3), dtype=us.FTYPE_TC).cuda()
        self.mat_id = torch.zeros((self.entity.n,), dtype=us.ITYPE_TC).cuda()

    def __repr__(self):
        return f'{_repr(self)}\n' \
               f'entity : {_repr(self.entity)}\n' \
               f'pos    : {_repr(self.pos)}\n' \
               f'vel    : {_repr(self.vel)}\n' \
               f'C      : {_repr(self.C)}\n' \
               f'F      : {_repr(self.F)}'
