import taichi as ti
import mpm as us
from .base import Base
from mpm.utils.repr import _simple_repr


@ti.data_oriented
class ElastoPlastic(Base):

    def __init__(
        self, 
        scene,
        name         = 'elasto-plastic',
        mu           = 416.67,
        lam          = 277.78,
        rho          = 1.0,
        filling      = 'random',
        yield_lower  = 2e-3,
        yield_higher = 3e-3,
    ):
        super().__init__(scene, name)

        self.mu      = mu
        self.lam     = lam
        self.rho     = rho
        self.filling = filling
        self.yield_lower  = yield_lower
        self.yield_higher = yield_higher

        # add to solver
        self.idx = scene.sim.mpm_solver.mat_count
        scene.sim.mpm_solver.mat_models[self.idx] = (self.update_stress, self.update_F)
        scene.sim.mpm_solver.mat_idxs.append(self.idx)
        scene.sim.mpm_solver.mat_count += 1

    @ti.func
    def update_stress(self, U, S, V, F_tmp, mu, lam, J):
        # NOTE: class member function inheritance will still introduce redundant computation graph in taichi
        r = U @ V.transpose()
        stress = 2 * mu * (F_tmp - r) @ F_tmp.transpose() + ti.Matrix.identity(us.FTYPE_TI, 3) * lam * J * (J - 1)

        return stress

    @ti.func
    def update_F(self, J, F_tmp, U, S, V):
        S_new = ti.Matrix.zero(us.FTYPE_TI, 3, 3)
        for d in ti.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - self.yield_lower), 1 + self.yield_higher)
        F_new = U @ S_new @ V.transpose()
        return F_new

    def __repr__(self):
        return _simple_repr(self)
