import numpy as np


class Identity:

    def __call__(self, cur_trafo):
        return cur_trafo

class Mirror:

    def __init__(self, apply) -> None:
        self.apply = apply

    def __call__(self, cur_trafo):
        if self.apply:
            trafo = np.eye(4, dtype=np.float32)
            if np.random.uniform() < 0.5:
                trafo[0, 0] = -1.0  # flip x
            if np.random.uniform() < 0.5:
                trafo[1, 1] = -1.0  # flip y
            return trafo @ cur_trafo
        else:
            return cur_trafo

class Scale:

    def __init__(self, scale_range) -> None:
        self.scale_min, self.scale_max = scale_range

    def __call__(self, cur_trafo):
        if self.scale_min != 1.0 or self.scale_max != 1.0:
            trafo = np.eye(4, dtype=np.float32)
            scales = np.random.uniform(self.scale_min, self.scale_max, size=3).astype(np.float32)
            trafo[:3, :3] = np.diag(scales)
            return trafo @ cur_trafo
        else:
            return cur_trafo
