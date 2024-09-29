import numpy as np
import mpm as us
from typing import Optional
from .options import Options


############################ Top level: simulator ############################
'''
Simulator is the physics coordination / integration layer
'''

class SimOptions(Options):
    step_dt            : float = 2e-3
    substep_dt         : float = 2e-4
    max_substeps_local : int   = 50
    gravity            : tuple = (0.0, 0.0, -10.0)
    floor_height       : float = 0.0


############################ Solvers inside simulator ############################
'''
Parameters in these solver-specific options will override SimOptions if available
'''

class ToolOptions(Options):
    step_dt      : Optional[float] = None
    substep_dt   : Optional[float] = None
    floor_height : float           = None


class MPMOptions(Options):
    step_dt                        : Optional[float] = None
    substep_dt                     : Optional[float] = None
    gravity                        : Optional[tuple] = None
    particle_diameter              : float           = 0.01
    grid_density                   : int             = 64

    # These will later be converted to discrete grid bound. The actual grid boundary could be slightly bigger.
    lower_bound                    : tuple           = (0.0, 0.0, 0.0)
    upper_bound                    : tuple           = (1.0, 1.0, 1.0)

    # Sparse computation parameter. Don't touch unless you know what you are doing.
    use_sparse_grid                : bool            = False
    leaf_block_size                : int             = 16 # TODO: should be 4 following taichi_elements, but will hang and crash. Probably due to some memory access issue.

    def __init__(self, **data):
        super().__init__(**data)
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            us.raise_exception('Invalid pair of upper_bound and lower_bound.')
