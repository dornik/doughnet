from .options import Options


class RendererOptions(Options):
    particle_radius : float = 0.0075
    lights          : list  = [{'pos': (0.5, 0.5, 1.5), 'color': (0.5, 0.5, 0.5)},
                              {'pos': (1.5, 0.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
