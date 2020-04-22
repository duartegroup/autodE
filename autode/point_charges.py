import numpy as np


class PointCharge:

    def __init__(self, charge, x=0.0, y=0.0, z=0.0, coord=None):
        self.charge = float(charge)
        self.coord = np.array([float(x), float(y), float(z)])

        # If initialised with a coordinate override the default
        if coord is not None:
            assert type(coord) is np.ndarray
            self.coord = coord
