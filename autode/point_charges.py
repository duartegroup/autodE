import numpy as np


class PointCharge:

    def __init__(self, charge, x=0.0, y=0.0, z=0.0, coord=None):
        """
        Point charge

        Arguments:
            charge (float): Charge in units of e

        Keyword Arguments:
            x (float): x coordinate (Å)
            y (float): y coordinate (Å)
            z (float): z coordinate (Å)
            coord (np.ndarray): Length 3 array of x, y, z coordinates or None
        """
        self.charge = float(charge)
        self.coord = np.array([float(x), float(y), float(z)])

        # If initialised with a coordinate override the default
        if coord is not None:
            assert type(coord) is np.ndarray
            assert len(coord) == 3
            self.coord = coord
