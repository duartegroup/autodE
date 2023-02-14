import numpy as np

from autode.bracket.imagepair import TwoSidedImagePair
from autode.values import Distance
from autode.opt.coordinates import OptCoordinates, CartesianCoordinates


class BinaryImagePair(TwoSidedImagePair):
    """
    A Binary-Image pair use for the BITSS procedure for
    transition state search
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._k_eng = None  # energy constraint
        self._k_dist = None  # distance constraint
        self._d_i = None  # d_i

    @property
    def dist_vec(self) -> np.ndarray:
        """The distance vector pointing to right_image from left_image"""
        return np.array(self.left_coord - self.right_coord)

    @property
    def dist(self) -> Distance:
        """
        Distance between BITSS images. (Currently implemented
        as Euclidean distance in Cartesian)

        Returns:
            (Distance):
        """
        return Distance(np.linalg.norm(self.dist_vec), 'ang')

    @property
    def target_dist(self) -> Distance:
        """
        The target distance (d_i) set for BITSS

        Returns:
            (Distance)
        """
        return Distance(self._d_i, 'ang')

    @target_dist.setter
    def target_dist(self, value):
        """
        Set the target distance(d_i) used for BITSS

        Args:
            value (Distance|float):
        """
        if value is None:
            return
        if isinstance(value, Distance):
            self._d_i = float(value.to('ang'))
        elif isinstance(value, float):
            self._d_i = value
        else:
            raise ValueError("Unknown type")

    @property
    def bitss_coords(self) -> OptCoordinates:
        """
        The BITSS coordinates. Concatenated coordinates
        of the left and right images

        Returns:
            (OptCoordinates):
        """
        return CartesianCoordinates(
            np.concatenate((self.left_coord, self.right_coord))
        )

    @bitss_coords.setter
    def bitss_coords(self, value):
        if isinstance(value, OptCoordinates):
            coords = value.to('ang').flatten()
        elif isinstance(value, np.ndarray):
            coords = value.flatten()
        else:
            raise ValueError("Unknown type")

        if coords.shape != (3 * 2 * self.n_atoms,):
            raise ValueError("Coordinates in wrong shape")

        self.left_coord = coords[:3 * self.n_atoms]
        self.right_coord = coords[3 * self.n_atoms:]   # todo check

    def bitss_energy(self):
        # E_BITSS = E_1 + E_2 + k_e(E_1 - E_2)^2 + k_d(d-d_i)^2
        e_1 = float(self.left_coord.e)
        e_2 = float(self.right_coord.e)
        return float(e_1 + e_2 + self._k_eng * (e_1 - e_2)**2
                     + self._k_dist * (self.dist - self._d_i)**2)

    def bitss_grad(self):
        pass

    def bitss_hess(self):
        pass


