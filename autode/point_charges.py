from typing import Sequence, Optional

from autode.atoms import DummyAtom
from autode.values import Coordinate


class PointCharge(DummyAtom):
    def __init__(
        self,
        charge: float,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        coord: Optional[Sequence] = None,
    ):
        """
        Point charge

        -----------------------------------------------------------------------
        Arguments:
            charge (float): Charge in units of e

        Keyword Arguments:
            x (float): x coordinate (Å)
            y (float): y coordinate (Å)
            z (float): z coordinate (Å)
            coord (np.ndarray | None): Length 3 array of x, y, z coordinates
                                       or None
        """
        super().__init__(x, y, z)

        self.charge = float(charge)

        if coord is not None:
            assert len(coord) == 3, "Coordinate much have 3 components: x,y,z"
            x, y, z = coord[0], coord[1], coord[2]
            self._coord = Coordinate(float(x), float(y), float(z))
