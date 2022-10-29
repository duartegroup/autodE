import numpy as np
from autode.values import EnergyArray as Energies
from autode.pes.reactive import ReactivePESnD


class TestPES(ReactivePESnD):

    __test__ = False

    def __init__(self, rs, species=None):
        super().__init__(species=species, rs=rs)

    def _default_keywords(self, method):
        raise NotImplementedError

    @property
    def _default_keyword_type(self):
        raise NotImplementedError

    def _calculate(self) -> None:
        """Skip the calculation in the test class"""


def harmonic_2d_pes():
    # Symmetric PES in x and y (atom indexes are dummy)
    pes = TestPES(
        rs={
            (0, 1): np.linspace(-1, 1, num=21),
            (1, 2): np.linspace(-1, 1, num=21),
        }
    )

    def energy(x, y):
        return 0.01 * (x**2 + y**2)

    pes._energies = Energies(energy(pes.r1, pes.r2))

    return pes
