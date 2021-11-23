import pytest
from .. import testutils
from autode.species import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.pes.relaxed import RelaxedPESnD


def test_calculate_no_species():
    pes = RelaxedPESnD()

    # cannot calculate a PES without a species
    with pytest.raises(ValueError):
        pes.calculate(method=XTB())


@testutils.requires_with_working_xtb_install
def test_calculate_1d():
    h2 = Molecule(atoms=[Atom('H'), Atom('H', x=0.7)])
    pes = RelaxedPESnD(species=h2, rs={(0, 1): (1.7, 10)})

    pes.calculate(method=XTB())
