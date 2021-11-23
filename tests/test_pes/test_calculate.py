import pytest
from .. import testutils
from autode.utils import work_in_tmp_dir
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
@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calculate_1d():
    h2 = Molecule(atoms=[Atom('H'), Atom('H', x=0.70)])
    pes = RelaxedPESnD(species=h2, rs={(0, 1): (1.7, 10)})

    pes.calculate(method=XTB())

    # All points should have a defined energy
    for i in range(10):
        assert pes._point_has_energy(point=(i,))
