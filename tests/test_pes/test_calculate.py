import numpy as np
import pytest
from .. import testutils
from autode.utils import work_in_tmp_dir
from autode.species import Molecule
from autode.atoms import Atom
from autode.methods import XTB
from autode.pes.relaxed import RelaxedPESnD


def h2():
    return Molecule(atoms=[Atom("H"), Atom("H", x=0.70)])


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calculate_no_species():
    pes = RelaxedPESnD(species=None, rs={})

    # cannot calculate a PES without a species
    with pytest.raises(ValueError):
        pes.calculate(method=XTB())


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calculate_1d():
    pes = RelaxedPESnD(species=h2(), rs={(0, 1): (1.7, 10)})

    pes.calculate(method=XTB())

    # All points should have a defined energy
    for i in range(10):
        assert pes._has_energy(point=(i,))


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calculate_1d_serial():
    """
    Ensure that a serial calculation results in the same
    PES as the default parallel analogue, although on a 1D
    surface they should be the same
    """

    n_points = 5
    pes = RelaxedPESnD(species=h2(), rs={(0, 1): (1.0, n_points)})

    pes.calculate(method=XTB())
    energies = [pes[i] for i in range(n_points)]
    pes.clear()

    # Recalculate in serial
    for i in range(n_points):
        e, coords = pes._single_energy_coordinates(pes._species_at(point=(i,)))
        pes._energies[i] = e
        pes._coordinates[i] = coords
        assert np.isclose(energies[i], e, atol=1e-6)
