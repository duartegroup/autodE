import pytest
import numpy as np
from .. import testutils
from autode.utils import work_in_tmp_dir
from autode.species.molecule import Molecule
from autode.wrappers.XTB import XTB
from autode.wrappers.keywords import SinglePointKeywords
from autode.atoms import Atom
from autode.pes.unrelaxed import UnRelaxedPES1D


@testutils.requires_working_xtb_install
@work_in_tmp_dir()
def test_h2_points():
    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=0.8)])

    pes = UnRelaxedPES1D(h2, rs={(0, 1): (1.5, 10)})  # -> 1.5 Å in 10 steps

    pes.calculate(method=XTB())

    # Energy should be monotonic increasing over H2 bond length expansion
    for n in range(1, pes.shape[0]):
        assert pes[n] > pes[n - 1]


def test_species_at():
    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=0.8)])

    pes = UnRelaxedPES1D(h2, rs={(0, 1): (1.5, 3)})
    pes._init_tensors()

    h2_final = pes._species_at(point=(2,))
    assert np.isclose(h2_final.distance(0, 1), 1.5, atol=1e-6)


def test_non_bond_unrelaxed():
    h2o = Molecule(atoms=[Atom("O"), Atom("H", x=-0.9), Atom("H", x=0.9)])

    pes = UnRelaxedPES1D(h2o, rs={(1, 2): (3.0, 3)})
    pes._init_tensors()

    # Cannot pass the checks if the scanned distance is not a bond
    with pytest.raises(ValueError):
        pes._check()


def test_unrelaxed_kwd_type():
    pes = UnRelaxedPES1D(
        Molecule(atoms=[Atom("H"), Atom("H", x=0.8)]), rs={(0, 1): (1.5, 10)}
    )

    assert isinstance(pes._default_keyword_type("a str"), SinglePointKeywords)


@testutils.requires_working_xtb_install
def test_unrelaxed_must_be_over_a_single_dim():
    h3 = Molecule(atoms=[Atom("H"), Atom("H", x=0.8), Atom("H", x=-0.8)])

    pes = UnRelaxedPES1D(h3, rs={(0, 1): (1.5, 10), (1, 2): (1.5, 10)})

    with pytest.raises(NotImplementedError):
        pes.calculate(method=XTB())


@testutils.requires_working_xtb_install
@work_in_tmp_dir()
def test_single_energy_works():
    # This method is called but in parallel so codecov can't see it

    h2 = Molecule(atoms=[Atom("H"), Atom("H", x=0.8)])

    pes = UnRelaxedPES1D(h2, rs={(0, 1): (0.9, 2)})
    xtb = XTB()
    pes._method = xtb
    pes._keywords = xtb.keywords.sp
    result = pes._single_energy(h2, n_cores=1)

    assert result is not None and result != np.nan
