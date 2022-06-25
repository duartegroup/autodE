import pytest

from autode.species.molecule import Molecule
from autode.atoms import Atom
from autode.values import Distance
from autode.opt.optimisers.baker_crfo import CRFOptimiser


def water_molecule():
    m = Molecule(name='water', charge=0, mult=1,
                 atoms=[Atom('O', -0.00110, 0.36310, 0.00000),
                        Atom('H', -0.82500, -0.18190, 0.00000),
                        Atom('H', 0.82610, -0.18120, 0.00000)])
    return m


def test_coordinate_setup():

    mol = water_molecule()
    mol.constraints.distance = {(0, 1): Distance(1., "Å")}

    dist_consts = mol.constraints.distance
    assert (0, 1) in dist_consts and (1, 0) in dist_consts

    opt = CRFOptimiser(maxiter=1, gtol=1E-5, etol=1E-5)

    with pytest.raises(RuntimeError):
        # Cannot set coordinates without a species
        opt._set_initial_coordinates()

    opt._species = mol
    opt._set_initial_coordinates()
    assert opt.n_constraints == 1
