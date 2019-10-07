from autode import substitution
from autode import bond_rearrangement
import numpy as np

rearrang = bond_rearrangement.BondRearrangement([(0, 1)], [(1, 2)])
coords = np.array(([0, 0, 0], [0, 0, 1], [0, 2, 1]))


def test_get_attacked_atom():
    attacked_atom = substitution.get_attacked_atom(rearrang)
    assert attacked_atom == 1


def test_get_lg_or_fr_atom():
    fr_atom = substitution.get_lg_or_fr_atom([(0, 1)], 1)
    assert fr_atom == 0


def test_get_normalised_lg_vector():
    lg_vector = substitution.get_normalised_lg_vector(rearrang, 1, coords)
    np.testing.assert_array_equal(lg_vector, np.array([0., -1., 0, ]))
