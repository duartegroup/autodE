from autode import bonds
from autode.atoms import Atom


def test_bond_assignment():

    # Check that if the pair isn't found then a length is still returned
    assert 1.4 < bonds.get_avg_bond_length(atom_i_label='X',
                                           atom_j_label='X') < 1.6

    # Check that if the pair isn't found but the VdW radii are defined then
    # return something sensible
    assert 2.0 < bonds.get_avg_bond_length(atom_i_label='Ir',
                                           atom_j_label='As') < 3.0

    # CH bond should be ~1.1 Å
    assert 0.8 < bonds.get_avg_bond_length(atom_i_label='C',
                                           atom_j_label='H') < 1.2


def test_get_ideal_bond_length_matrix():

    atoms = [Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)]
    bond_list = [(0, 1)]

    matrix = bonds.get_ideal_bond_length_matrix(atoms=atoms,
                                                bonds=bond_list)
    assert matrix.shape == (2, 2)
    assert 0.5 < matrix[0, 1] < 1.0   # H-H ~ 0.7 Å
