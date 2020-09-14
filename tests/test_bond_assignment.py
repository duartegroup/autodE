from autode import bond_lengths
from autode.atoms import Atom


def test_bond_assignment():

    # Check that if the pair isn't found then a length is still returned
    assert 1.4 < bond_lengths.get_avg_bond_length(atom_i_label='X',
                                                  atom_j_label='X') < 1.6

    # CH bond should be ~1.1 Å
    assert 0.8 < bond_lengths.get_avg_bond_length(atom_i_label='C',
                                                  atom_j_label='H') < 1.2


def test_get_ideal_bond_length_matrix():

    atoms = [Atom('H', 0.0, 0.0, 0.0), Atom('H', 1.0, 0.0, 0.0)]
    bond_list = [(0, 1)]

    matrix = bond_lengths.get_ideal_bond_length_matrix(atoms=atoms,
                                                       bonds=bond_list)
    assert matrix.shape == (2, 2)
    assert 0.5 < matrix[0, 1] < 1.0   # H-H ~ 0.7 Å
