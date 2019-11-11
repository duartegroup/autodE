from autode import atoms


def test_atoms():

    assert atoms.get_maximal_valance(atom_label='C') == 4
    assert atoms.get_maximal_valance(atom_label='Rf') == 6
    assert atoms.get_atomic_weight(atom_label='C') == 12
    assert atoms.get_atomic_weight(atom_label='Rf') == 70
