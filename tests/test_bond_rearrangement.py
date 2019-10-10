from autode import bond_rearrangement
from autode.molecule import Molecule

# Reaction H + H2 -> H2 + H
rearrag = bond_rearrangement.BondRearrangement(forming_bonds=[(0, 1)],
                                               breaking_bonds=[(1, 2)])


def test_bondrearr_obj():

    assert rearrag.n_fbonds == 1
    assert rearrag.n_bbonds == 1

    rearrag2 = bond_rearrangement.BondRearrangement(forming_bonds=[(0, 1)],
                                                    breaking_bonds=[(1, 2)])
    assert rearrag2 == rearrag

    h2_h_mol = Molecule(name='mol', xyzs=[['H', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, -1.0], ['H', 0.0, 0.0, 1.0]])
    active_atom_nl = rearrag.get_active_atom_neighbour_lists(mol=h2_h_mol, depth=1)
    assert len(active_atom_nl) == 4
    assert active_atom_nl == [['H'], ['H'], ['H'], ['H']]
