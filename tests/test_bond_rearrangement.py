from autode import bond_rearrangement

# Reaction H + H2 -> H2 + H
rearrag = bond_rearrangement.BondRearrangement(
    forming_bonds=[(0, 1)], breaking_bonds=[(1, 2)])


def test_bondrearr_obj():

    assert rearrag.n_fbonds == 1
    assert rearrag.n_bbonds == 1

    rearrag2 = bond_rearrangement.BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(1, 2)])
    assert rearrag2 == rearrag
