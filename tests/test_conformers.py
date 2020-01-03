from autode.conformers import conformers


def test_reasonable_conformers():
    h2_good = [[['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]]
    h2_good_check = conformers.rdkit_conformer_geometries_are_resonable(h2_good)
    assert h2_good_check == True

    h4_flat = [[['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0], ['H', 2.0, 0.0, 0.0], ['H', 3.0, 0.0, 0.0]]]
    h4_flat_check = conformers.rdkit_conformer_geometries_are_resonable(h4_flat)
    assert h4_flat_check == False

    h4_too_close = [[['H', 0.0, 0.0, 0.0], ['H', 0.4, 0.0, 0.4], ['H', 2.0, 0.0, 1.0], ['H', 3.0, 0.0, 1.0]]]
    h4_too_close_check = conformers.rdkit_conformer_geometries_are_resonable(h4_too_close)
    assert h4_too_close_check == False
