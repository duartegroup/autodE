from autode.wrappers.G16 import G16


def test_g16():

    # Identical to G09 so tests are implemented in G09
    g16 = G16()
    assert g16.name == "g16"
