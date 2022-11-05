from autode.methods import ORCA, NWChem, G09, G16, MOPAC, XTB


def test_reprs():

    assert "orca" in repr(ORCA()).lower()
    assert "nwchem" in repr(NWChem()).lower()
    assert "gaussian" in repr(G09()).lower()
    assert "gaussian" in repr(G16()).lower()
    assert "mopac" in repr(MOPAC()).lower()
    assert "xtb" in repr(XTB()).lower()
