from autode import methods
from autode import Config
from autode.exceptions import MethodUnavailable
from autode.wrappers.XTB import XTB
from autode.wrappers.ORCA import ORCA
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_get_hmethod():
    Config.hcode = None
    Config.ORCA.path = here  # A path that exists

    method1 = methods.get_hmethod()
    assert method1.name == "orca"

    methods.Config.hcode = "orca"
    method2 = methods.get_hmethod()
    assert method2.name == "orca"

    Config.hcode = "g09"
    Config.G09.path = here
    method3 = methods.get_hmethod()
    assert method3.name == "g09"

    Config.hcode = "NwChem"
    Config.NWChem.path = here
    method4 = methods.get_hmethod()
    assert method4.name == "nwchem"

    with pytest.raises(MethodUnavailable):
        Config.hcode = "x"
        methods.get_hmethod()


def test_get_lmethod():
    Config.lcode = None
    Config.XTB.path = here

    method3 = methods.get_lmethod()
    assert method3.name == "xtb"

    Config.lcode = "xtb"
    method4 = methods.get_lmethod()
    assert method4.name == "xtb"

    Config.lcode = "mopac"
    Config.MOPAC.path = here

    method4 = methods.get_lmethod()
    assert method4.name == "mopac"

    # Back to default
    Config.lcode = None


def test_method_unavailable():

    Config.hcode = None

    Config.ORCA.path = "/an/incorrect/path"
    Config.NWChem.path = "/an/incorrect/path"
    Config.G09.path = "/an/incorrect/path"

    with pytest.raises(MethodUnavailable):
        methods.get_hmethod()

    # Specifying a method that with an executable that doesn't exist should
    # raise an error
    Config.hcode = "ORCA"

    with pytest.raises(MethodUnavailable):
        methods.get_hmethod()


def test_nwchem_ecps():

    nwchem = methods.NWChem()
    assert nwchem.keywords.opt.ecp is not None
    assert nwchem.keywords.sp.ecp is not None
    assert nwchem.keywords.sp.ecp.nwchem is not None


def test_method_equality():

    orca = methods.ORCA()
    g09 = methods.G09()

    assert orca == methods.ORCA()
    assert orca != g09

    orca.keywords.sp = "Some different keywords"
    default_orca = methods.ORCA()

    # Single point keywords are different, so the methods are different
    assert orca.keywords.sp != default_orca.keywords.sp
    assert orca != methods.ORCA()


def test_get_method_or_default_lmethod():  # l <=> lower

    Config.lcode = None
    Config.XTB.path = here  # spoof an XTB install

    assert methods.method_or_default_lmethod(None) is not None
    assert isinstance(methods.method_or_default_lmethod(XTB()), XTB)


def test_get_method_or_default_hmethod():  # h <=> higher

    Config.hcode = None
    Config.ORCA.path = here  # spoof an XTB install

    assert methods.method_or_default_hmethod(None) is not None
    assert isinstance(methods.method_or_default_lmethod(ORCA()), ORCA)
