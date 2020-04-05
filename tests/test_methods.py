from autode import methods
from autode import Config
from autode.exceptions import MethodUnavailable
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_get_hmethod():
    Config.hcode = None
    Config.ORCA.path = here       # A path that exists

    method1 = methods.get_hmethod()
    assert method1.name == 'orca'

    methods.Config.hcode = 'orca'
    method2 = methods.get_hmethod()
    assert method2.name == 'orca'


def test_get_lmethod():
    Config.lcode = None
    Config.XTB.path = here

    method3 = methods.get_lmethod()
    assert method3.name == 'xtb'

    Config.lcode = 'xtb'
    method4 = methods.get_lmethod()
    assert method4.name == 'xtb'

    Config.lcode = 'mopac'
    Config.MOPAC.path = here

    method4 = methods.get_lmethod()
    assert method4.name == 'mopac'


def test_method_unavalible():

    Config.ORCA.path = None
    Config.NWChem.path = None
    Config.G09.path = None

    with pytest.raises(MethodUnavailable):
        methods.get_hmethod()
