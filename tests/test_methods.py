from autode import methods


def test_get_hmethod():
    methods.Config.hcode = None
    method1 = methods.get_hmethod()
    assert method1 == methods.ORCA

    methods.Config.hcode = 'ORCA'
    method2 = methods.get_hmethod()
    assert method2 == methods.ORCA


def test_get_lmethod():
    methods.Config.lcode = None
    method3 = methods.get_lmethod()
    assert method3 == methods.XTB

    methods.Config.lcode = 'XTB'
    method4 = methods.get_lmethod()
    assert method4 == methods.XTB

    methods.Config.lcode = 'MOPAC'
    method5 = methods.get_lmethod()
    assert method5 == methods.MOPAC
