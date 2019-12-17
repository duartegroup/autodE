from autode.wrappers.XTB import XTB
from autode.calculation import Calculation
from autode.molecule import Molecule
import os
here = os.path.dirname(os.path.abspath(__file__))
import pytest


def test_xtb_calculation():

    os.chdir(os.path.join(here, 'data'))
    XTB.available = True

    test_mol = Molecule(name='test_mol', smiles='O=C(C=C1)[C@@](C2NC3C=C2)([H])[C@@]3([H])C1=O')
    calc = Calculation(name='opt', molecule=test_mol, method=XTB, opt=True)
    calc.run()

    assert os.path.exists('opt_xtb.xyz') is True
    assert os.path.exists('opt_xtb.out') is True
    assert len(calc.get_final_xyzs()) == 22
    assert calc.get_energy() == -36.962231783045
    assert calc.output_file_exists is True
    assert calc.rev_output_file_lines is not None
    assert calc.output_file_lines is not None
    assert calc.input_filename == 'opt_xtb.xyz'
    assert calc.output_filename == 'opt_xtb.out'
    with pytest.raises(NotImplementedError):
        _ = calc.optimisation_nearly_converged()
    with pytest.raises(NotImplementedError):
        _ = calc.get_imag_freqs()
    with pytest.raises(NotImplementedError):
        _ = calc.get_normal_mode_displacements(4)

    const_opt = Calculation(name='opt', molecule=test_mol,
                            method=XTB, opt=True, distance_constraints={(0, 1): 1.2539792})
    const_opt.generate_input()
    assert os.path.exists('xcontrol_opt')
    assert const_opt.flags == ['--chrg', '0',
                               '--opt', '--input', 'xcontrol_opt']

    os.remove('opt_xtb.xyz')
    os.remove('xcontrol_opt')
    os.chdir(here)
