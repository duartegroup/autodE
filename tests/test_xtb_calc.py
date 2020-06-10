import pytest
from autode.wrappers.XTB import XTB
from autode.calculation import Calculation
from autode.species.molecule import Molecule
from autode.config import Config
import os
here = os.path.dirname(os.path.abspath(__file__))

method = XTB()
method.available = True


def test_xtb_calculation():

    os.chdir(os.path.join(here, 'data'))
    XTB.available = True

    test_mol = Molecule(name='test_mol',
                        smiles='O=C(C=C1)[C@@](C2NC3C=C2)([H])[C@@]3([H])C1=O')
    calc = Calculation(name='opt', molecule=test_mol, method=method,
                       keywords=Config.XTB.keywords.opt)
    calc.run()

    assert os.path.exists('opt_xtb.xyz') is True
    assert os.path.exists('opt_xtb.out') is True
    assert len(calc.get_final_atoms()) == 22
    assert calc.get_energy() == -36.990267613593
    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.input.filename == 'opt_xtb.xyz'
    assert calc.output.filename == 'opt_xtb.out'

    with pytest.raises(NotImplementedError):
        calc.optimisation_nearly_converged()
    with pytest.raises(NotImplementedError):
        calc.get_imaginary_freqs()
    with pytest.raises(NotImplementedError):
        calc.get_normal_mode_displacements(4)

    charges = calc.get_atomic_charges()
    assert len(charges) == 22
    assert all(-1.0 < c < 1.0 for c in charges)

    const_opt = Calculation(name='const_opt', molecule=test_mol,
                            method=method,
                            distance_constraints={(0, 1): 1.2539792},
                            cartesian_constraints=[0],
                            keywords=Config.XTB.keywords.opt)

    const_opt.generate_input()
    assert os.path.exists('xcontrol_const_opt_xtb')

    os.remove('const_opt_xtb.xyz')
    os.remove('xcontrol_const_opt_xtb')
    os.remove('opt_xtb.xyz')
    os.chdir(here)
