from autode.wrappers.XTB import XTB
from autode.calculation import Calculation
from autode.molecule import Molecule
import os
cwd = os.getcwd()
here = os.path.dirname(os.path.abspath(__file__))


def test_xtb_calculation():

    os.chdir(here)

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

    os.remove('opt_xtb.xyz')
    os.chdir(cwd)
