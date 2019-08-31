from autode.wrappers.wrappers import ORCA
from autode.calculation import Calculation
from autode.molecule import Molecule
import os


def test_orca_opt_calculation():

    test_mol = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent='water')
    calc = Calculation(name='opt', molecule=test_mol, method=ORCA, opt=True,
                       keywords=['Opt', 'PBE0', 'RIJCOSX', 'D3BJ', 'def2-SVP', 'def2/J'])
    calc.run()

    assert os.path.exists('opt_orca.inp') is True
    assert os.path.exists('opt_orca.out') is True
    assert len(calc.get_final_xyzs()) == 5
    assert calc.get_energy() == -499.730559818596
    assert calc.output_file_exists is True
    assert calc.rev_output_file_lines is not None
    assert calc.output_file_lines is not None
    assert calc.get_imag_freqs() is None
    assert calc.get_normal_mode_displacements(mode_number=1) is None
    assert calc.get_scan_values_xyzs_energies() is None
    assert calc.input_filename == 'opt_orca.inp'
    assert calc.output_filename == 'opt_orca.out'
    assert calc.terminated_normally is True
    assert calc.calculation_terminated_normally() is True
    assert calc.optimisation_converged() is True
    assert calc.optimisation_nearly_converged() is False

    os.remove('opt_orca.inp')
