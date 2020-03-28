from autode.wrappers.MOPAC import MOPAC
from autode.calculation import Calculation
from autode.molecule import Molecule
from autode.constants import Constants
import os
import pytest

here = os.path.dirname(os.path.abspath(__file__))
method = MOPAC()


def test_mopac_opt_calculation():

    os.chdir(os.path.join(here, 'data'))

    methylchloride = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]', solvent_name='water')
    calc = Calculation(name='opt', molecule=methylchloride,
                       method=method, opt=True)
    calc.run()

    assert os.path.exists('opt_mopac.mop') is True
    assert os.path.exists('opt_mopac.out') is True
    assert len(calc.get_final_atoms()) == 5

    # Actual energy in Hartrees
    energy = Constants.eV2ha * -430.43191
    assert energy - 0.0001 < calc.get_energy() < energy + 0.0001

    assert calc.output_file_exists is True
    assert calc.rev_output_file_lines is not None
    assert calc.output_file_lines is not None
    assert calc.input_filename == 'opt_mopac.mop'
    assert calc.output_filename == 'opt_mopac.out'
    assert calc.terminated_normally is True
    assert calc.calculation_terminated_normally() is True
    assert calc.optimisation_converged() is True
    with pytest.raises(NotImplementedError):
        _ = calc.optimisation_nearly_converged()
    with pytest.raises(NotImplementedError):
        _ = calc.get_imag_freqs()
    with pytest.raises(NotImplementedError):
        _ = calc.get_normal_mode_displacements(4)

    os.remove('opt_mopac.mop')
    os.chdir(here)
