from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.MOPAC import get_keywords
from autode.calculation import Calculation, CalculationInput
from autode.species.molecule import Molecule
from autode.constants import Constants
from autode.config import Config
import os
import pytest

here = os.path.dirname(os.path.abspath(__file__))
method = MOPAC()
method.available = True

methylchloride = Molecule(name='CH3Cl', smiles='[H]C([H])(Cl)[H]',
                          solvent_name='water')


def test_mopac_opt_calculation():

    os.chdir(os.path.join(here, 'data'))
    calc = Calculation(name='opt', molecule=methylchloride,
                       method=method, keywords=Config.MOPAC.keywords.opt)
    calc.run()

    assert os.path.exists('opt_mopac.mop') is True
    assert os.path.exists('opt_mopac.out') is True
    assert len(calc.get_final_atoms()) == 5

    # Actual energy in Hartrees
    energy = Constants.eV2ha * -430.43191
    assert energy - 0.0001 < calc.get_energy() < energy + 0.0001

    assert calc.output.exists()
    assert calc.output.file_lines is not None
    assert calc.input.filename == 'opt_mopac.mop'
    assert calc.output.filename == 'opt_mopac.out'
    assert calc.terminated_normally()
    assert calc.optimisation_converged() is True

    with pytest.raises(NotImplementedError):
        _ = calc.optimisation_nearly_converged()
    with pytest.raises(NotImplementedError):
        _ = calc.get_imaginary_freqs()
    with pytest.raises(NotImplementedError):
        _ = calc.get_normal_mode_displacements(4)

    os.remove('opt_mopac.mop')
    os.chdir(here)


def test_mopac_keywords():

    calc_input = CalculationInput(keywords=Config.MOPAC.keywords.sp,
                                  solvent=None,
                                  added_internals=None,
                                  additional_input=None,
                                  point_charges=None)

    keywords = get_keywords(calc_input=calc_input, molecule=methylchloride)
    assert any('1scf' == kw.lower() for kw in keywords)

    calc_input.keywords = Config.MOPAC.keywords.grad
    keywords = get_keywords(calc_input=calc_input, molecule=methylchloride)
    assert any('grad' == kw.lower() for kw in keywords)

    h = Molecule(name='H', smiles='[H]')
    keywords = get_keywords(calc_input=calc_input, molecule=h)
    assert any('doublet' == kw.lower() for kw in keywords)
