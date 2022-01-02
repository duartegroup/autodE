import os
import pytest
from autode.wrappers.qchem import QChem
from autode.calculation import Calculation
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.wrappers.keywords import SinglePointKeywords
from autode.wrappers.basis_sets import def2svp
from autode.wrappers.functionals import pbe0
from autode.utils import work_in_tmp_dir
from .testutils import work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
method = QChem()


def _blank_calc(name='test'):
    """Blank calculation of a single H atom"""

    calc = Calculation(name=name,
                       molecule=Molecule(atoms=[Atom('H')]),
                       method=method,
                       keywords=SinglePointKeywords())

    return calc


def _completed_thf_calc():

    calc = _blank_calc()
    calc.output.filename = 'smd_thf.out'

    assert calc.output.exists
    assert len(calc.output.file_lines) > 0

    return calc


def test_in_out_name():

    calc = _blank_calc(name='test')
    assert method.get_input_filename(calc) == 'test_qchem.in'
    assert method.get_output_filename(calc) == 'test_qchem.out'


@work_in_zipped_dir(os.path.join(here, 'data', 'qchem.zip'))
def test_version_extract():

    version = method.get_version(calc=_completed_thf_calc())

    assert version == '5.4.1'


@work_in_zipped_dir(os.path.join(here, 'data', 'qchem.zip'))
def test_thf_calc_terminated_normally():

    assert _completed_thf_calc().terminated_normally


@work_in_zipped_dir(os.path.join(here, 'data', 'qchem.zip'))
def test_terminated_abnormally():

    # Without any output the calculation cannot have terminated normally
    calc = _blank_calc()
    assert not calc.terminated_normally

    # A broken file containing one fewer H atom for an invalid 2S+1
    calc.output.filename = 'smd_thf_broken.out'
    assert calc.output.exists
    assert not calc.terminated_normally

    # If the output is not a QChem output file..
    with open('tmp.out', 'w') as tmp_out_file:
        print('not', 'a', 'out', 'file',
              sep='\n', file=tmp_out_file)

    calc = _blank_calc()
    calc.output.filename = 'tmp.out'
    assert calc.output.exists

    assert not calc.terminated_normally

    os.remove('tmp.out')


def test_blank_input_generation():

    calc = _blank_calc()
    calc.input.filename = None

    with pytest.raises(ValueError):
        method.generate_input(calc=calc, molecule=calc.molecule)


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_unsupported_keywords():

    calc = Calculation(name='test',
                       molecule=Molecule(atoms=[Atom('H')], mult=2),
                       method=method,
                       keywords=SinglePointKeywords('$rem\n'
                                                    'method b3lyp\n'
                                                    'basis 6-31G*\n'
                                                    '$end'))

    # Having $blocks in the keywords is not supported
    with pytest.raises(Exception):
        calc.generate_input()


def test_simple_input_generation():

    expected_inp = ('$molecule\n'
                    '0 2\n'
                    'H    0.00000000   0.00000000   0.00000000 \n'
                    '$end\n'
                    '$rem\n'
                    'method pbe0\n'
                    'basis def2-SVP\n'
                    '$end\n')

    # Simple PBE0/def2-SVP calculation of a hydrogen atom
    h_atom = Molecule(atoms=[Atom('H')], mult=2)
    calc = Calculation(name='H_atom',
                       molecule=h_atom,
                       method=method,
                       keywords=SinglePointKeywords([pbe0, def2svp]))

    # Generate the required input
    calc.input.filename = 'test.in'
    method.generate_input(calc, molecule=h_atom)

    # Remove any blank lines from the input file for comparison, as they are
    # ignored
    inp_lines = [line for line in open(calc.input.filename, 'r').readlines()
                 if line != '\n']

    assert ''.join(inp_lines) == expected_inp

    os.remove('test.in')
