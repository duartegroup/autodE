import os
import pytest
import numpy as np
from autode.wrappers.QChem import QChem
from autode.calculation import Calculation
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.wrappers.keywords import SinglePointKeywords
from autode.wrappers.basis_sets import def2svp
from autode.wrappers.functionals import pbe0
from autode.utils import work_in_tmp_dir
from autode.exceptions import CalculationException
from .testutils import work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
qchem_data_zip_path = os.path.join(here, 'data', 'qchem.zip')

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


def _custom_output_calc(*lines):
    calc = _blank_calc()
    with open('tmp.out', 'w') as out_file:
        print(*lines, sep='\n', file=out_file)

    calc.output.filename = 'tmp.out'
    assert calc.output.exists

    return calc


def _broken_output_calc():
    return _custom_output_calc('a', 'broken', 'output', 'file')


def _broken_output_calc2():
    return _custom_output_calc('broken', 'Total energy')


def test_base_method():

    assert 'qchem' in repr(method).lower()


def test_in_out_name():

    calc = _blank_calc(name='test')
    assert method.get_input_filename(calc) == 'test_qchem.in'
    assert method.get_output_filename(calc) == 'test_qchem.out'


@work_in_zipped_dir(qchem_data_zip_path)
def test_version_extract():

    # Version extraction from a blank calculation should not raise an error
    assert isinstance(method.get_version(_blank_calc()), str)

    version = method.get_version(calc=_completed_thf_calc())

    assert version == '5.4.1'


@work_in_zipped_dir(qchem_data_zip_path)
def test_version_extract_broken_output_file():

    # Should not raise an exception
    version = method.get_version(_broken_output_calc())
    assert isinstance(version, str)


@work_in_zipped_dir(qchem_data_zip_path)
def test_thf_calc_terminated_normally():

    assert _completed_thf_calc().terminated_normally


@work_in_zipped_dir(qchem_data_zip_path)
def test_terminated_abnormally():

    # Without any output the calculation cannot have terminated normally
    calc = _blank_calc()
    assert not method.calculation_terminated_normally(calc)
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


@work_in_zipped_dir(qchem_data_zip_path)
def test_energy_extraction():

    calc = _completed_thf_calc()
    energy = calc.get_energy()

    assert np.isclose(energy.to('Ha'), -232.45463628, atol=1e-8)

    for calc in (_blank_calc(), _broken_output_calc(), _broken_output_calc2()):
        with pytest.raises(CalculationException):
            method.get_energy(calc)


def _file_contains_one(filename, string):
    """A file contains one line that is an exact match to a string"""
    _list = [line.strip() for line in open(filename, 'r')]
    return sum(item.lower() == string for item in _list) == 1


def _tmp_input_contains(string):
    flag = _file_contains_one(filename='tmp.in', string=string)
    os.remove('tmp.in')
    return flag


def test_jobtype_inference():
    """Check that the jobtype can be infered from the keyword type"""

    def kwd_type_has_job_type(kwd_type, job_type, remove_explicit=True):

        calc = _blank_calc()
        keywords = getattr(method.keywords, kwd_type)

        if remove_explicit:
            # Remove any explicit declaration of the job type in the keywords
            keywords = keywords.__class__([w for w in keywords
                                           if 'jobtype' not in w.lower()])

        calc.input.keywords = keywords

        with QChem._InputFileWriter('tmp.in') as inp_file:
            inp_file.add_rem_block(calc)

        return _tmp_input_contains(f'jobtype {job_type}')

    assert kwd_type_has_job_type('opt', 'opt')
    assert kwd_type_has_job_type('low_opt', 'opt')
    assert kwd_type_has_job_type('opt_ts', 'ts', remove_explicit=False)

    assert kwd_type_has_job_type('grad', 'force')
    assert kwd_type_has_job_type('hess', 'freq')


def test_ecp_writing():

    calc = _blank_calc()
    calc.input.keywords = method.keywords.sp

    def write_tmp_input():
        with QChem._InputFileWriter('tmp.in') as inp_file:
            inp_file.add_rem_block(calc)

    # No ECP for a H atom
    assert calc.molecule.n_atoms == 1 and calc.molecule.atoms[0].label == 'H'
    write_tmp_input()
    assert not _tmp_input_contains('ecp def2-ecp')

    # Should add an ECP for lead
    calc.molecule = Molecule(atoms=[Atom('Pb')])
    write_tmp_input()
    assert _tmp_input_contains('ecp def2-ecp')


@work_in_zipped_dir(qchem_data_zip_path)
def test_h2o_opt():
    """Check that the energy and geometry is extracted correctly"""

    h2o = Molecule(smiles='O')
    h2o.optimise(method=method)

    assert h2o.energy is not None
    assert np.isclose(h2o.energy.to('Ha'),
                      -76.2766126261376,
                      atol=1E-8)

    assert np.isclose(h2o.distance(0, 1).to('Å'),
                      0.962586,
                      atol=1E-5)

    assert np.isclose(h2o.angle(1, 0, 2).to('deg'),
                      103.154810,
                      atol=1E-3)


@work_in_zipped_dir(qchem_data_zip_path)
def test_gradient_extraction_h2o():

    calc = _blank_calc()
    calc.molecule = Molecule(smiles='O')
    calc.output.filename = 'H2O_opt_qchem.out'

    assert calc.output.exists

    grad = calc.get_gradients()
    assert grad.shape == (3, 3)

    # The minimum should have a gradient close to zero
    assert np.allclose(grad, np.zeros(shape=(3, 3)), atol=1E-4)


@work_in_zipped_dir(qchem_data_zip_path)
def test_butane_gradient_extraction():

    calc = _blank_calc()
    calc.output.filename = 'partial_C4H10_opt_qchem.out'
    calc.molecule = Molecule(smiles='CCCC')

    assert calc.molecule.n_atoms == 14

    grad = method.get_gradients(calc)
    assert grad.shape == (14, 3)


@work_in_zipped_dir(qchem_data_zip_path)
def test_h2o_hessian_extraction():

    calc = _blank_calc()
    calc.output.filename = 'H2O_hess_qchem.out'
    calc.molecule = Molecule(smiles='O')

    hess = method.get_hessian(calc)
    assert hess.shape == (9, 9)

    print(hess)

