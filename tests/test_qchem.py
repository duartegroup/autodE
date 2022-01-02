from autode.wrappers.qchem import QChem
from autode.calculation import Calculation
from autode.atoms import Atom
from autode.species.molecule import Molecule
from autode.wrappers.keywords import SinglePointKeywords

method = QChem()


def _blank_calc(name='test'):
    """Blank calculation of a single H atom"""

    calc = Calculation(name=name,
                       molecule=Molecule(atoms=[Atom('H')]),
                       method=method,
                       keywords=SinglePointKeywords())

    return calc


def test_in_out_name():

    calc = _blank_calc(name='test')
    assert method.get_input_filename(calc) == 'test_qchem.in'
    assert method.get_output_filename(calc) == 'test_qchem.out'


def test_version_extract():

    calc = _blank_calc()
    calc.output.filename = 'smd_thf.out'

    assert calc.output.exists

    version = method.get_version(calc=calc)
    assert version == '5.4.1'
