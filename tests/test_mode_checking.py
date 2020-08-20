from autode.calculation import Calculation
from autode.transition_states.base import imag_mode_has_correct_displacement
from autode.species.molecule import Reactant
from autode.input_output import xyz_file_to_atoms
from autode.bond_rearrangement import BondRearrangement
from autode.methods import ORCA
import os

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, 'data', 'mode_checking')
orca = ORCA()


def test_incorrect_imag_mode():

    assert not has_correct_mode('incorrect_ts_mode',
                                bbonds=[(1, 6)],
                                fbonds=[(2, 6)])


def test_correct_mode1():

    assert has_correct_mode('correct_ts_mode',
                            bbonds=[(0, 1), (1, 2)],
                            fbonds=[(0, 2)])


def test_correct_mode2():

    assert has_correct_mode('correct_ts_mode_2',
                            bbonds=[(3, 8), (3, 2)],
                            fbonds=[(2, 8)])


def has_correct_mode(name, fbonds, bbonds):

    xyz_path = os.path.join(data_path, f'{name}.xyz')
    reac = Reactant(name='r', atoms=xyz_file_to_atoms(xyz_path))

    calc = Calculation(name=name,
                       molecule=reac,
                       method=orca,
                       keywords=orca.keywords.opt_ts,
                       n_cores=1)

    output_path = os.path.join(data_path, f'{name}.out')
    calc.output.filename = output_path
    calc.output.file_lines = open(output_path, 'r').readlines()

    bond_rearr = BondRearrangement(breaking_bonds=bbonds,
                                   forming_bonds=fbonds)

    # Don't require all bonds to be breaking/making in a 'could be ts' function
    return imag_mode_has_correct_displacement(calc, bond_rearr,
                                              delta_threshold=0.05,
                                              req_all=False)
