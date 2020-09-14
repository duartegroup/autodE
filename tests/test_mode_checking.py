from autode.calculation import Calculation
from autode.transition_states.base import imag_mode_has_correct_displacement
from autode.species.molecule import Reactant
from autode.input_output import xyz_file_to_atoms
from autode.bond_rearrangement import BondRearrangement
from autode.methods import ORCA
from . import testutils
import os

here = os.path.dirname(os.path.abspath(__file__))
orca = ORCA()


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'mode_checking.zip'))
def test_imag_modes():

    assert not has_correct_mode('incorrect_ts_mode',
                                bbonds=[(1, 6)],
                                fbonds=[(2, 6)])

    assert has_correct_mode('correct_ts_mode',
                            bbonds=[(0, 1), (1, 2)],
                            fbonds=[(0, 2)])

    assert has_correct_mode('correct_ts_mode_2',
                            bbonds=[(3, 8), (3, 2)],
                            fbonds=[(2, 8)])


def has_correct_mode(name, fbonds, bbonds):

    reac = Reactant(name='r', atoms=xyz_file_to_atoms(f'{name}.xyz'))

    calc = Calculation(name=name,
                       molecule=reac,
                       method=orca,
                       keywords=orca.keywords.opt_ts,
                       n_cores=1)

    calc.output.filename = f'{name}.out'
    calc.output.file_lines = open(f'{name}.out', 'r').readlines()

    bond_rearr = BondRearrangement(breaking_bonds=bbonds,
                                   forming_bonds=fbonds)

    # Don't require all bonds to be breaking/making in a 'could be ts' function
    return imag_mode_has_correct_displacement(calc, bond_rearr,
                                              delta_threshold=0.05,
                                              req_all=False)
