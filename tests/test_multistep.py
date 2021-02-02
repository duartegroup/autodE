from autode.config import Config
from autode.reactions.reaction import Reaction
from autode.reactions.multistep import MultiStepReaction
from autode.wrappers.implicit_solvent_types import cpcm
from . import testutils
import shutil
import os

here = os.path.dirname(os.path.abspath(__file__))


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'multistep.zip'))
def test_multistep_reaction():

    Config.num_conformers = 1
    Config.keyword_prefixes = True

    # Spoof installs
    Config.lcode = 'xtb'
    Config.XTB.path = here

    Config.hcode = 'orca'
    Config.ORCA.path = here

    # Don't run the calculation without a working XTB install
    if shutil.which('xtb') is None or not shutil.which('xtb').endswith('xtb'):
        return

    Config.XTB.path = shutil.which('xtb')

    Config.ORCA.implicit_solvation_type = cpcm
    Config.make_ts_template = False
    Config.num_complex_sphere_points = 2
    Config.num_complex_random_rotations = 1

    # SN2 forwards then backwards example
    forwards = Reaction('CCl.[F-]>>CF.[Cl-]',
                        name='sn2_forwards',
                        solvent_name='water')

    backwards = Reaction('CF.[Cl-]>>CCl.[F-]',
                         name='sn2_backwards',
                         solvent_name='water')

    reaction = MultiStepReaction(forwards, backwards)
    reaction.calculate_reaction_profile()

    assert reaction.reactions is not None
    assert len(reaction.reactions) == 2
    assert reaction.reactions[0].ts is not None
    Config.keyword_prefixes = False
