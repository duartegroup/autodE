from autode.atoms import Atom
from autode.conformers.conformer import Conformer
from autode.wrappers.ORCA import orca
from scipy.spatial import distance_matrix
import numpy as np
import os
from autode.conformers.conformers import get_unique_confs

here = os.path.dirname(os.path.abspath(__file__))

orca.available = True


def test_conf_class():

    os.chdir(os.path.join(here, 'data', 'conformers'))

    h2_conf = Conformer(name='h2_conf', charge=0, mult=1,
                        atoms=[Atom('H', 0.0, 0.0, 0.0),
                               Atom('H', 0.0, 0.0, 0.7)])

    assert hasattr(h2_conf, 'optimise')
    assert hasattr(h2_conf, 'dist_consts')

    assert h2_conf.n_atoms == 2
    assert h2_conf.energy is None
    assert h2_conf.dist_consts is None

    h2_conf.optimise(method=orca)
    assert h2_conf.energy == -1.160780546661
    assert h2_conf.atoms is not None
    assert h2_conf.n_atoms == 2

    # Check that if the conformer calculation does not complete successfully then
    # don't raise an exception for a conformer
    h2_conf_broken = Conformer(name='h2_conf_broken', charge=0, mult=1,
                               atoms=[Atom('H', 0.0, 0.0, 0.0),
                                      Atom('H', 0.0, 0.0, 0.7)])
    h2_conf_broken.optimise(method=orca)

    assert h2_conf_broken.atoms is None
    assert h2_conf_broken.n_atoms == 0

    # Clear the generated input file
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.inp'):
            os.remove(filename)

    os.chdir(here)


def test_unique_confs():

    conf1 = Conformer()
    conf2 = Conformer()
    conf3 = Conformer()

    # Set two energies the same and leave one as none..
    conf1.energy = 1
    conf2.energy = 1

    unique_confs = get_unique_confs(conformers=[conf1, conf2, conf3])
    assert len(unique_confs) == 1
    assert type(unique_confs[0]) is Conformer
    assert unique_confs[0].energy == 1
