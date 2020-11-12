from autode.atoms import Atom
from autode.conformers.conformer import Conformer
from autode.wrappers.ORCA import orca
from autode.config import Config
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.conformers.conformers import conf_is_unique_rmsd
from autode.conformers.conformers import get_unique_confs
from autode.constants import Constants
from . import testutils
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
Config.keyword_prefixes = False

orca.available = True


@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'conformers.zip'))
def test_conf_class():

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

    # Check that if the conformer calculation does not complete successfully
    # then don't raise an exception for a conformer
    h2_conf_broken = Conformer(name='h2_conf_broken', charge=0, mult=1,
                               atoms=[Atom('H', 0.0, 0.0, 0.0),
                                      Atom('H', 0.0, 0.0, 0.7)])
    h2_conf_broken.optimise(method=orca)

    assert h2_conf_broken.atoms is None
    assert h2_conf_broken.n_atoms == 0


def test_rdkit_atoms():

    mol = Chem.MolFromSmiles('C')
    mol = Chem.AddHs(mol)

    AllChem.EmbedMultipleConfs(mol, numConfs=1)

    atoms = atoms_from_rdkit_mol(rdkit_mol_obj=mol, conf_id=0)
    assert len(atoms) == 5

    coords = np.array([atom.coord for atom in atoms])
    dist_mat = distance_matrix(coords, coords)

    # No distance between the same atom
    assert dist_mat[0, 0] == 0.0

    # CH bond should be ~1 Å
    assert 0.9 < dist_mat[0, 1] < 1.2


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


def test_unique_confs_none():

    conf1 = Conformer()
    conf1.energy = 0.1

    # Conformer with energy just below the threshold
    conf2 = Conformer()
    conf2.energy = 0.1 + (0.9 / Constants.ha2kJmol)

    unique_confs = get_unique_confs(conformers=[conf1, conf2],
                                    energy_threshold_kj=1)
    assert len(unique_confs) == 1

    # If the energy is above the threshold there should be two unique
    # conformers
    conf2.energy += 0.2 / Constants.ha2kJmol
    unique_confs = get_unique_confs(conformers=[conf1, conf2],
                                    energy_threshold_kj=1)
    assert len(unique_confs) == 2


def test_rmsd_confs():

    methane1 = Conformer(name='methane1', charge=0, mult=1,
                         atoms=[Atom('C', -1.38718,  0.38899,  0.00000),
                                Atom('H', -0.27778,  0.38899, -0.00000),
                                Atom('H', -1.75698,  1.06232,  0.80041),
                                Atom('H', -1.75698, -0.64084,  0.18291),
                                Atom('H', -1.75698,  0.74551, -0.98332)])

    methane2 = Conformer(name='methane2', charge=0, mult=1,
                         atoms=[Atom('C', -1.38718,  0.38899,  0.00000),
                                Atom('H', -0.43400,  0.50158, -0.55637),
                                Atom('H', -2.23299,  0.69379, -0.64998),
                                Atom('H', -1.36561,  1.03128,  0.90431),
                                Atom('H', -1.51612, -0.67068,  0.30205)])

    # Methane but rotated should have an RMSD ~ 0 Angstroms
    assert not conf_is_unique_rmsd(conf=methane2, conf_list=[methane1],
                                   rmsd_tol=0.1)
