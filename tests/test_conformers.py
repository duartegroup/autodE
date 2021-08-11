from autode.atoms import Atom
from autode.species import Molecule
from autode.conformers import Conformer, Conformers
from autode.wrappers.ORCA import orca
from autode.wrappers.XTB import XTB
from autode.config import Config
from autode.values import Energy
from autode.wrappers.keywords import SinglePointKeywords
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from autode.conformers.conformers import atoms_from_rdkit_mol
from . import testutils
import numpy as np
import pytest
import os

here = os.path.dirname(os.path.abspath(__file__))


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


def test_confs_energy_pruning1():

    conf1 = Conformer(atoms=[Atom('H')])
    conf2 = Conformer(atoms=[Atom('H')])

    # with no energies no conformers should be pruned on energy
    confs = Conformers([conf1, conf2])
    confs.prune_on_energy()
    assert len(confs) == 2

    conf3 = Conformer(atoms=[Atom('H')])
    confs = Conformers([conf1, conf2, conf3])

    # Set two energies the same and leave one as none..
    conf1.energy = 1
    conf2.energy = 1

    # which should prune to two, one with energy = 1 and one energy = None
    confs.prune_on_energy()
    assert len(confs) == 2

    # If they all have an energy then they should prune to a single conformer
    conf3.energy = 1
    confs.prune_on_energy()
    assert len(confs) == 1


def test_confs_energy_pruning2():

    conf1 = Conformer(atoms=[Atom('H')])
    conf1.energy = 1
    assert conf1.energy.units == 'Ha'

    conf2 = Conformer(atoms=[Atom('H')])
    conf2.energy = 1.1

    confs = Conformers([conf1, conf2])
    assert len(confs) == 2

    # If the threshold is smaller than the difference then should leave both
    confs.prune_on_energy(e_tol=Energy(0.05, units='Ha'))
    assert len(confs) == 2

    # but not if the threshold is larger
    confs.prune_on_energy(e_tol=Energy(0.2, units='Ha'))
    assert len(confs) == 1


def test_confs_energy_pruning3():

    n = 100

    #                              μ         α
    energies = np.random.normal(loc=0.0, scale=0.1, size=n)
    confs = Conformers([Conformer(atoms=[Atom('H')]) for _ in range(n)])
    for conf, energy in zip(confs, energies):
        conf.energy = energy

    diff_e_conf = Conformer(atoms=[Atom('H')])
    diff_e_conf.energy = 3.0
    confs.append(diff_e_conf)

    # Should remove the conformer with the very different energy
    confs.prune_on_energy(e_tol=Energy(1E-10), n_sigma=5)
    assert len(confs) == 100



"""
def test_unique_confs_none():

    conf1 = Conformer(atoms=[Atom('H')])
    conf1.energy = 0.1

    # Conformer with energy just below the threshold
    conf2 = Conformer(atoms=[Atom('H')])
    conf2.energy = 0.19

    unique_confs = get_unique_confs(conformers=[conf1, conf2],
                                    energy_threshold=Energy(0.1))
    assert len(unique_confs) == 1

    # If the energy is above the threshold there should be two unique
    # conformers
    conf2.energy += 0.2
    unique_confs = get_unique_confs(conformers=[conf1, conf2],
                                    energy_threshold=Energy(0.1))
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

"""

@testutils.work_in_zipped_dir(os.path.join(here, 'data', 'sp_conformers.zip'))
def _test_sp_hmethod_ranking():

    Config.hmethod_sp_conformers = True

    butane = Molecule(smiles='CCCC')
    xtb = XTB()
    cwd = os.getcwd()

    # Need to set hmethod.low_sp
    with pytest.raises(AssertionError):
        butane.find_lowest_energy_conformer(lmethod=xtb,
                                            hmethod=orca)

    # work_in function will change directories and not change back when an
    # exception is raised, so ensure we're working in the same dir as before
    os.chdir(cwd)

    orca.keywords.low_sp = SinglePointKeywords(['PBE', 'D3BJ', 'def2-SVP'])
    butane.find_lowest_energy_conformer(lmethod=xtb,
                                        hmethod=orca)
    assert butane.energy is not None

    Config.hmethod_sp_conformers = False
