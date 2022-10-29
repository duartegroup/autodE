from autode.atoms import Atom, Atoms
from autode.species import Molecule, NCIComplex
from autode.conformers import Conformer, Conformers
from autode.conformers.conformers import _calc_conformer
from autode.wrappers.ORCA import ORCA
from autode.wrappers.XTB import XTB
from autode.config import Config
from autode.values import Energy
from autode.utils import work_in_tmp_dir
from autode.wrappers.keywords import SinglePointKeywords
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from autode.conformers.conformers import atoms_from_rdkit_mol
from . import testutils
import numpy as np
import pytest
import os
import shutil

here = os.path.dirname(os.path.abspath(__file__))
orca = ORCA()


@testutils.work_in_zipped_dir(os.path.join(here, "data", "conformers.zip"))
def test_conf_class():

    h2_conf = Conformer(
        name="h2_conf",
        charge=0,
        mult=1,
        atoms=[Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 0.7)],
    )

    assert "conformer" in repr(h2_conf).lower()
    assert hasattr(h2_conf, "optimise")
    assert h2_conf != "a"

    assert h2_conf.n_atoms == 2
    assert h2_conf.energy is None
    assert not h2_conf.constraints.any

    h2_conf.optimise(method=orca)
    assert h2_conf.energy == -1.160780546661
    assert h2_conf.atoms is not None
    assert h2_conf.n_atoms == 2

    # Check that if the conformer calculation does not complete successfully
    # then don't raise an exception for a conformer
    h2_conf_broken = Conformer(
        name="h2_conf_broken",
        charge=0,
        mult=1,
        atoms=[Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 0.7)],
    )
    h2_conf_broken.optimise(method=orca)

    assert h2_conf_broken.atoms is None
    assert h2_conf_broken.n_atoms == 0


def test_conf_from_species():

    h2o = Molecule(smiles="O")
    conformer = Conformer(species=h2o)
    assert conformer.n_atoms == 3
    assert conformer.mult == 1
    assert conformer.charge == 0
    assert conformer.solvent is None

    h2o_cation = Molecule(smiles="[O+H2]", charge=1, solvent_name="water")
    assert h2o_cation.charge == 1

    conformer = Conformer(species=h2o_cation)
    assert conformer.n_atoms == 3
    assert conformer.mult == 2
    assert conformer.charge == 1

    assert conformer.solvent is not None
    assert conformer.solvent.name == "water"


def test_rdkit_atoms():

    mol = Chem.MolFromSmiles("C")
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

    conf1 = Conformer(atoms=[Atom("H")])
    confs = Conformers([conf1])

    # Shouldn't prune a single conformer
    confs.prune_on_rmsd()
    assert len(confs) == 1
    confs.prune_on_energy()
    assert len(confs) == 1

    conf2 = Conformer(atoms=[Atom("H")])

    # with no energies no conformers should be pruned on energy
    confs = Conformers([conf1, conf2])
    confs.prune_on_energy()
    assert len(confs) == 2

    conf3 = Conformer(atoms=[Atom("H")])
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

    conf1 = Conformer(atoms=[Atom("H")])
    conf1.energy = 1
    assert conf1.energy.units == "Ha"

    conf2 = Conformer(atoms=[Atom("H")])
    conf2.energy = 1.1

    confs = Conformers([conf1, conf2])
    assert len(confs) == 2

    # If the threshold is smaller than the difference then should leave both
    confs.prune_on_energy(e_tol=Energy(0.05, units="Ha"))
    assert len(confs) == 2

    # but not if the threshold is larger
    confs.prune_on_energy(e_tol=Energy(0.2, units="Ha"))
    assert len(confs) == 1


def test_confs_energy_pruning3():

    n = 100

    #                              μ         α
    energies = np.random.normal(loc=0.0, scale=0.1, size=n)
    confs = Conformers([Conformer(atoms=[Atom("H")]) for _ in range(n)])
    for conf, energy in zip(confs, energies):
        conf.energy = energy

    diff_e_conf = Conformer(atoms=[Atom("H")])
    diff_e_conf.energy = 3.0
    confs.append(diff_e_conf)

    # Should remove the conformer with the very different energy
    confs.prune_on_energy(e_tol=Energy(1e-10), n_sigma=5)
    assert len(confs) == 100


def test_confs_no_energy_pruning():
    # Check that if energies are unassigned then conformers are removed

    confs = Conformers([Conformer(atoms=[Atom("H")])])
    confs.prune(remove_no_energy=True)

    assert len(confs) == 0


def test_confs_rmsd_pruning1():

    confs = Conformers(
        [Conformer(atoms=[Atom("H")]), Conformer(atoms=[Atom("H")])]
    )

    # Same two structures -> one when pruned
    confs.prune_on_rmsd()
    assert len(confs) == 1


def test_confs_rmsd_pruning2():

    confs = Conformers(
        [
            Conformer(atoms=[Atom("H", x=-1.0), Atom("O"), Atom("H", x=1.0)]),
            Conformer(atoms=[Atom("H", x=-1.0), Atom("O"), Atom("H", x=10.0)]),
        ]
    )

    # Should check only on heavy atoms, thus these two 'water' molecules
    # have a 0 RMSD
    confs.prune_on_rmsd()
    assert len(confs) == 1


def test_confs_rmsd_puning3():

    # Butane molecules, without hydrogen atoms
    trans_butane = Conformer(
        atoms=[
            Atom("C", -0.86310, -0.72859, 0.62457),
            Atom("C", 0.10928, -0.05429, 1.42368),
            Atom("C", 1.17035, -0.79134, 2.03167),
            Atom("C", 2.14109, -0.11396, 2.83018),
        ]
    )

    cis_butane = Conformer(
        atoms=[
            Atom("C", -0.07267, 1.32686, 1.73687),
            Atom("C", 0.10928, -0.05429, 1.42368),
            Atom("C", 1.17035, -0.79134, 2.03167),
            Atom("C", 2.14109, -0.11396, 2.83018),
        ]
    )

    confs = Conformers([trans_butane, cis_butane])

    confs.prune_on_rmsd(rmsd_tol=0.1)
    assert len(confs) == 2

    confs.prune_on_rmsd(rmsd_tol=0.5)
    assert len(confs) == 1


@testutils.work_in_zipped_dir(os.path.join(here, "data", "sp_conformers.zip"))
def test_sp_hmethod():

    Config.hmethod_sp_conformers = True
    orca.keywords.low_sp = SinglePointKeywords(["PBE", "D3BJ", "def2-SVP"])

    conf = Conformer(name="C4H10_conf0", species=Molecule(smiles="CCCC"))

    conf.single_point(method=orca)
    assert conf.energy is not None

    Config.hmethod_sp_conformers = False


@testutils.work_in_zipped_dir(os.path.join(here, "data", "sp_conformers.zip"))
def test_sp_hmethod_ranking():

    Config.hmethod_sp_conformers = True
    orca.keywords.low_sp = None

    butane = Molecule(smiles="CCCC")
    xtb = XTB()
    cwd = os.getcwd()

    # Need to set hmethod.low_sp for hmethod_sp_conformers
    with pytest.raises(AssertionError):
        butane.find_lowest_energy_conformer(lmethod=xtb, hmethod=orca)

    # work_in function will change directories and may not change back when an
    # exception is raised, so ensure we're working in the same dir as before
    os.chdir(cwd)

    orca.keywords.low_sp = SinglePointKeywords(["PBE", "D3BJ", "def2-SVP"])
    butane.find_lowest_energy_conformer(lmethod=xtb, hmethod=orca)
    assert butane.energy is not None

    Config.hmethod_sp_conformers = False


def test_calculation_over_no_conformers():

    confs = Conformers()
    confs.single_point(method=orca)

    # Should not raise an exception
    assert len(confs) == 0


def test_complex_conformers_diff_names():

    Config.num_complex_sphere_points = 2
    Config.num_complex_random_rotations = 2

    water = Molecule(smiles="O")
    h2o_dimer = NCIComplex(water, water, name="dimer")
    h2o_dimer._generate_conformers()
    assert len(set(conf.name for conf in h2o_dimer.conformers)) > 1

    if os.path.exists("conformers"):
        shutil.rmtree("conformers")


@testutils.requires_with_working_xtb_install
@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calc_conformer():

    h2_conf = Conformer(
        name="h2_conf",
        charge=0,
        mult=1,
        atoms=[Atom("H", 0.0, 0.0, 0.0), Atom("H", 0.0, 0.0, 0.7)],
    )

    _xtb = XTB()
    _xtb.path = shutil.which("xtb")
    assert _xtb.is_available

    h2_conf = _calc_conformer(
        conformer=h2_conf,
        calc_type="single_point",
        method=_xtb,
        keywords=_xtb.keywords.sp,
        n_cores=1,
    )

    assert h2_conf.energy is not None


def test_conformers_inherit_atom_classes():
    def has_correct_atom_class(_species):
        return sum(atom.atom_class == 1 for atom in _species.atoms) == 1

    mol = Molecule(smiles="C[Br:1]")
    assert has_correct_atom_class(mol)

    mol.populate_conformers(n_confs=2)
    assert len(mol.conformers) > 0  # Should generate one conformer

    assert has_correct_atom_class(mol.conformers[0])


def test_conformer_coordinate_setting_no_atoms():

    conf = Conformer()
    assert conf.atoms is None
    assert conf.coordinates is None

    # Cannot set the coordinates without any atoms
    with pytest.raises(ValueError):
        conf.coordinates = np.array([0.0, 0.0, 0.0])


def test_conformer_coordinate_setting_with_atoms():
    # Setting the atoms should also set coordinates
    conf = Conformer()

    conf.atoms = Atoms([Atom("H")])
    assert conf.coordinates is not None
    assert np.allclose(conf.coordinates, np.zeros(shape=(1, 3)))
    assert conf.atoms is not None

    # Discarding the atoms should also discard the coordinates
    conf.atoms = None
    assert conf.coordinates is None


def test_conformer_coordinate_setting_with_different_atomic_attr():
    # Atom classes should persist
    conf = Conformer(species=Molecule(smiles="[C:9]"))

    def has_correct_atom_class(_species):
        return sum(atom.atom_class == 9 for atom in _species.atoms) == 1

    assert has_correct_atom_class(conf)
    conf.atoms = Atoms([Atom("C")])
    assert has_correct_atom_class(conf)
    conf.coordinates = np.ones_like(conf.coordinates)
    assert has_correct_atom_class(conf)

    # But cannot set atoms with a different label (e.g. atomic symbol)
    with pytest.raises(ValueError):
        conf.atoms = Atoms([Atom("H")])
