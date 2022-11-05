import os
import numpy as np
import pytest

from autode.atoms import Atom
from autode.methods import XTB
from autode.path import Path, AdaptivePath, PathPoint
from autode.path.adaptive import pruned_active_bonds
from autode.bonds import FormingBond, BreakingBond
from autode.species import Species, Molecule
from autode.units import Unit, KcalMol

test_species = Species(name="tmp", charge=0, mult=1, atoms=[Atom("He")])
test_mol = Molecule(smiles="O")


def test_path_properties_empty():

    path = Path()

    assert len(path) == 0
    assert isinstance(path.units, Unit)

    assert path == Path()  # should be able to compare paths
    assert path != 0

    with pytest.raises(ValueError):
        _ = Path("does not have correct attributes")

    # With no species there should be no peak/saddle/energies
    assert len(path.rel_energies) == 0
    assert len(path.energies) == 0

    assert not path.contains_peak
    assert path.peak_idx is None
    assert not path.is_saddle(idx=0)

    # Should not plot plot a path without any structures
    path.plot_energies(save=True, name="tmp", color="black", xlabel="none")
    assert not os.path.exists("tmp.pdf")


def test_path_properties():
    p1 = PathPoint(test_species.copy(), constraints={})
    p1.energy = -3
    p2 = PathPoint(test_species.copy(), constraints={})
    p2.energy = -2

    path = Path(p1, p2, units=KcalMol)
    assert all(np.isclose(path.energies, np.array([-3, -2])))
    assert all(np.isclose(path.rel_energies, 627.509 * np.array([0, 1])))

    p3 = PathPoint(test_species.copy(), constraints={})
    path = Path(p1, p2, p3)

    # There is an energy not set, should not be able to find a peak
    assert path.peak_idx is None
    assert not path.contains_peak
    assert not path.is_saddle(idx=1)

    # setting the energy of the final point should allow a peak
    path[2].energy = -3
    assert path.contains_peak
    assert path.peak_idx == 1
    assert path.is_saddle(idx=1)

    path.plot_energies(save=True, name="tmp", color="black", xlabel="none")
    assert os.path.exists("tmp.pdf")
    os.remove("tmp.pdf")

    # Should ba able to print an xyz file containing the structures along the
    # path
    path.print_geometries(name="tmp")
    assert os.path.exists("tmp.xyz")
    os.remove("tmp.xyz")


def test_point_properties():

    point = PathPoint(species=test_species.copy(), constraints={})

    assert point.energy is None
    assert point.grad is None
    assert not point.species.constraints.any
    assert point.species.name == "tmp"

    point.energy = 1
    point.grad = np.zeros(shape=(1, 3))

    # Copies should not copy energy or gradients
    new_point = point.copy()
    assert new_point.energy is None
    assert new_point.grad is None


def test_pruning_bonds():

    h3 = Species(
        name="h3",
        charge=0,
        mult=2,
        atoms=[Atom("H"), Atom("H", x=1), Atom("H", x=0.5, y=0.5)],
    )

    fbond = FormingBond(atom_indexes=(0, 1), species=h3)
    bbond1 = BreakingBond(atom_indexes=(0, 2), species=h3)
    bbond2 = BreakingBond(atom_indexes=(1, 2), species=h3)

    new_bonds = pruned_active_bonds(
        reactant=h3, fbonds=[fbond], bbonds=[bbond1, bbond2]
    )
    assert len(new_bonds) == 2
    # Should prune to one breaking and one forming bond
    assert (
        isinstance(new_bonds[0], FormingBond)
        and isinstance(new_bonds[1], BreakingBond)
    ) or (
        isinstance(new_bonds[1], FormingBond)
        and isinstance(new_bonds[0], BreakingBond)
    )

    # Test the correct assigment of the final bond distance
    ru_reac = Species(
        name="Ru_alkene",
        charge=0,
        mult=1,
        atoms=[
            Atom("Ru", 0.45366, 0.70660, -0.25056),
            Atom("C", 0.72920, 1.42637, 1.37873),
            Atom("C", -1.75749, -0.39358, 0.57059),
            Atom("C", -1.10229, -1.02739, -0.43978),
        ],
    )

    ru_prod = Species(
        name="Ru_cycylobutane",
        charge=0,
        mult=1,
        atoms=[
            Atom("Ru", 0.28841, -1.68905, 0.39833),
            Atom("C", -0.85865, -0.07597, -0.29711),
            Atom("C", 0.10995, 0.44156, -1.35018),
            Atom("C", 1.26946, -0.42574, -0.91200),
        ],
    )

    bbond = BreakingBond(
        atom_indexes=[0, 2], species=ru_reac, final_species=ru_prod
    )

    assert np.isclose(bbond.final_dist, ru_prod.distance(0, 2))


def test_pruning_bonds2():

    h2 = Species(
        name="h2", charge=0, mult=2, atoms=[Atom("H"), Atom("H", x=1)]
    )

    h2_close = Species(
        name="h2", charge=0, mult=2, atoms=[Atom("H"), Atom("H", x=0.5)]
    )

    bbond = BreakingBond(
        atom_indexes=[0, 1], species=h2, final_species=h2_close
    )

    # A breaking bond with a final distance shorter than the current
    # (which is possible) should be pruned
    assert len(pruned_active_bonds(h2, fbonds=[], bbonds=[bbond])) == 0


def test_products_made():

    path = Path(PathPoint(species=test_mol, constraints={}))

    assert not path.products_made(product=None)
    # Species have no graphs
    assert not path.products_made(product=test_species)

    # with a single point and a molecule with the same graph then the products
    # are made, at the first point
    assert path.products_made(product=test_mol)

    diff_mol = test_mol.copy()
    diff_mol.graph.remove_edge(0, 1)
    assert not path.products_made(product=diff_mol)


def test_adaptive_path():

    species_no_atoms = Species(name="tmp", charge=0, mult=1, atoms=[])
    path1 = AdaptivePath(init_species=species_no_atoms, bonds=[], method=XTB())
    assert len(path1) == 0
    assert path1.method.name == "xtb"
    assert len(path1.bonds) == 0

    assert path1 != 0
    assert path1 == path1
