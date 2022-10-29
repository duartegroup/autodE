"""
Tests that are in the documentation. If they break the documentation is wrong!
If there is any change to the code please also change the examples to
accommodate the changes.
"""
from autode.species import Species
from autode.species import Molecule
from autode.config import Config
from autode.mol_graphs import split_mol_across_bond
from autode.atoms import Atom
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_species():

    species = Species(name="species", atoms=None, charge=0, mult=1)
    assert species.n_atoms == 0

    h2 = Species(name="H2", charge=0, mult=1, atoms=[Atom("H"), Atom("H")])
    assert h2.n_atoms == 2

    # Expecting both atoms to be initialised at the origin
    assert np.linalg.norm(h2.atoms[0].coord - h2.atoms[1].coord) < 1e-6

    atom1, atom2 = h2.atoms
    atom1.translate(vec=np.array([1.0, 0.0, 0.0]))
    atom1.rotate(theta=np.pi, axis=np.array([0.0, 0.0, 1.0]))

    assert np.linalg.norm(atom1.coord - np.array([-1.0, 0.0, 0.0])) < 1e-6

    assert h2.solvent is None

    f = Species(
        name="F-", charge=-1, mult=1, atoms=[Atom("F")], solvent_name="DCM"
    )
    assert f.solvent.g09 == "Dichloromethane"
    assert f.solvent.xtb == "CH2Cl2"


def test_molecule():

    molecule = Molecule(name="molecule")
    assert molecule.charge == 0
    assert molecule.mult == 1

    water = Molecule(name="h2o", smiles="O")
    assert water.n_atoms == 3
    assert all(node in water.graph.nodes for node in (0, 1, 2))
    assert (0, 1) in water.graph.edges
    assert (0, 2) in water.graph.edges

    # Shift so the first atom is at the origin
    water.translate(vec=-water.atoms[0].coord)
    assert np.linalg.norm(water.atoms[0].coord - np.zeros(3)) < 1e-6


def test_manipulation():
    methane = Molecule(name="CH4", smiles="C")
    assert methane.n_atoms == 5
    ch3_nodes, h_nodes = split_mol_across_bond(methane.graph, bond=(0, 1))

    ch3 = Molecule(
        name="CH3", mult=2, atoms=[methane.atoms[i] for i in ch3_nodes]
    )
    assert ch3.n_atoms == 4

    h = Molecule(name="H", mult=2, atoms=[methane.atoms[i] for i in h_nodes])
    assert h.n_atoms == 1


def test_conformers(tmpdir):
    os.chdir(tmpdir)

    butane = Molecule(name="butane", smiles="CCCC")
    butane.populate_conformers(n_confs=10)

    n_confs = len(butane.conformers)
    assert n_confs > 1

    # Lowing the RMSD threshold should afford more conformers
    Config.rmsd_threshold = 0.01
    butane.populate_conformers(n_confs=10)
    assert len(butane.conformers) > n_confs

    # Conformer generation should also work if the RDKit method fails
    butane.rdkit_conf_gen_is_fine = False
    butane.populate_conformers(n_confs=10)
    assert len(butane.conformers) > 1

    # Change RMSD threshold back
    Config.rmsd_threshold = 0.3

    os.chdir(here)
