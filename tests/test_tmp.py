import pytest
import numpy as np
from autode import Molecule
from autode.atoms import Atom
from autode.smiles.atom_types import TetrahedralAtom
from autode.geom import are_coords_reasonable, calc_heavy_atom_rmsd
from autode.smiles.parser import Parser, SMILESBonds, RingBond, SMILESAtom
from autode.smiles.builder import Builder, Angle, Dihedral
from autode.exceptions import SMILESBuildFailed
from autode.mol_graphs import get_mapping

parser = Parser()
builder = Builder()


def built_molecule_is_reasonable(smiles):
    """Is the molecule built from a SMILES string sensible?"""

    parser.parse(smiles)
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    # mol.print_xyz_file(filename='tmp.xyz')

    return are_coords_reasonable(mol.coordinates)


def test_many_ring_double_bonds():

    assert built_molecule_is_reasonable(smiles=r'C1=C\N=C/C=N\C=C/C/1')

