import pytest
import numpy as np
from autode import Molecule
from autode.atoms import Atom
from autode.geom import are_coords_reasonable, calc_heavy_atom_rmsd
from autode.smiles.parser import Parser, SMILESBonds
from autode.smiles.builder import Builder, Angle, Dihedral
from autode.exceptions import SMILESBuildFailed

parser = Parser()
builder = Builder()


def built_molecule_is_reasonable(smiles):
    """Is the molecule built from a SMILES string sensible?"""

    parser.parse(smiles)
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    mol.print_xyz_file(filename='tmp.xyz')

    return are_coords_reasonable(mol.coordinates)


def built_molecule_is_usually_reasonable(smiles, n_trys=3):
    """Is building this molecule mostly sensible?"""

    for _ in range(n_trys):

        if built_molecule_is_reasonable(smiles):
            return True

    return False


def test_base_builder():

    # Builder needs SMILESAtom-s
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[Atom('H')], bonds=SMILESBonds())

    # Builder needs at least some atoms
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=None, bonds=[])

    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[], bonds=[])

    parser.parse(smiles='O')
    builder.build(atoms=parser.atoms, bonds=parser.bonds)
    assert builder.non_bonded_idx_matrix.shape == (3, 3)
    assert builder.non_bonded_idx_matrix[0, 1] == 0   # O-H
    assert builder.non_bonded_idx_matrix[0, 0] == 0   # O-O
    assert builder.non_bonded_idx_matrix[1, 2] == 1   # H-H


def test_explicit_hs():

    parser.parse(smiles='C')
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

    # Should convert all implicit Hs to explicit atoms
    assert len(builder.atoms) == 5
    assert len(builder.bonds) == 4
    assert len([atom for atom in builder.atoms if atom.label == 'H']) == 4

    assert builder.graph.number_of_nodes() == 5
    assert builder.graph.number_of_edges() == 4

    parser.parse(smiles='CC(C)(C)C')
    assert len([True for (i, j) in parser.bonds
                if parser.atoms[i].label == parser.atoms[j].label == 'C']) == 4
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)
    assert builder.n_atoms == 17


def test_d8():
    d8_smiles_list = ['[PH3]=[Pd](Cl)(Cl)=[PH3]',
                      '[PH3+][Pd-2](Cl)([PH3+])Cl',
                      #TODO Add some more test cases here
                      ]

    for smiles in d8_smiles_list:

        parser.parse(smiles)
        builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

        pd_idx = next(idx for idx, atom in enumerate(builder.atoms)
                      if atom.label == 'Pd')
        assert builder._atom_is_d8(idx=pd_idx)


def test_angle():

    water = Molecule(smiles='O')
    angle = Angle(idxs=[1, 0, 2])   # H, O, H

    assert angle.phi_ideal is None
    assert np.isclose(angle.phi0, np.deg2rad(100), atol=10)
    assert np.isclose(angle.value(atoms=water.atoms), np.deg2rad(105),
                      atol=np.deg2rad(15))  # ±15 degrees, a pretty loose tol


def test_dihedrals():

    trans = [Atom('C', -0.94807, -1.38247, -0.02522),
             Atom('C',  0.54343, -1.02958, -0.02291),
             Atom('C', -1.81126, -0.12418, -0.02130),
             Atom('C',  1.40662, -2.28788, -0.02401)]
    dihedral = Dihedral(idxs=[2, 0, 1, 3])
    assert np.isclose(dihedral.value(trans),
                      np.pi, atol=0.05)

    gauche = [Atom('C', 0.33245, -2.84500, 0.36258),
              Atom('C', 1.20438, -1.58016, 0.31797),
              Atom('C', 0.85514, -3.97306, -0.52713),
              Atom('C', 2.61201, -1.79454, 0.87465)]

    assert np.isclose(dihedral.value(gauche),
                      np.deg2rad(-64.5), atol=0.01)

    zero = [Atom('C', 0.0, 0.0, 0.0),
            Atom('C', 0.0, 0.0, 0.0),
            Atom('C', 0.0, 0.0, 0.0),
            Atom('C', 2.61201, -1.79454, 0.87465)]

    # Can't have a dihedral with vectors of zero length
    with pytest.raises(ValueError):
        _ = dihedral.value(zero)


def test_cdihedral_rotation():

    try:
        from cdihedrals import rotate

    except ModuleNotFoundError:
        return

    # If the extension is found then ensure that the dihedral rotation works
    parser.parse(smiles='CC')
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)

    dihedral = Dihedral(idxs=[2, 0, 1, 5])
    coords = mol.coordinates

    rot_idxs = np.zeros(shape=(1, mol.n_atoms), dtype='i4')
    rot_idxs[0, [5, 6, 7]] = 1

    rot_coords = rotate(py_coords=np.array(coords, dtype='f8'),
                        py_angles=np.array([-dihedral.value(builder.atoms)],
                                           dtype='f8'),
                        py_axes=np.array([[1, 0]], dtype='i4'),
                        py_rot_idxs=rot_idxs,
                        py_origins=np.array([0], dtype='i4')
                        )

    mol.coordinates = rot_coords

    assert np.isclose(dihedral.value(mol.atoms), 0.0, atol=1E-5)
    assert are_coords_reasonable(mol.coordinates)


def test_simple_alkane():
    """A few simple linear and branched alkanes"""

    simple_smiles = ['C', 'CC', 'CCC', 'CCCC', 'CC(C)C']

    for smiles in simple_smiles:
        assert built_molecule_is_reasonable(smiles)


def test_long_alkane():
    """Should be able to build a long alkane without overlapping atoms"""

    assert built_molecule_is_reasonable(smiles='CCCCCCC')


def test_simple_multispecies():
    """Some simple molecules """

    assert built_molecule_is_reasonable(smiles='O')   # water
    assert built_molecule_is_reasonable(smiles='N')   # ammonia
    assert built_molecule_is_reasonable(smiles='B')   # BH3


def test_simple_multispecies2():
    """A small set of molecules with more than just carbon atoms"""

    assert built_molecule_is_reasonable(smiles='N#N')
    assert built_molecule_is_reasonable(smiles='OO')
    assert built_molecule_is_reasonable(smiles='O=[N]=O')
    assert built_molecule_is_reasonable(smiles='CN=C=O')


def test_simple_ring():
    """Small unsubstituted rings"""

    parser.parse(smiles='C1CCCCC1')                          # cyclohexane
    builder.set_atoms_bonds(parser.atoms, parser.bonds)
    ring_dihedrals = list(builder._ring_dihedrals(ring_bond=[3, 4]))
    assert len(ring_dihedrals) == 3

    assert built_molecule_is_reasonable(smiles='C1CCCC1')     # cyclopentane
    assert built_molecule_is_reasonable(smiles='C1CCCCC1')    # cyclohexane
    assert built_molecule_is_reasonable(smiles='C1CCCCCC1')   # cycloheptane
    assert built_molecule_is_reasonable(smiles='C1CCCCCCC1')  # cycloctane


def test_double_bonds():

    assert built_molecule_is_reasonable(smiles='C=C')
    assert built_molecule_is_reasonable(smiles='CC/C=C/CCC')

    # Trans
    parser.parse(smiles='C/C=C/C')
    builder.build(parser.atoms, parser.bonds)

    dihedral = Dihedral(idxs=[0, 1, 2, 3])
    value = np.abs(dihedral.value(builder.atoms))

    assert (np.isclose(value, -np.pi, atol=1E-4)
            or np.isclose(value, np.pi, atol=1E-4))

    # Cis double bond
    for cis_smiles in (r'C/C=C\C', r'C\C=C/C'):
        parser.parse(cis_smiles)
        builder.build(parser.atoms, parser.bonds)

        value = np.abs(dihedral.value(builder.atoms))

        assert np.isclose(value, 0.0, atol=1E-4)


def test_chiral_tetrahedral():
    """Check simple chiral carbons"""

    parser.parse(smiles='C[C@@H](Cl)F')
    builder.build(parser.atoms, parser.bonds)
    r_mol = Molecule(atoms=builder.atoms, name='R_chiral')
    # r_mol.print_xyz_file()

    coords = r_mol.coordinates

    v1 = coords[0] - coords[1]  # C-C
    v1 /= np.linalg.norm(v1)

    v2 = coords[3] - coords[1]  # C-F
    v2 /= np.linalg.norm(v2)

    v3 = coords[2] - coords[1]   # C-Cl
    # C-Cl vector should be pointing out of the plane, with a positive
    # component along the normal to the C-C-F plane
    assert np.dot(v3, np.cross(v1, v2)) > 0

    parser.parse(smiles='C[C@H](Cl)F')
    builder.build(parser.atoms, parser.bonds)
    s_mol = Molecule(atoms=builder.atoms, name='S_chiral')
    # s_mol.print_xyz_file()

    # Different chirality should have RMSD > 0.1 Å on heavy atoms
    assert calc_heavy_atom_rmsd(s_mol.atoms, r_mol.atoms) > 0.1


def test_macrocycle():

    # Large linear structure with stereochemistry
    lin_smiles = ('C/C=C/[C@@H](C)[C@H](O[Si](C)(C)C)[C@@H](OC)/C=C'
                  '/CC/C=C/C(OC)=O')
    assert built_molecule_is_reasonable(smiles=lin_smiles)

    # Large macrocyclic ring with stereochemistry
    macro_smiles = ('C/C1=C/[C@@H](C)[C@H](O[Si](C)(C)C)[C@@H](OC)/C=C'
                    '/CC/C=C/C(OC1)=O')
    assert built_molecule_is_usually_reasonable(smiles=macro_smiles)


def test_branches_on_rings():
    """Branches on rings should be fine"""

    assert built_molecule_is_reasonable(smiles='C1CC(CCC)C(CC)CC1')
    assert built_molecule_is_reasonable(smiles='C1NC(CNC)C(CO)CC1')
    assert built_molecule_is_reasonable(smiles='C1C(C)C(C)C(C)C(C)C1')


def test_aromatics():

    assert built_molecule_is_reasonable(smiles='C1=CC=CC=C1')  # benzene
    assert built_molecule_is_reasonable(smiles='c1ccccc1')     # benzene


def test_small_rings():
    """Small rings may need angle adjustment to be reasonable"""

    parser.parse(smiles='C1CC1')
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    assert 1.3 < mol.distance(1, 2) < 1.7    # Closing CC bond should be close

    parser.parse(smiles='C1CCC1')
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    assert 1.3 < mol.distance(2, 3) < 1.7    # Closing CC bond should be close


def test_wikipedia_examples():
    """From: wikipedia.org/wiki/Simplified_molecular-input_ine-entry_system"""

    smiles_list = ['O=Cc1ccc(O)c(OC)c1',
                   'COc1cc(C=O)ccc1O',
                   'CC(=O)NCCC1=CNc2c1cc(OC)cc2']

    for smiles in smiles_list:
        assert built_molecule_is_reasonable(smiles=smiles)


def test_metal_complexes():

    # Check some simple complexes with CN > 4                      [Co(Cl)5]2-
    assert built_molecule_is_reasonable(smiles='Cl[Co-2](Cl)(Cl)(Cl)Cl')
    #                                                              [Co(Cl)6]3-
    assert built_molecule_is_reasonable(smiles='Cl[Co-3](Cl)(Cl)(Cl)(Cl)Cl')

    # Some higher coordinations (unphysical)
    assert built_molecule_is_reasonable(smiles='F[Co](F)(F)(F)(F)(F)F')
    assert built_molecule_is_reasonable(smiles='F[Co](F)(F)(F)(F)(F)(F)F')


def test_fused_rings():
    """Test building fused rings, for which dihedral adjustment
    is not sufficent"""

    # Fused cyclobutane/cyclopentane
    parser.parse(smiles='C1CC2C(C1)CC2')
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    # mol.print_xyz_file(filename='tmp.xyz')

    assert are_coords_reasonable(mol.coordinates)
    assert all(1.3 < mol.distance(*pair) < 1.7 for pair in mol.graph.edges
               if mol.atoms[pair[0]].label == 'C'
                  and mol.atoms[pair[1]].label == 'C')


def _test_tmp():

    parser.parse(smiles='C1CCCC1')
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    mol.print_xyz_file(filename='tmp.xyz')
    # print(mol.distance(9, 15))
    #assert built_molecule_is_reasonable(smiles=r'O[C@@H]1[C@H](/C=C\CC/C=C'
    #                                           r'\C(CC/C(C)=C/[C@H]1C)=O)OC')
