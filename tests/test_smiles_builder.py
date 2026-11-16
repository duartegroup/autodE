import pytest
import numpy as np
from autode import Molecule
from autode.atoms import Atom
from autode.smiles.smiles import init_smiles
from autode.smiles.atom_types import TetrahedralAtom
from autode.geom import are_coords_reasonable, calc_heavy_atom_rmsd
from autode.smiles.parser import Parser, SMILESBonds, RingBond, SMILESAtom
from autode.smiles.builder import Builder, SAngle, SDihedral
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


def built_molecule_is_usually_reasonable(smiles, n_trys=3):
    """Is building this molecule mostly sensible?"""

    for _ in range(n_trys):
        if built_molecule_is_reasonable(smiles):
            return True

    return False


def test_base_builder():
    # Builder needs SMILESAtom-s
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[Atom("H")], bonds=SMILESBonds())

    # Builder needs at least some atoms
    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=None, bonds=[])

    with pytest.raises(SMILESBuildFailed):
        builder.build(atoms=[], bonds=[])

    parser.parse(smiles="O")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)
    assert builder.non_bonded_idx_matrix.shape == (3, 3)
    assert builder.non_bonded_idx_matrix[0, 1] == 0  # O-H
    assert builder.non_bonded_idx_matrix[0, 0] == 0  # O-O
    assert builder.non_bonded_idx_matrix[1, 2] == 1  # H-H

    for atom in builder.canonical_atoms:
        assert isinstance(atom, Atom)
        assert hasattr(atom, "coord")

    for atom in builder.canonical_atoms_at_origin:
        assert np.isclose(np.linalg.norm(atom.coord), 0.0, atol=1e-4)


def test_build_single_atom():
    """No building needed for a single atom, but the builder should be fine"""

    parser.parse(smiles="[H]")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)

    assert builder.n_atoms == 1
    assert parser.mult == 2


def test_ring_path():
    parser.parse(smiles="C1C1")
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

    # Define a 'ring' bond in this system of just two atoms in a 'ring'
    ring_bond = RingBond(idx_i=0, symbol="-")
    ring_bond.close(idx=1, symbol="-")

    # There then is no valid path that traverses the ring
    with pytest.raises(SMILESBuildFailed):
        _ = builder._ring_path(ring_bond)


def test_too_high_valance():
    parser.parse(smiles="CC(C)(C)(C)(C)(C)(C)(C)(C)C")

    with pytest.raises(Exception):
        builder.build(atoms=parser.atoms, bonds=parser.bonds)


def test_explicit_hs():
    parser.parse(smiles="C")
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

    # Should convert all implicit Hs to explicit atoms
    assert len(builder.atoms) == 5
    assert len(builder.bonds) == 4
    assert len([atom for atom in builder.atoms if atom.label == "H"]) == 4

    assert builder.graph.number_of_nodes() == 5
    assert builder.graph.number_of_edges() == 4

    parser.parse(smiles="CC(C)(C)C")
    assert (
        len(
            [
                True
                for (i, j) in parser.bonds
                if parser.atoms[i].label == parser.atoms[j].label == "C"
            ]
        )
        == 4
    )
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)
    assert builder.n_atoms == 17


def test_d8():
    d8_smiles_list = [
        "[PH3]=[Pd](Cl)(Cl)=[PH3]",
        "[PH3+][Pd-2](Cl)([PH3+])Cl",
        # TODO Add some more test cases here
    ]

    for smiles in d8_smiles_list:
        parser.parse(smiles)
        builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)

        pd_idx = next(
            idx for idx, atom in enumerate(builder.atoms) if atom.label == "Pd"
        )
        assert builder._atom_is_d8(idx=pd_idx)


def test_angle():
    water = Molecule(smiles="O")
    angle = SAngle(idxs=[1, 0, 2])  # H, O, H

    assert angle.phi_ideal is None
    assert np.isclose(angle.phi0, np.deg2rad(100), atol=10)
    assert np.isclose(
        angle.value(atoms=water.atoms), np.deg2rad(105), atol=np.deg2rad(15)
    )  # ±15 degrees, a pretty loose tol


def test_dihedrals():
    trans = [
        Atom("C", -0.94807, -1.38247, -0.02522),
        Atom("C", 0.54343, -1.02958, -0.02291),
        Atom("C", -1.81126, -0.12418, -0.02130),
        Atom("C", 1.40662, -2.28788, -0.02401),
    ]
    dihedral = SDihedral(idxs=[2, 0, 1, 3])
    assert np.isclose(dihedral.value(trans), np.pi, atol=0.05)

    gauche = [
        Atom("C", 0.33245, -2.84500, 0.36258),
        Atom("C", 1.20438, -1.58016, 0.31797),
        Atom("C", 0.85514, -3.97306, -0.52713),
        Atom("C", 2.61201, -1.79454, 0.87465),
    ]

    assert np.isclose(dihedral.value(gauche), np.deg2rad(-64.5), atol=0.01)

    zero = [
        Atom("C", 0.0, 0.0, 0.0),
        Atom("C", 0.0, 0.0, 0.0),
        Atom("C", 0.0, 0.0, 0.0),
        Atom("C", 2.61201, -1.79454, 0.87465),
    ]

    # Can't have a dihedral with vectors of zero length
    with pytest.raises(Exception):
        _ = dihedral.value(zero)


def test_cdihedral_rotation():
    try:
        # from ade_dihedrals import rotate
        from autode.ext.ade_dihedrals import rotate

    except ModuleNotFoundError:
        return

    # If the extension is found then ensure that the dihedral rotation works
    parser.parse(smiles="CC")
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)

    dihedral = SDihedral(idxs=[2, 0, 1, 5])
    coords = mol.coordinates

    rot_idxs = np.zeros(shape=(1, mol.n_atoms), dtype="i4")
    rot_idxs[0, [5, 6, 7]] = 1

    rot_coords = rotate(
        py_coords=np.array(coords, dtype="f8"),
        py_angles=np.array([-dihedral.value(builder.atoms)], dtype="f8"),
        py_axes=np.array([[1, 0]], dtype="i4"),
        py_rot_idxs=rot_idxs,
        py_origins=np.array([0], dtype="i4"),
    )

    mol.coordinates = rot_coords

    assert np.isclose(dihedral.value(mol.atoms), 0.0, atol=1e-5)
    assert are_coords_reasonable(mol.coordinates)

    # Minimising on this dihedral should rotate it from 0º, however as
    # 0.0 is a saddle point there needs to be a slight displacement, and
    # should minimise to the next minimum at a 60º dihedral
    rot_coords = rotate(
        py_coords=np.array(mol.coordinates, dtype="f8"),
        py_angles=np.array([0.1], dtype="f8"),
        py_axes=np.array([[1, 0]], dtype="i4"),
        py_rot_idxs=rot_idxs,
        py_origins=np.array([0], dtype="i4"),
        minimise=True,
    )

    mol.coordinates = rot_coords

    assert np.isclose(dihedral.value(mol.atoms), np.deg2rad(60), atol=0.1)
    assert are_coords_reasonable(mol.coordinates)


def test_simple_alkane():
    """A few simple linear and branched alkanes"""

    simple_smiles = ["C", "CC", "CCC", "CCCC", "CC(C)C"]

    for smiles in simple_smiles:
        assert built_molecule_is_reasonable(smiles)


def test_long_alkane():
    """Should be able to build a long alkane without overlapping atoms"""

    assert built_molecule_is_reasonable(smiles="CCCCCCC")


def test_simple_multispecies():
    """Some simple molecules"""

    assert built_molecule_is_reasonable(smiles="O")  # water
    assert built_molecule_is_reasonable(smiles="N")  # ammonia
    assert built_molecule_is_reasonable(smiles="B")  # BH3


def test_simple_multispecies2():
    """A small set of molecules with more than just carbon atoms"""

    assert built_molecule_is_reasonable(smiles="N#N")
    assert built_molecule_is_reasonable(smiles="OO")
    assert built_molecule_is_reasonable(smiles="O=[N]=O")
    assert built_molecule_is_reasonable(smiles="CN=C=O")


def test_simple_ring():
    """Small unsubstituted rings"""

    parser.parse(smiles="C1CCCCC1")  # cyclohexane
    builder.set_atoms_bonds(parser.atoms, parser.bonds)
    ring_dihedrals = list(builder._ring_dihedrals(ring_bond=[3, 4]))
    assert len(ring_dihedrals) == 3

    assert builder.max_ring_n == 6

    assert built_molecule_is_reasonable(smiles="C1CCCC1")  # cyclopentane
    assert built_molecule_is_reasonable(smiles="C1CCCCC1")  # cyclohexane
    assert built_molecule_is_reasonable(smiles="C1CCCCCC1")  # cycloheptane
    assert built_molecule_is_usually_reasonable(
        smiles="C1CCCCCCC1"
    )  # cycloctane


def test_double_bonds():
    assert built_molecule_is_reasonable(smiles="C=C")
    assert built_molecule_is_reasonable(smiles="CC/C=C/CCC")

    # Trans
    parser.parse(smiles="C/C=C/C")
    builder.build(parser.atoms, parser.bonds)

    dihedral = SDihedral(idxs=[0, 1, 2, 3])
    value = np.abs(dihedral.value(builder.atoms))

    assert np.isclose(value, -np.pi, atol=1e-4) or np.isclose(
        value, np.pi, atol=1e-4
    )

    # Cis double bond
    for cis_smiles in (r"C/C=C\C", r"C\C=C/C"):
        parser.parse(cis_smiles)
        builder.build(parser.atoms, parser.bonds)

        value = np.abs(dihedral.value(builder.atoms))

        assert np.isclose(value, 0.0, atol=1e-4)


def test_chiral_tetrahedral():
    """Check simple chiral carbons"""

    parser.parse(smiles="C[C@@H](Cl)F")
    builder.build(parser.atoms, parser.bonds)
    r_mol = Molecule(atoms=builder.atoms, name="R_chiral")
    # r_mol.print_xyz_file()

    coords = r_mol.coordinates

    v1 = coords[0] - coords[1]  # C-C
    v1 /= np.linalg.norm(v1)

    v2 = coords[3] - coords[1]  # C-F
    v2 /= np.linalg.norm(v2)

    v3 = coords[2] - coords[1]  # C-Cl
    # C-Cl vector should be pointing out of the plane, with a positive
    # component along the normal to the C-C-F plane
    assert np.dot(v3, np.cross(v1, v2)) > 0

    parser.parse(smiles="C[C@H](Cl)F")
    builder.build(parser.atoms, parser.bonds)
    s_mol = Molecule(atoms=builder.atoms, name="S_chiral")
    # s_mol.print_xyz_file()

    # Different chirality should have RMSD > 0.1 Å on heavy atoms
    assert calc_heavy_atom_rmsd(s_mol.atoms, r_mol.atoms) > 0.1


def test_chiral_tetrahedral2():
    s_smiles = "F[C@]([H])(C)Cl"
    parser.parse(s_smiles)
    builder.build(parser.atoms, parser.bonds)
    s_mol = Molecule(atoms=builder.atoms)

    parser.parse(smiles="F[C@H](C)Cl")
    builder.build(parser.atoms, parser.bonds)
    s_mol = Molecule(atoms=builder.atoms)

    # Two representations should be the same
    assert calc_heavy_atom_rmsd(s_mol.atoms, builder.atoms) < 0.1


def test_chiral_tetrahedral3():
    """Equivalent SMILES strings from http://opensmiles.org/opensmiles.html"""

    parser.parse(smiles="N[C@](Br)(O)C")
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)

    equiv_smiles = [
        "N[C@](Br)(O)C",
        "Br[C@](O)(N)C",
        "O[C@](Br)(C)N",
        "Br[C@](C)(O)N",
        "C[C@](Br)(N)O",
        "Br[C@](N)(C)O",
        "C[C@@](Br)(O)N",
        "Br[C@@](N)(O)C",
        "[C@@](C)(Br)(O)N",
        "[C@@](Br)(N)(O)C",
    ]

    for smiles in equiv_smiles:
        parser.parse(smiles)
        builder.build(parser.atoms, parser.bonds)
        equiv_mol = Molecule(atoms=builder.atoms)

        # Atoms may not be in the same order, so map them
        mapping = get_mapping(equiv_mol.graph, mol.graph)
        atoms = [equiv_mol.atoms[i] for i in sorted(mapping, key=mapping.get)]

        assert calc_heavy_atom_rmsd(mol.atoms, atoms) < 0.2


def test_sq_planar_xe():
    parser.parse(smiles="F[Xe](F)(F)F")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)
    coords = builder.coordinates

    assert builder.atoms[1].label == "Xe"

    normal = np.cross((coords[0] - coords[1]), (coords[2] - coords[1]))

    for f_atom_idx in range(2, 4):
        dot_product = np.dot((coords[f_atom_idx] - coords[1]), normal)

        # Dot product should be close to zero as the Xe-F vectors should be
        # roughly orthogonal to the normal out of the plane, if the geometry
        # is square planar
        assert np.isclose(dot_product, 0.0, atol=0.1)


def test_macrocycle():
    # Large linear structure with stereochemistry
    lin_smiles = (
        "C/C=C/[C@@H](C)[C@H](O[Si](C)(C)C)[C@@H](OC)/C=C" "/CC/C=C/C(OC)=O"
    )
    assert built_molecule_is_reasonable(smiles=lin_smiles)

    # Large macrocyclic ring with stereochemistry
    macro_smiles = (
        "C/C1=C/[C@@H](C)[C@H](O[Si](C)(C)C)[C@@H](OC)/C=C" "/CC/C=C/C(OC1)=O"
    )
    assert built_molecule_is_usually_reasonable(smiles=macro_smiles)


def test_branches_on_rings():
    """Branches on rings should be fine"""

    assert built_molecule_is_reasonable(smiles="C1CC(CCC)C(CC)CC1")
    assert built_molecule_is_reasonable(smiles="C1NC(CNC)C(CO)CC1")
    assert built_molecule_is_reasonable(smiles="C1C(C)C(C)C(C)C(C)C1")


def test_aromatics():
    assert built_molecule_is_reasonable(smiles="C1=CC=CC=C1")  # benzene
    assert built_molecule_is_reasonable(smiles="c1ccccc1")  # benzene


def test_small_rings():
    """Small rings may need angle adjustment to be reasonable"""

    parser.parse(smiles="C1CC1")
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    assert 1.3 < mol.distance(1, 2) < 1.7  # Closing CC bond should be close

    parser.parse(smiles="C1CCC1")
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    assert 1.3 < mol.distance(2, 3) < 1.7  # Closing CC bond should be close


def test_wikipedia_examples():
    """From: wikipedia.org/wiki/Simplified_molecular-input_ine-entry_system"""

    smiles_list = [
        "O=Cc1ccc(O)c(OC)c1",
        "COc1cc(C=O)ccc1O",
        "CC(=O)NCCC1=CNc2c1cc(OC)cc2",
    ]

    for smiles in smiles_list:
        assert built_molecule_is_reasonable(smiles=smiles)


def test_metal_complexes():
    # Check some simple complexes with CN > 4                      [Co(Cl)5]2-
    assert built_molecule_is_reasonable(smiles="Cl[Co-2](Cl)(Cl)(Cl)Cl")
    #                                                              [Co(Cl)6]3-
    assert built_molecule_is_reasonable(smiles="Cl[Co-3](Cl)(Cl)(Cl)(Cl)Cl")

    # Some higher coordinations (unphysical)
    assert built_molecule_is_reasonable(smiles="F[Co](F)(F)(F)(F)(F)F")
    assert built_molecule_is_reasonable(smiles="F[Co](F)(F)(F)(F)(F)(F)F")


def test_fused_rings():
    """Test building fused rings, for which dihedral adjustment
    is not sufficent"""

    # Fused cyclobutane/cyclopentane
    parser.parse(smiles="C1CC2C(C1)CC2")
    builder.build(parser.atoms, parser.bonds)
    mol = Molecule(atoms=builder.atoms)
    # mol.print_xyz_file(filename='tmp.xyz')

    assert are_coords_reasonable(mol.coordinates)
    assert all(
        1.3 < mol.distance(*pair) < 1.7
        for pair in mol.graph.edges
        if mol.atoms[pair[0]].label == "C" and mol.atoms[pair[1]].label == "C"
    )


def test_trans_small_rings():
    """Rings with trans double bonds need to be possible"""

    parser.parse(smiles="C1CCC/C=C/CC1")
    builder.build(parser.atoms, parser.bonds)
    assert are_coords_reasonable(builder.coordinates)

    dihedral = SDihedral(idxs=[3, 4, 5, 6])
    # Should be anything with no defined stereochem
    parser.parse(smiles="C1CCCC=CCC1")
    builder.build(parser.atoms, parser.bonds)
    assert -np.pi < dihedral.value(builder.atoms) < np.pi

    # or defined as cis
    parser.parse(smiles=r"C1CCC/C=C\CC1")
    builder.build(parser.atoms, parser.bonds)
    assert -np.pi / 2.0 < dihedral.value(builder.atoms) < np.pi / 2.0

    # but should be close to π if defined as trans
    parser.parse(smiles="C1CCC/C=C/CC1")
    builder.build(parser.atoms, parser.bonds)
    assert np.isclose(
        np.abs(dihedral.value(builder.atoms)), np.pi, atol=np.pi / 2.0
    )


def test_dihedral_force():
    parser.parse(smiles="CCCC")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)

    dihedral = SDihedral(idxs=[0, 1, 2, 3], phi0=np.pi / 2.0)

    # Can only force to 0º or 180º, distances are hard coded
    with pytest.raises(ValueError):
        builder._force_double_bond_stereochem(dihedral=dihedral)


def test_close_flat_ring():
    unclosed_coords = np.array(
        [
            [2.227521, -0.038228, -2.175656],
            [-1.704563, 1.083528, 1.226912],
            [-1.030701, 0.949700, 0.029400],
            [0.303700, 1.342600, -0.014200],
            [1.298500, 0.393400, -0.042500],
            [1.462636, -0.467365, -1.114587],
            [2.347920, -0.711203, -2.998888],
            [-2.742329, 0.782273, 1.276031],
            [-1.527315, 0.551663, -0.845852],
            [0.505600, 2.411600, -0.024200],
            [1.971412, 0.321051, 0.801897],
            [0.984661, -1.435514, -1.071969],
        ]
    )

    # populate everything by parsing normally
    parser.parse("c1ccccc1")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)

    # Set the coordinates as something that is nor already closed
    builder.coordinates = unclosed_coords

    bond = RingBond(0, symbol="-")  # 0-1 atom indexes define the closing bond
    bond.close(idx=1, symbol="-")
    bond.r0 = np.linalg.norm(unclosed_coords[3] - unclosed_coords[2])

    # re-apply the closure
    builder._close_ring(ring_bond=bond)
    mol = Molecule(atoms=builder.atoms)

    for atom_i in range(0, 5):
        # All C-C distances should be ~1.5 Å in a benzene ring
        assert np.isclose(mol.distance(atom_i, atom_i + 1), 1.5, atol=0.2)


def test_close_non_flat_ring():
    parser.parse("C1CCCCCC1")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)
    mol = Molecule(atoms=builder.atoms)

    assert are_coords_reasonable(mol.coordinates)

    for atom_i in range(0, 5):
        # All C-C distances should be ~1.5 Å in a cyclohexane ring
        assert np.isclose(mol.distance(atom_i, atom_i + 1), 1.5, atol=0.2)


def test_double_bond_stereo_branch():
    """Check that a double bond stereochemisty is well defined over
    explicit hydrogens"""

    parser.parse(smiles=r"C/C([H])=C([H])/C")
    builder.build(atoms=parser.atoms, bonds=parser.bonds)

    # Carbon dihedral should be ~π for this trans double bond, as hydrogens
    # don't count
    dihedral = SDihedral(
        idxs=[i for i, atom in enumerate(builder.atoms) if atom.label == "C"]
    )
    assert np.isclose(np.abs(dihedral.value(builder.atoms)), np.pi, atol=0.1)


def test_fused_ring_system():
    """Multiply fused rings should be buildable - repulsion needed"""

    assert built_molecule_is_reasonable(
        smiles="[SiH3]C12[C@@]3(CCC4)C4=C" "[C@@H](C1C=CC2)C3"
    )


def test_build_exceptions():
    builder.set_atoms_bonds(atoms=[SMILESAtom("H")], bonds=SMILESBonds())

    # Cannot find a ring with no atoms
    with pytest.raises(SMILESBuildFailed):
        builder._ring_idxs(inc_idxs=(0, 1))

    # Cannot find a ring path with no rings
    bond = RingBond(1, symbol="-")
    bond.close(1, symbol="-")
    with pytest.raises(SMILESBuildFailed):
        builder._ring_path(ring_bond=bond)


def test_ff_dist_matrix():
    parser.parse(smiles="C=O")
    builder_ = Builder()
    builder_.build(atoms=parser.atoms, bonds=parser.bonds)

    # Should not try and add any distance constraints across the double
    # bond with no neighbours on oxygen
    dist_matrix = builder_._ff_distance_matrix()
    assert dist_matrix.shape == (builder_.n_atoms, builder_.n_atoms)


def test_difficult_reset_onto():
    points = np.array(
        [
            [5.32654, 0.20363, -1.54392],
            [6.25674, 0.01882, 0.69857],
            [7.37946, 1.18199, -1.08374],
        ]
    )

    coord = np.array([6.54572, 0.03685, -0.65653])

    atom = TetrahedralAtom()
    atom.reset_onto(points=points, coord=coord)

    mol = Molecule(
        atoms=[Atom("C"), Atom("O"), Atom("C"), Atom("H"), Atom("C")]
    )
    mol.coordinates = np.array(
        points.tolist() + [atom.empty_site() + coord] + [coord.tolist()]
    )

    assert mol.angle(1, 4, 3).to("deg") > 90


def test_max_ring_size():
    parser.parse(
        smiles="O=C1C2=C(O[Si](C)(C)C)C[C@H](CCCC3)C3=C4C2CC[C@@H]4O1"
    )
    builder.set_atoms_bonds(atoms=parser.atoms, bonds=parser.bonds)
    assert builder.max_ring_n == 7


def test_cis_dihedral_force():
    """Test the forcing of a cis-dihedral from a trans geometry"""

    parser.parse(smiles="CC=CC")
    builder.build(parser.atoms, parser.bonds)
    # pre-generated trans geometry
    coords = [
        [-0.86310, -0.72859, 0.62457],
        [0.10928, -0.05429, 1.42368],
        [1.17035, -0.79134, 2.03167],
        [2.14109, -0.11396, 2.83018],
        [-0.46878, -1.52095, 0.23716],
        [-1.16448, -0.14182, -0.08133],
        [-1.61598, -0.98052, 1.17515],
        [0.04809, 0.90129, 1.55228],
        [1.23123, -1.74691, 1.90286],
        [2.16958, -0.51498, 3.70870],
        [3.00946, -0.19036, 2.41364],
        [1.90166, 0.81880, 2.90791],
    ]

    for atom, new_coord in zip(builder.atoms, coords):
        atom.coord = new_coord

    builder._force_double_bond_stereochem(
        dihedral=SDihedral([0, 1, 2, 3], phi0=0.0)
    )

    # Distance between the end carbons needs to be smaller than the trans
    assert 2.0 < builder.distance(0, 3) < 3.5

    mol = Molecule()
    init_smiles(mol, smiles=r"C/C=C\C")
    assert -20 < mol.dihedral(0, 1, 2, 3).to("deg") < 20


def test_many_ring_double_bonds():
    assert built_molecule_is_reasonable(smiles=r"C1=C\N=C/C=N\C=C/C/1")
    assert built_molecule_is_reasonable(smiles=r"C1=CC=C/N=N\C=C1")
