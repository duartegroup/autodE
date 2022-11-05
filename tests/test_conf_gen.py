from autode.atoms import Atom
from autode.conformers import conf_gen
from autode.species.molecule import Molecule
from autode.species.molecule import Reactant, Product
from autode.species.complex import ReactantComplex, ProductComplex
from autode.config import Config
from autode.geom import calc_rmsd
from autode.geom import are_coords_reasonable
from autode.transition_states.ts_guess import TSguess
from autode.transition_states.transition_state import TransitionState
from autode.bond_rearrangement import BondRearrangement
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))
Config.n_cores = 1

butane = Molecule(
    name="butane",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", -0.63938, -0.83117, 0.06651),
        Atom("C", 0.89658, -0.77770, 0.06222),
        Atom("H", -0.95115, -1.71970, 0.65729),
        Atom("H", -1.01425, -0.95802, -0.97234),
        Atom("C", -1.28709, 0.40256, 0.69550),
        Atom("H", 1.27330, -1.74033, -0.34660),
        Atom("H", 1.27226, -0.67376, 1.10332),
        Atom("C", 1.46136, 0.35209, -0.79910),
        Atom("H", 1.10865, 0.25011, -1.84737),
        Atom("H", 2.57055, 0.30082, -0.79159),
        Atom("H", 1.16428, 1.34504, -0.40486),
        Atom("H", -0.93531, 0.53113, 1.74115),
        Atom("H", -2.38997, 0.27394, 0.70568),
        Atom("H", -1.05698, 1.31807, 0.11366),
    ],
)

methane = Molecule(
    name="methane",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", 0.70879, 0.95819, -0.92654),
        Atom("H", 1.81819, 0.95820, -0.92655),
        Atom("H", 0.33899, 0.14642, -0.26697),
        Atom("H", 0.33899, 0.79287, -1.95935),
        Atom("H", 0.33899, 1.93529, -0.55331),
    ],
)


def test_bcp_confs(tmpdir):
    os.chdir(tmpdir)

    mol = Molecule(smiles="C1CC2C1C2")
    mol.rdkit_conf_gen_is_fine = False
    mol.populate_conformers(n_confs=100)

    assert all(conf.energy is not None for conf in mol.conformers)
    energies = np.array([conf.energy for conf in mol.conformers])

    # Hard coded standard deviation as, once the pruning has happened then
    # the standard deviation is different
    avg, std = np.average(energies), 0.005

    # This fused ring system has a reasonable probability of generating a
    # high energy conformer with RR, with a minimum that is very congested
    # it should be removed when the conformers are set
    assert all(np.abs(conf.energy - avg) / std < 5 for conf in mol.conformers)
    assert mol.n_conformers > 0

    os.chdir(here)


def test_setero_metal(tmpdir):
    os.chdir(tmpdir)

    # (R)-sec butyl lithium
    mol = Molecule(smiles="[Li][C@H](C)CC")
    mol.print_xyz_file()
    assert are_coords_reasonable(coords=mol.coordinates)

    os.chdir(here)


def test_conf_gen(tmpdir):
    os.chdir(tmpdir)

    atoms = conf_gen.get_simanl_atoms(species=methane)
    assert len(atoms) == 5
    assert os.path.exists("methane_conf0_siman.xyz")

    # Rerunning the conformer generation should read the generated .xyz file
    atoms = conf_gen.get_simanl_atoms(species=methane)
    assert len(atoms) == 5

    os.remove("methane_conf0_siman.xyz")

    # Ensure the new graph is identical
    regen = Molecule(name="regenerated_methane", atoms=atoms)

    assert regen.graph.edges == methane.graph.edges
    assert regen.graph.nodes == methane.graph.nodes

    # Should be able to generate a conformer directly using the method
    conf = conf_gen.get_simanl_conformer(species=methane)
    assert len(atoms) == 5
    assert are_coords_reasonable(conf.coordinates)
    assert conf.energy is not None
    assert conf.solvent is None

    os.remove("methane_conf0_siman.xyz")

    os.chdir(here)


def test_conf_gen_dist_const(tmpdir):
    os.chdir(tmpdir)

    hydrogen = Molecule(
        name="H2",
        charge=0,
        mult=1,
        atoms=[Atom(atomic_symbol="H"), Atom(atomic_symbol="H", z=0.7)],
    )

    # H2 at a bond length (r) of 0.7 Å has a bond
    assert len(hydrogen.graph.edges) == 1

    # H2 at r = 2 Å is definitely not bonded
    atoms = conf_gen.get_simanl_atoms(
        species=hydrogen, dist_consts={(0, 1): 2}
    )

    long_hydrogen = Molecule(name="H2", atoms=atoms, charge=0, mult=1)
    assert long_hydrogen.n_atoms == 2
    assert len(long_hydrogen.graph.edges) == 0

    os.chdir(here)


def test_chiral_rotation(tmpdir):
    os.chdir(tmpdir)

    chiral_ethane = Molecule(
        name="chiral_ethane",
        charge=0,
        mult=1,
        atoms=[
            Atom("C", -0.26307, 0.59858, -0.07141),
            Atom("C", 1.26597, 0.60740, -0.09729),
            Atom("Cl", -0.91282, 2.25811, 0.01409),
            Atom("F", -0.72365, -0.12709, 1.01313),
            Atom("H", -0.64392, 0.13084, -1.00380),
            Atom("Cl", 1.93888, 1.31880, 1.39553),
            Atom("H", 1.61975, 1.19877, -0.96823),
            Atom("Br", 1.94229, -1.20011, -0.28203),
        ],
    )

    chiral_ethane.graph.nodes[0]["stereo"] = True
    chiral_ethane.graph.nodes[1]["stereo"] = True

    atoms = conf_gen.get_simanl_atoms(chiral_ethane)
    regen = Molecule(name="regenerated_ethane", charge=0, mult=1, atoms=atoms)

    regen_coords = regen.coordinates
    coords = chiral_ethane.coordinates

    # Atom indexes of the C(C)(Cl)(F)(H) chiral centre
    ccclfh = [0, 1, 2, 3, 4]

    # Atom indexes of the C(C)(Cl)(Br)(H) chiral centre
    ccclbrh = [1, 0, 5, 7, 6]

    for centre_idxs in [ccclfh, ccclbrh]:
        # Ensure the fragmented centres map almost identically
        # if calc_rmsd(template_coords=coords[centre_idxs], coords_to_
        # fit=regen_coords[centre_idxs]) > 0.5:
        #     chiral_ethane.print_xyz_file(filename=os.path.join(here,
        # 'chiral_ethane.xyz'))
        #     regen.print_xyz_file(filename=os.path.join(here, 'regen.xyz'))

        # RMSD on the 5 atoms should be < 0.5 Å
        assert (
            calc_rmsd(
                coords1=coords[centre_idxs], coords2=regen_coords[centre_idxs]
            )
            < 0.5
        )

    os.chdir(here)


def test_butene(tmpdir):
    os.chdir(tmpdir)

    butene = Molecule(
        name="z-but-2-ene",
        charge=0,
        mult=1,
        atoms=[
            Atom("C", -1.69185, -0.28379, -0.01192),
            Atom("C", -0.35502, -0.40751, 0.01672),
            Atom("C", -2.39437, 1.04266, -0.03290),
            Atom("H", -2.13824, 1.62497, 0.87700),
            Atom("H", -3.49272, 0.88343, -0.05542),
            Atom("H", -2.09982, 1.61679, -0.93634),
            Atom("C", 0.57915, 0.76747, 0.03048),
            Atom("H", 0.43383, 1.38170, -0.88288),
            Atom("H", 1.62959, 0.40938, 0.05452),
            Atom("H", 0.39550, 1.39110, 0.93046),
            Atom("H", -2.29700, -1.18572, -0.02030),
            Atom("H", 0.07422, -1.40516, 0.03058),
        ],
    )

    butene.graph.nodes[0]["stereo"] = True
    butene.graph.nodes[1]["stereo"] = True

    # Conformer generation should retain the stereochemistry
    atoms = conf_gen.get_simanl_atoms(species=butene)
    regen = Molecule(name="regenerated_butene", atoms=atoms, charge=0, mult=1)

    regen.print_xyz_file()
    regen_coords = regen.coordinates

    # The Z-butene isomer has a r(C_1 C_2) < 3.2 Å where C_1C=CC_2
    assert np.linalg.norm(regen_coords[6] - regen_coords[2]) < 3.6

    os.chdir(here)


def test_ts_conformer(tmpdir):
    os.chdir(tmpdir)

    ch3cl = Reactant(
        charge=0,
        mult=1,
        atoms=[
            Atom("Cl", 1.63664, 0.02010, -0.05829),
            Atom("C", -0.14524, -0.00136, 0.00498),
            Atom("H", -0.52169, -0.54637, -0.86809),
            Atom("H", -0.45804, -0.50420, 0.92747),
            Atom("H", -0.51166, 1.03181, -0.00597),
        ],
    )
    f = Reactant(charge=-1, mult=1, atoms=[Atom("F", 4.0, 0.0, 0.0)])

    ch3f = Product(
        charge=0,
        mult=1,
        atoms=[
            Atom("C", -0.05250, 0.00047, -0.00636),
            Atom("F", 1.31229, -0.01702, 0.16350),
            Atom("H", -0.54993, -0.04452, 0.97526),
            Atom("H", -0.34815, 0.92748, -0.52199),
            Atom("H", -0.36172, -0.86651, -0.61030),
        ],
    )
    cl = Reactant(charge=-1, mult=1, atoms=[Atom("Cl", 4.0, 0.0, 0.0)])

    f_ch3cl_tsguess = TSguess(
        reactant=ReactantComplex(f, ch3cl),
        product=ProductComplex(ch3f, cl),
        atoms=[
            Atom("F", -2.66092, -0.01426, 0.09700),
            Atom("Cl", 1.46795, 0.05788, -0.06166),
            Atom("C", -0.66317, -0.01826, 0.02488),
            Atom("H", -0.78315, -0.58679, -0.88975),
            Atom("H", -0.70611, -0.54149, 0.97313),
            Atom("H", -0.80305, 1.05409, 0.00503),
        ],
    )

    f_ch3cl_tsguess.bond_rearrangement = BondRearrangement(
        breaking_bonds=[(2, 1)], forming_bonds=[(0, 2)]
    )

    f_ch3cl_ts = TransitionState(ts_guess=f_ch3cl_tsguess)

    atoms = conf_gen.get_simanl_atoms(
        species=f_ch3cl_ts, dist_consts=f_ch3cl_ts.active_bond_constraints
    )

    regen = Molecule(name="regenerated_ts", charge=-1, mult=1, atoms=atoms)

    # Ensure the making/breaking bonds retain their length
    regen_coords = regen.coordinates
    assert are_coords_reasonable(regen_coords) is True

    assert 1.9 < np.linalg.norm(regen_coords[0] - regen_coords[2]) < 2.1
    assert 2.0 < np.linalg.norm(regen_coords[1] - regen_coords[2]) < 2.2

    os.chdir(here)


def test_metal_eta_complex(tmpdir):
    os.chdir(tmpdir)

    # eta-6 benzene Fe2+ complex used in the molassembler paper
    m = Molecule(
        smiles="[C@@H]12[C@H]3[C@H]4[C@H]5[C@H]6[C@@H]1[Fe]265437N"
        "(C8=CC=CC=C8)C=CC=[N+]7C9=CC=CC=C9"
    )
    m.print_xyz_file()
    assert are_coords_reasonable(coords=m.coordinates)

    os.chdir(here)


def test_salt():

    salt = Molecule(name="salt", smiles="[Li][Br]")
    assert salt.n_atoms == 2
    assert are_coords_reasonable(coords=salt.coordinates)


def test_potential():

    # Approximate H2 coordinates
    bond_length = 0.7
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])

    eq_bond_length = 0.75
    d0 = np.array([[0.0, eq_bond_length], [eq_bond_length, 0.0]])

    v = conf_gen._get_v(
        coords, bonds=[(0, 1)], k=0.7, c=0.3, d0=d0, fixed_bonds=[], exponent=8
    )

    expected_v = (
        0.7 * (bond_length - eq_bond_length) ** 2 + 0.3 / bond_length**8
    )
    assert np.abs(v - expected_v) < 1e-6
