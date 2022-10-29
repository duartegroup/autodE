from autode.transition_states.truncation import get_truncated_species
from autode.bond_rearrangement import BondRearrangement
from autode.input_output import xyz_file_to_atoms
from autode.mol_graphs import is_isomorphic
from autode.species.complex import ReactantComplex
from autode.species.molecule import Reactant
from autode.atoms import Atom
from . import testutils
import os

here = os.path.dirname(os.path.abspath(__file__))


methane = Reactant(
    name="methane",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", 0.93919, -0.81963, 0.00000),
        Atom("H", 2.04859, -0.81963, -0.00000),
        Atom("H", 0.56939, -0.25105, 0.87791),
        Atom("H", 0.56938, -1.86422, 0.05345),
        Atom("H", 0.56938, -0.34363, -0.93136),
    ],
)

ethene = Reactant(
    name="ethene",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", 0.84102, -0.74223, 0.00000),
        Atom("C", -0.20368, 0.08149, 0.00000),
        Atom("H", 1.63961, -0.61350, -0.72376),
        Atom("H", 0.90214, -1.54881, 0.72376),
        Atom("H", -0.26479, 0.88807, -0.72376),
        Atom("H", -1.00226, -0.04723, 0.72376),
    ],
)

propene = Reactant(
    name="propene",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", 1.06269, -0.71502, 0.09680),
        Atom("C", 0.01380, 0.10714, 0.00458),
        Atom("H", 0.14446, 1.16840, -0.18383),
        Atom("H", -0.99217, -0.28355, 0.11871),
        Atom("C", 2.47243, -0.22658, -0.05300),
        Atom("H", 0.89408, -1.77083, 0.28604),
        Atom("H", 2.51402, 0.86756, -0.24289),
        Atom("H", 2.95379, -0.75333, -0.90290),
        Atom("H", 3.03695, -0.44766, 0.87649),
    ],
)

but1ene = Reactant(
    name="but-1-ene",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", 1.32424, -0.75672, 0.09135),
        Atom("C", 0.19057, -0.05301, 0.04534),
        Atom("H", 0.20022, 1.00569, -0.19632),
        Atom("H", -0.75861, -0.53718, 0.25084),
        Atom("C", 2.64941, -0.10351, -0.19055),
        Atom("H", 1.27555, -1.81452, 0.33672),
        Atom("C", 3.79608, -1.10307, -0.07851),
        Atom("H", 2.81589, 0.72011, 0.53717),
        Atom("H", 2.63913, 0.32062, -1.21799),
        Atom("H", 3.83918, -1.52715, 0.94757),
        Atom("H", 4.75802, -0.59134, -0.29261),
        Atom("H", 3.66134, -1.92824, -0.81028),
    ],
)

benzene = Reactant(name="benzene", charge=0, mult=1, smiles="c1ccccc1")

ethanol = Reactant(
    name="ethanol",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", -1.12058, -0.88145, -0.01072),
        Atom("C", 0.06169, 0.07347, -0.11534),
        Atom("H", -1.23059, -1.23894, 1.03497),
        Atom("H", -0.96469, -1.75405, -0.67985),
        Atom("H", -2.05248, -0.35802, -0.31150),
        Atom("O", 1.25088, -0.57948, 0.23621),
        Atom("H", -0.09854, 0.96628, 0.53077),
        Atom("H", 0.15114, 0.43369, -1.16189),
        Atom("H", 1.26514, -0.63012, 1.22767),
    ],
)

methlyethylether = Reactant(
    name="ether",
    charge=0,
    mult=1,
    atoms=[
        Atom("C", -1.25448, -0.89454, -0.18195),
        Atom("C", -0.05755, 0.05009, -0.17717),
        Atom("H", -1.39475, -1.34927, 0.82063),
        Atom("H", -1.09810, -1.70182, -0.92836),
        Atom("H", -2.17364, -0.33138, -0.44843),
        Atom("O", 1.13580, -0.66150, 0.09251),
        Atom("H", -0.23702, 0.89804, 0.52589),
        Atom("H", 0.03878, 0.50740, -1.18451),
        Atom("C", 1.44492, -0.60526, 1.46737),
        Atom("H", 2.35098, -1.21969, 1.64663),
        Atom("H", 0.63086, -1.03969, 2.08801),
        Atom("H", 1.68693, 0.43411, 1.78118),
    ],
)


def test_core_strip():
    bond_rearr = BondRearrangement(breaking_bonds=[(0, 1)])

    stripped = get_truncated_species(methane, bond_rearr)
    # Should not strip any atoms if the carbon is designated as active
    assert stripped.n_atoms == 5

    stripped = get_truncated_species(ethene, bond_rearr)
    assert stripped.n_atoms == 6

    bond_rearr = BondRearrangement(breaking_bonds=[(1, 3)])
    # Propene should strip to ethene if the terminal C=C is the active atom
    stripped = get_truncated_species(propene, bond_rearr)
    assert stripped.n_atoms == 6
    assert is_isomorphic(stripped.graph, ethene.graph)

    # But-1-ene should strip to ethene if the terminal C=C is the active atom
    stripped = get_truncated_species(but1ene, bond_rearr)
    assert stripped.n_atoms == 6
    assert is_isomorphic(stripped.graph, ethene.graph)

    # Benzene shouldn't be truncated at all
    stripped = get_truncated_species(benzene, bond_rearr)
    assert stripped.n_atoms == 12

    bond_rearr = BondRearrangement(breaking_bonds=[(0, 1)])
    # Ethanol with the terminal C as the active atom should not replace the OH
    # with a H
    stripped = get_truncated_species(ethanol, bond_rearr)
    assert stripped.n_atoms == 9

    # Ether with the terminal C as the active atom should replace the OMe with
    # OH
    stripped = get_truncated_species(methlyethylether, bond_rearr)
    assert stripped.n_atoms == 9
    assert is_isomorphic(stripped.graph, ethanol.graph)


def test_reactant_complex_truncation():

    # Non-sensical bond rearrangement
    bond_rearr = BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(0, 5)]
    )

    methane_dimer = ReactantComplex(methane, methane)

    # Should not truncate methane dimer at all
    truncated = get_truncated_species(methane_dimer, bond_rearr)
    assert truncated.n_atoms == 10


def test_product_complex_truncation():

    # H atom transfer from methane to ethene
    bond_rearr = BondRearrangement(
        breaking_bonds=[(0, 1)], forming_bonds=[(1, 5)]
    )

    methane_ethene = ReactantComplex(methane, ethene, name="product_complex")

    # Should retain all atoms
    truncated = get_truncated_species(methane_ethene, bond_rearr)
    assert truncated.n_atoms == 11


def test_enone_truncation():

    enone = Reactant(name="enone", smiles="CC(O)=CC(=O)OC")
    reactant = ReactantComplex(enone)

    bond_rearr = BondRearrangement(
        breaking_bonds=[(2, 11)], forming_bonds=[(11, 5)]
    )
    truncated = get_truncated_species(reactant, bond_rearr)
    assert truncated.n_atoms == 10
    assert truncated.graph.number_of_edges() == 9


@testutils.work_in_zipped_dir(os.path.join(here, "data", "truncation.zip"))
def test_large_truncation():

    mol = ReactantComplex(
        Reactant(name="product", atoms=xyz_file_to_atoms("product.xyz"))
    )

    bond_rearr = BondRearrangement(breaking_bonds=[(7, 8), (14, 18)])

    assert mol.n_atoms == 50

    truncated = get_truncated_species(
        species=mol, bond_rearrangement=bond_rearr
    )

    assert truncated.n_atoms == 27
    assert truncated.graph.number_of_edges() == 28


@testutils.work_in_zipped_dir(os.path.join(here, "data", "truncation.zip"))
def test_two_component_truncation():

    propylbromide = Reactant(name="RBr", atoms=xyz_file_to_atoms("RBr.xyz"))
    chloride = Reactant(name="Cl", smiles="[Cl-]")

    mol = ReactantComplex(chloride, propylbromide)
    bond_rearr = BondRearrangement(
        forming_bonds=[(0, 3)], breaking_bonds=[(3, 4)]
    )

    truncated = get_truncated_species(
        species=mol, bond_rearrangement=bond_rearr
    )

    # Should truncate to ethylbromide + Cl-
    assert truncated.n_atoms == 9
