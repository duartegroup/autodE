import os
import numpy as np
import pytest
import autode as ade
from autode import bond_rearrangement as br
from autode.mol_graphs import MolecularGraph
from autode.species.molecule import Molecule
from autode.bond_rearrangement import BondRearrangement, get_bond_rearrangs
from autode.species.complex import ReactantComplex, ProductComplex
from autode.atoms import Atom
from autode.mol_graphs import is_isomorphic
from autode.mol_graphs import make_graph
from autode.utils import work_in_tmp_dir


# Some of the 'reactions' here are not physical, hence for some the graph will
# be regenerated allowing for invalid hydrogen valencies


def test_prune_small_rings3():
    # Square H4 "molecule"
    h4 = Molecule(
        atoms=[
            Atom("H"),
            Atom("H", x=0.5),
            Atom("H", y=0.5),
            Atom("H", x=0.5, y=0.5),
        ]
    )
    make_graph(h4, allow_invalid_valancies=True)

    # Some unphysical bond rearrangements
    three_mem = BondRearrangement(
        forming_bonds=[(0, 3)], breaking_bonds=[(1, 2)]
    )
    four_mem = BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(1, 2)]
    )
    bond_rearrs = [three_mem, four_mem]

    ade.Config.skip_small_ring_tss = True
    br.prune_small_ring_rearrs(bond_rearrs, h4)

    # Should not prune if there are different ring sizes
    assert len(bond_rearrs) == 2


def test_prune_small_rings2():
    reaction = ade.Reaction("CCCC=C>>C=C.C=CC")

    ade.Config.skip_small_ring_tss = False
    bond_rearrs = br.get_bond_rearrangs(
        reactant=reaction.reactant,
        product=reaction.product,
        name="tmp",
        save=False,
    )
    assert len(bond_rearrs) > 2

    ade.Config.skip_small_ring_tss = True

    br.prune_small_ring_rearrs(bond_rearrs, reaction.reactant)
    assert len(bond_rearrs) == 2

    # Should find the 6-membered TS
    assert bond_rearrs[0].n_membered_rings(reaction.reactant) == [
        6
    ] or bond_rearrs[1].n_membered_rings(reaction.reactant) == [6]


def test_n_membered_rings():
    h2o = Molecule(atoms=[Atom("O"), Atom("H", x=-1), Atom("H", x=1)])
    bond_rearr = BondRearrangement(forming_bonds=[(1, 2)])

    # Forming bond over H-H should give a single 3-membered ring
    assert bond_rearr.n_membered_rings(h2o) == [3]

    bond_rearr = BondRearrangement(breaking_bonds=[(0, 1)])
    assert bond_rearr.n_membered_rings(h2o) == []

    # Breaking an O-H and forming a H-H should not make any rings
    bond_rearr = BondRearrangement(
        breaking_bonds=[(0, 2)], forming_bonds=[(1, 2)]
    )
    assert bond_rearr.n_membered_rings(h2o) == [3]


def test_prune_small_rings():
    # Cope rearrangement reactant
    cope_r = Molecule(
        atoms=[
            Atom("C", -1.58954, 1.52916, -0.43451),
            Atom("C", -1.46263, 0.23506, 0.39601),
            Atom("C", -0.57752, 2.62322, -0.15485),
            Atom("H", -2.59004, 1.96603, -0.22830),
            Atom("H", -1.55607, 1.26799, -1.51381),
            Atom("C", 0.40039, 2.56394, 0.75883),
            Atom("C", -0.24032, -0.62491, 0.13974),
            Atom("H", -2.34641, -0.39922, 0.17008),
            Atom("H", -1.50638, 0.49516, 1.47520),
            Atom("C", 0.72280, -0.36227, -0.75367),
            Atom("H", -0.66513, 3.53229, -0.74242),
            Atom("H", 0.55469, 1.70002, 1.39347),
            Atom("H", 1.07048, 3.40870, 0.88117),
            Atom("H", -0.14975, -1.53366, 0.72733),
            Atom("H", 0.70779, 0.51623, -1.38684),
            Atom("H", 1.55578, -1.04956, -0.86026),
        ]
    )

    six_mem = BondRearrangement(
        forming_bonds=[(5, 9)], breaking_bonds=[(1, 0)]
    )
    assert six_mem.n_membered_rings(mol=cope_r) == [6]

    four_mem = BondRearrangement(
        forming_bonds=[(0, 9)], breaking_bonds=[(1, 0)]
    )
    assert four_mem.n_membered_rings(cope_r) == [4]

    ade.Config.skip_small_ring_tss = False

    bond_rearrs = [six_mem, four_mem]
    br.prune_small_ring_rearrs(possible_brs=bond_rearrs, mol=cope_r)
    # Should not prune if Config.skip_small_ring_tss = False
    assert len(bond_rearrs) == 2

    ade.Config.skip_small_ring_tss = True

    br.prune_small_ring_rearrs(possible_brs=bond_rearrs, mol=cope_r)
    # should remove the 4-membered ring
    assert len(bond_rearrs) == 1


def test_multiple_possibilities():
    r1 = Molecule(name="h_dot", smiles="[H]")
    r2 = Molecule(name="methane", smiles="C")
    p1 = Molecule(name="h2", smiles="[HH]")
    p2 = Molecule(name="ch3_dot", smiles="[CH3]")

    reac = ReactantComplex(r1, r2)

    rearrs = br.get_bond_rearrangs(
        reac, ProductComplex(p1, p2), name="H_subst", save=False
    )

    # All H abstractions are the same
    assert len(rearrs) == 1


def test_multiple_possibles2():
    # Attack on oxirane by AcO-
    reaction = ade.Reaction("[O-]C(C)=O.C1CO1>>[O-]CCOC(C)=O")

    rearrs = br.get_bond_rearrangs(
        reaction.reactant, reaction.product, name="oxir_attack", save=False
    )
    assert len(rearrs) == 1


def test_bondrearr_class():
    # Reaction H + H2 -> H2 + H
    rearrang = br.BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(1, 2)]
    )

    assert rearrang.n_fbonds == 1
    assert rearrang.n_bbonds == 1
    assert str(rearrang) == "0-1_1-2"

    rearrag2 = br.BondRearrangement(
        forming_bonds=[(0, 1)], breaking_bonds=[(1, 2)]
    )
    assert rearrag2 == rearrang

    mol = Molecule(
        name="mol",
        atoms=[
            Atom("H", 0.0, 0.0, 0.0),
            Atom("H", 0.0, 0.0, -0.7),
            Atom("H", 0.0, 0.0, 0.7),
        ],
    )
    mol_c = ReactantComplex(mol)

    assert set(rearrang.active_atoms) == {0, 1, 2}
    active_atom_nl = rearrang.get_active_atom_neighbour_lists(mol_c, depth=1)
    assert len(active_atom_nl) == 3
    assert active_atom_nl == [["H"], ["H"], ["H"]]

    #
    assert rearrang.get_active_atom_neighbour_lists(mol, depth=1) == [
        ["H"],
        ["H"],
        ["H"],
    ]

    # Cannot get neighbour list with atoms not in the complex
    with pytest.raises(ValueError):
        rearrang = br.BondRearrangement(forming_bonds=[(3, 4)])
        _ = rearrang.get_active_atom_neighbour_lists(mol_c, depth=1)


def test_get_bond_rearrangs(caplog):
    # ethane --> Ch3 + Ch3
    reac = Molecule(smiles="CC")
    prod = Molecule(
        atoms=[
            Atom("C", -8.3, 1.4, 0.0),
            Atom("C", 12, 1.7, -0.0),
            Atom("H", -8.6, 0.5, -0.5),
            Atom("H", -8.6, 2.3, -0.4),
            Atom("H", -8.6, 1.3, 1),
            Atom("H", 12.3, 1.7, -1.0),
            Atom("H", 12.4, 0.8, 0.4),
            Atom("H", 12.3, 2.5, 0.5),
        ]
    )

    assert br.get_bond_rearrangs(
        ReactantComplex(reac), ProductComplex(prod), name="test", save=False
    ) == [br.BondRearrangement(breaking_bonds=[(0, 1)])]

    # Rerunning the get function should read test_bond_rearrangs.txt, so modify
    #  it, swapping 0 and 1 in the breaking
    # bond then reopen
    with open("test_bond_rearrangs.txt", "w") as rearr_file:
        print("fbond\n" "bbonds\n" "1 0\n" "end", file=rearr_file)

    rearr = br.get_bond_rearrangs(
        ReactantComplex(reac), ProductComplex(prod), name="test"
    )[0]
    assert rearr == BondRearrangement(breaking_bonds=[(1, 0)])
    os.remove("test_bond_rearrangs.txt")

    # If we try to get the bond rearrangement from the other way
    # it should print a warning
    with caplog.at_level("INFO"):
        assert br.get_bond_rearrangs(
            ReactantComplex(prod),
            ProductComplex(reac),
            name="test2",
            save=False,
        ) == [br.BondRearrangement(forming_bonds=[(0, 1)])]
    assert "More bonds in product than in reactant" in caplog.text
    assert "suggest swapping them" in caplog.text

    # If reactants and products are identical then the rearrangement is
    # undetermined
    assert (
        br.get_bond_rearrangs(
            ReactantComplex(reac),
            ProductComplex(reac),
            name="test3",
            save=False,
        )
        is None
    )


def test_two_possibles():
    ch2ch3f = Molecule(
        name="radical", charge=0, mult=2, smiles="FC[C]([H])[H]"
    )

    ch3ch2f = Molecule(name="radical", charge=0, mult=2, smiles="C[C]([H])F")

    rearrs = br.get_bond_rearrangs(
        ReactantComplex(ch2ch3f),
        ProductComplex(ch3ch2f),
        name="H_migration",
        save=False,
    )

    # There are two possibilities for H migration by they should be considered
    # the same
    assert len(rearrs) == 1


def test_add_bond_rearrang():
    reac = Molecule(atoms=[Atom("H", 0, 0, 0), Atom("H", 0.6, 0, 0)])
    prod = Molecule(atoms=[Atom("H", 0, 0, 0), Atom("H", 10, 0, 0)])
    assert br.add_bond_rearrangment([], reac, prod, [], [(0, 1)]) == [
        br.BondRearrangement(breaking_bonds=[(0, 1)])
    ]


def test_generate_rearranged_graph():
    init_graph = MolecularGraph()
    final_graph = MolecularGraph()
    init_edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6)]
    final_edges = [(0, 1), (2, 3), (3, 4), (4, 5), (5, 6)]
    for edge in init_edges:
        init_graph.add_edge(*edge)
    for edge in final_edges:
        final_graph.add_edge(*edge)
    assert is_isomorphic(
        br.generate_rearranged_graph(init_graph, [(3, 4)], [(1, 2)]),
        final_graph,
    )


def test_2b():
    reac = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("H", 0.6, 0, 0), Atom("H", 1.2, 0, 0)]
    )
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("H", 10, 0, 0), Atom("H", 20, 0, 0)]
    )

    # Reactants to products must break two bonds
    rearrs = br.get_bond_rearrangs(
        ReactantComplex(reac),
        ProductComplex(prod),
        name="2b_test",
        save=False,
    )
    assert len(rearrs) == 1
    assert rearrs == [br.BondRearrangement(breaking_bonds=[(0, 1), (1, 2)])]


def test_3b():
    reac = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("H", 0.6, 0, 0),
            Atom("H", 1.2, 0, 0),
            Atom("H", 1.8, 0, 0),
        ]
    )
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("H", 10, 0, 0),
            Atom("H", 20, 0, 0),
            Atom("H", 30, 0, 0),
        ]
    )

    # Reactants to products must break three bonds
    assert br.get_bond_rearrangs(
        ReactantComplex(reac),
        ProductComplex(prod),
        name="3b_test",
        save=False,
    ) == [BondRearrangement(breaking_bonds=[(0, 1), (1, 2), (2, 3)])]


def test_1b1f():
    reac = Molecule(
        atoms=[Atom("C", 0, 0, 0), Atom("H", 0.6, 0, 0), Atom("H", 10, 0, 0)]
    )
    prod = Molecule(
        atoms=[Atom("C", 0, 0, 0), Atom("H", 10, 0, 0), Atom("H", 10.6, 0, 0)]
    )

    rearrs = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearrs == [
        br.BondRearrangement(forming_bonds=[(1, 2)], breaking_bonds=[(0, 1)])
    ]

    reac = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("H", 0.6, 0, 0), Atom("H", 10, 0, 0)]
    )
    prod = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("H", 10, 0, 0), Atom("H", 10.6, 0, 0)]
    )

    rearrs = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearrs == [
        br.BondRearrangement(forming_bonds=[(0, 2)], breaking_bonds=[(0, 1)])
    ]


def test_2b1f():
    reac = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("C", 0.6, 0, 0), Atom("O", 1.4, 0, 0)]
    )
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("C", 10, 0, 0), Atom("O", 0.6, 0, 0)]
    )

    rearrs = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearrs == [
        br.BondRearrangement(
            forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)]
        )
    ]

    reac = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("C", 0.6, 0, 0), Atom("H", 1.2, 0, 0)]
    )
    make_graph(reac, allow_invalid_valancies=True)
    prod = Molecule(
        atoms=[Atom("H", 0, 0, 0), Atom("C", 10, 0, 0), Atom("H", 0.6, 0, 0)]
    )
    rearrs = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearrs == [
        br.BondRearrangement(
            forming_bonds=[(0, 2)], breaking_bonds=[(0, 1), (1, 2)]
        )
    ]


def test_2b2f():
    reac = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 0.6, 0, 0),
            Atom("N", 10, 0, 0),
            Atom("O", 10.6, 0, 0),
        ]
    )
    prod = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 10, 0, 0),
            Atom("N", 0.6, 0, 0),
            Atom("O", 10.6, 0, 0),
        ]
    )

    rearrs = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearrs == [
        br.BondRearrangement(
            forming_bonds=[(0, 2), (1, 3)], breaking_bonds=[(0, 1), (2, 3)]
        )
    ]

    reac = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 0.6, 0, 0),
            Atom("H", 10, 0, 0),
            Atom("N", 10.6, 0, 0),
            Atom("O", 20, 0, 0),
        ]
    )
    prod = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 10, 0, 0),
            Atom("H", 1.2, 0, 0),
            Atom("N", 20, 0, 0),
            Atom("O", 0.6, 0, 0),
        ]
    )

    rearrs = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearrs == [
        br.BondRearrangement(
            forming_bonds=[(0, 4), (2, 4)], breaking_bonds=[(0, 1), (2, 3)]
        )
    ]

    reac = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 0.6, 0, 0),
            Atom("H", 1.2, 0, 0),
            Atom("O", 10, 0, 0),
        ]
    )
    prod = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 10, 0, 0),
            Atom("H", 11.2, 0, 0),
            Atom("O", 10.6, 0, 0),
        ]
    )
    rearrs = br.BondRearrGenerator(reac, prod, 0).get_valid_bond_rearrs()
    assert rearrs == [
        br.BondRearrangement(
            forming_bonds=[(0, 3), (1, 3)], breaking_bonds=[(0, 1), (1, 2)]
        ),
        br.BondRearrangement(
            forming_bonds=[(1, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)]
        ),
    ]

    reac = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 0.6, 0, 0),
            Atom("H", 1.2, 0, 0),
            Atom("O", 10, 0, 0),
        ]
    )
    prod = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 10, 0, 0),
            Atom("H", 1.2, 0, 0),
            Atom("O", 0.6, 0, 0),
        ]
    )

    rearr = br.get_bond_rearrangs(reac, prod, name="test", save=False)
    assert rearr == [
        br.BondRearrangement(
            forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)]
        )
    ]

    reac = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 0.6, 0, 0),
            Atom("N", 1.2, 0, 0),
            Atom("C", 10, 0, 0),
        ]
    )
    prod = Molecule(
        atoms=[
            Atom("H", 0, 0, 0),
            Atom("C", 10, 0, 0),
            Atom("N", 1.2, 0, 0),
            Atom("C", 0.6, 0, 0),
        ]
    )
    rearrs = br.BondRearrGenerator(reac, prod, 0, 2).get_valid_bond_rearrs()
    assert rearrs == [
        br.BondRearrangement(
            forming_bonds=[(0, 3), (2, 3)], breaking_bonds=[(0, 1), (1, 2)]
        )
    ]


def test_br_from_file():
    path = "/a/path/that/doesnt/exist"
    assert br.get_bond_rearrangs_from_file(filename=path) is None

    with open("tmp.txt", "w") as br_file:
        print("fbonds\n" "0 1\n" "end", file=br_file)

    saved_brs = br.get_bond_rearrangs_from_file(filename="tmp.txt")
    assert len(saved_brs) == 1

    saved_br = saved_brs[0]
    assert saved_br.n_fbonds == 1
    assert saved_br.n_bbonds == 0

    with open("tmp.txt", "w") as br_file:
        print(
            "fbonds\n" "1 12\n" "bbonds\n" "6 12\n" "7 8\n" "endn\n",
            file=br_file,
        )

    saved_brs = br.get_bond_rearrangs_from_file(filename="tmp.txt")
    assert len(saved_brs) == 1

    saved_br = saved_brs[0]
    assert saved_br.n_fbonds == 1
    assert saved_br.n_bbonds == 2

    os.remove("tmp.txt")


@work_in_tmp_dir()
def test_2b2f_single_bond_type():
    """
    Test that the bond rearrangement can be found where only OO bonds break
    and only OH bonds form. Not a very realistic reaction
    """

    reac = Molecule(
        atoms=[
            Atom("O", 0.0, 0.0, 0.0),
            Atom("O", 1.5, 0.0, 0.0),
            Atom("O", 0.0, 1.5, 0.0),
            Atom("O", 1.5, 1.5, 0.0),
            Atom("H", 9.0, 0.0, 0.0),
            Atom("H", 9.0, 2.0, 0.0),
        ]
    )

    prod = Molecule(
        atoms=[
            Atom("O", -9.0, 0.0, 0.0),
            Atom("O", 8.0, 0.0, 0.0),
            Atom("O", -9.0, 1.0, 0.0),
            Atom("O", 8.0, 1.0, 0.0),
            Atom("H", 9.0, 0.0, 0.0),
            Atom("H", 9.0, 1.0, 0.0),
        ]
    )

    brs = get_bond_rearrangs(reac, prod, "test")
    assert brs is not None and len(brs) == 1


@work_in_tmp_dir()
def test_metal_bond_rearr():
    rct = Molecule(
        atoms=[
            Atom("C", -1.3767, 0.0570, -1.3664),
            Atom("C", -2.6791, -0.0962, -0.9319),
            Atom("C", -2.9564, -0.6049, 0.3450),
            Atom("C", -1.9590, -1.0294, 1.2302),
            Atom("C", -0.5961, -0.9681, 0.9718),
            Atom("Rh", 0.2867, 0.0575, -0.4059),
            Atom("P", 2.4304, -0.6254, 0.3308),
            Atom("P", 0.1808, 2.2070, 0.2134),
            Atom("H", 0.4144, 2.4032, 1.6007),
            Atom("H", 0.9267, 3.3450, -0.2281),
            Atom("H", -1.1026, 2.8113, 0.1430),
            Atom("H", 3.2304, -1.3384, -0.6069),
            Atom("H", 3.5014, 0.1821, 0.8252),
            Atom("H", 2.4909, -1.5907, 1.3765),
            Atom("H", -4.0026, -0.7346, 0.6414),
            Atom("H", -2.2860, -1.4755, 2.1778),
            Atom("H", -3.5210, 0.0778, -1.6125),
            Atom("H", -1.2152, 0.2895, -2.4390),
            Atom("H", 0.0448, -1.3115, 1.8013),
        ]
    )

    prod = rct.copy()
    prod.coordinates = np.array(
        [
            [-2.13498901816895, -0.33612687288335, -0.86754280713549],
            [-2.52485510044462, 0.40370659667954, 0.31248882468858],
            [-2.11142719924124, -0.33450226041684, 1.43145119220829],
            [-1.46413065518446, -1.53453226838949, 0.94874413980954],
            [-1.5699580883199, -1.57824199166861, -0.46835599227779],
            [-0.24452290403628, 0.17294517838954, 0.19800468613376],
            [1.58455459079778, 0.03099824045501, 1.41573815831704],
            [0.65667523851578, 1.69691872713977, -1.11195946726777],
            [1.97524695781375, 2.17696797868182, -0.86040708916778],
            [0.83937461626034, 1.45484527521222, -2.50835466167594],
            [0.04811296591758, 2.97927455619315, -1.27732209848744],
            [2.70620482100756, 0.8650601517448, 1.13526655449729],
            [1.5524764441543, 0.27376850175013, 2.82361385040179],
            [2.3162173059062, -1.19299867782956, 1.50800208661024],
            [-2.2436105492963, -0.05703932355732, 2.47583874220784],
            [-1.06823763466507, -2.3302069408402, 1.57985644964237],
            [-3.03903079987432, 1.3632151872862, 0.32308325841509],
            [-2.35128706528227, -0.03857206339642, -1.89358109052685],
            [-1.23761392585987, -2.3820899945504, -1.12143473639278],
        ]
    )
    # Going from metallabenzene -> metal cyclopentadienyl, formation
    # of 4 bonds in one step!
    rearrs = br.get_bond_rearrangs(rct, prod, name="test", save=False)
    assert rearrs == [
        BondRearrangement(forming_bonds=[(0, 4), (1, 5), (2, 5), (3, 5)])
    ]


def test_bond_rearr_repr():
    bond_rearr = BondRearrangement([(0, 1), (2, 3)], [(1, 2)])
    assert repr(bond_rearr) == "Form(0-1,2-3)+Break(1-2)"
