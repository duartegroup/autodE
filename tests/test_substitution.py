import os
from autode.species.molecule import Reactant
from autode.species.complex import ReactantComplex
from autode.atoms import Atom
from autode.bond_rearrangement import BondRearrangement
from autode.substitution import (
    get_substc_and_add_dummy_atoms,
    attack_cost,
    SubstitutionCentre,
)

here = os.path.dirname(os.path.abspath(__file__))


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
reac_complex = ReactantComplex(f, ch3cl)

bond_rearr = BondRearrangement(breaking_bonds=[(2, 1)], forming_bonds=[(0, 2)])


def test_subst_centre():

    subst_centers = get_substc_and_add_dummy_atoms(
        reactant=reac_complex, bond_rearrangement=bond_rearr, shift_factor=2
    )
    # Only one atom gets attacked in an SN2
    assert len(subst_centers) == 1

    sc = subst_centers[0]
    assert type(sc) is SubstitutionCentre
    assert reac_complex.atoms[sc.a_atom].label == "F"
    assert reac_complex.atoms[sc.c_atom].label == "C"
    assert reac_complex.atoms[sc.x_atom].label == "Cl"

    # The attacking flouride ion doesn't have any nearest neighbours
    assert len(sc.a_atom_nn) == 0

    # The attacking atom to substitution centre atom ideal bond length
    # should be ~ 3 Å
    assert 2.0 < sc.r0_ac < 5.0


def test_attack_cost():

    subst_centers = get_substc_and_add_dummy_atoms(
        reactant=reac_complex, bond_rearrangement=bond_rearr, shift_factor=2
    )

    ideal_complex = ReactantComplex(f, ch3cl)
    ideal_complex.atoms = [
        Atom("F", -2.99674, -0.35248, 0.17493),
        Atom("Cl", 1.63664, 0.02010, -0.05829),
        Atom("C", -0.14524, -0.00136, 0.00498),
        Atom("H", -0.52169, -0.54637, -0.86809),
        Atom("H", -0.45804, -0.50420, 0.92747),
        Atom("H", -0.51166, 1.03181, -0.00597),
    ]

    cost = attack_cost(reac_complex, subst_centers, attacking_mol_idx=0)
    ideal_attack_cost = attack_cost(
        ideal_complex, subst_centers, attacking_mol_idx=0
    )

    # The cost function should be larger for the randomly located reaction
    # complex compared to the ideal
    assert cost >= ideal_attack_cost
