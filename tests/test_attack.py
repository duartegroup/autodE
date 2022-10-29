from autode.species.complex import ReactantComplex
from autode.species.species import Species
from autode.atoms import Atom
from autode.substitution import SubstitutionCentre
from autode.mol_graphs import make_graph
from autode.substitution import attack_cost
from autode.substitution import get_cost_rotate_translate
import numpy as np


nh3 = Species(
    name="nh3",
    charge=0,
    mult=1,
    atoms=[
        Atom("N", -3.13130, 0.40668, -0.24910),
        Atom("H", -3.53678, 0.88567, 0.55690),
        Atom("H", -3.33721, 1.03222, -1.03052),
        Atom("H", -3.75574, -0.38765, -0.40209),
    ],
)
make_graph(nh3)


ch3cl = Species(
    name="CH3Cl",
    charge=0,
    mult=1,
    atoms=[
        Atom("Cl", 1.63751, -0.03204, -0.01858),
        Atom("C", -0.14528, 0.00318, 0.00160),
        Atom("H", -0.49672, -0.50478, 0.90708),
        Atom("H", -0.51741, -0.51407, -0.89014),
        Atom("H", -0.47810, 1.04781, 0.00015),
    ],
)
make_graph(ch3cl)


def test_attack():

    reactant = ReactantComplex(nh3, ch3cl)
    subst_centre = SubstitutionCentre(
        a_atom_idx=0, c_atom_idx=5, x_atom_idx=4, a_atom_nn_idxs=[1, 2, 3]
    )
    subst_centre.r0_ac = 1.38

    cost = attack_cost(
        reactant=reactant,
        subst_centres=[subst_centre],
        attacking_mol_idx=0,
        a=1,
        b=1,
        c=10,
        d=1,
    )

    assert np.abs(cost - 2.919) < 1e-3

    # Rotation by 2π in place, translation by 0.0 and rotation by another 2π
    # should leave the cost unchanged..

    rot_axis_inplace = [1.0, 1.0, 1.0]
    rot_angle_inplace = 2 * np.pi

    translation_vec = [0.0, 0.0, 0.0]

    rot_axis = [1.0, 1.0, 1.0]
    rot_angle = 2 * np.pi

    x = (
        rot_axis_inplace
        + [rot_angle_inplace]
        + translation_vec
        + rot_axis
        + [rot_angle]
    )

    cost_trans_rot = get_cost_rotate_translate(
        x=np.array(x),
        reactant=reactant,
        subst_centres=[subst_centre],
        attacking_mol_idx=0,
    )

    # Requires a=1, b=1, c=1, d=10 in the attack_cost() called by get_cost_
    # rotate_translate()
    assert np.abs(cost_trans_rot - 3.5072) < 1e-3
