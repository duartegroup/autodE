from autode import substitution
from autode.bond_rearrangement import BondRearrangement
from autode.molecule import Reactant
from autode.molecule import Product
from autode.reaction import Reaction
from autode.transition_states.locate_tss import get_reactant_and_product_complexes
import numpy as np


rearrang = BondRearrangement([(0, 1)], [(1, 2)])
coords = np.array(([0, 0, 0], [0, 0, 1], [0, 2, 1]))


def test_get_attacked_atom():
    attacked_atom = substitution.get_attacked_atom(rearrang)
    assert attacked_atom == 1


def test_get_lg_or_fr_atom():
    fr_atom = substitution.get_lg_or_fr_atom([(0, 1)], 1)
    assert fr_atom == 0


def test_get_normalised_lg_vector():
    lg_vector = substitution.get_normalised_lg_vector(rearrang, 1, coords)
    np.testing.assert_allclose(
        lg_vector, np.array([0., -1., 0., ]), atol=0.001)


def test_get_rot_matrix():
    lg_vector = np.array([0., -1., 0.])
    attack_vector = np.array([0., 0., 1., ])
    rot_matrix = substitution.get_rot_matrix(attack_vector, lg_vector)
    ideal_rot_matrix = np.array(([1., 0., 0.], [0., 0., 1., ], [0., -1., 0.]))
    np.testing.assert_allclose(rot_matrix, ideal_rot_matrix, atol=0.001)


hydroxide = Reactant(name='OH-', xyzs=[['F', 0., 0., 0.]], charge=-1)
methyl_chloride = Reactant(name='CH3Cl', smiles='[H]C([H])(Cl)[H]')
chloride = Product(name='Cl-',  xyzs=[['Cl', 0., 0., 0.]], charge=-1)
methyl_hydroxide = Product(name='CH3OH', smiles='[H]C([H])(F)[H]')
reaction = Reaction(hydroxide, methyl_chloride, chloride, methyl_hydroxide)
reactants, products = get_reactant_and_product_complexes(reaction)
bond_rearrang = BondRearrangement([(0,1)], [(1, 2)])


def test_set_complex_xyzs_translated_rotated():
    substitution.set_complex_xyzs_translated_rotated(reactants, reaction.reacs, bond_rearrang)  
    new_xyzs = reactants.xyzs
    CF =[(new_xyzs[0][i]-new_xyzs[1][i]) for i in range(1,4)]
    CCl =[(new_xyzs[1][i]-new_xyzs[2][i]) for i in range(1,4)]

    assert 1.999 < np.linalg.norm(CF) < 2.001
    np.testing.assert_allclose(np.cross(CF, CCl), [0,0,0], atol=0.001)
    assert new_xyzs[1] == ['C', 0.0,0.0,0.0]