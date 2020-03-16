from autode.complex import ReactantComplex
from autode.species import Species
from autode.atoms import Atom
from autode.substitution import SubstitutionCentre
from autode.mol_graphs import make_graph
from autode.transition_states.locate_tss import attack_cost


nh3 = Species(name='nh3', charge=0, mult=1, atoms=[Atom('N', -3.13130,  0.40668, -0.24910),
                                                   Atom('H', -3.53678,  0.88567,  0.55690),
                                                   Atom('H', -3.33721,  1.03222, -1.03052),
                                                   Atom('H', -3.75574, -0.38765, -0.40209)])
make_graph(nh3)


ch3cl = Species(name='CH3Cl', charge=0, mult=1, atoms=[Atom('Cl',  1.63751, -0.03204, -0.01858),
                                                       Atom('C ', -0.14528,  0.00318,  0.00160),
                                                       Atom('H ', -0.49672, -0.50478,  0.90708),
                                                       Atom('H ', -0.51741, -0.51407, -0.89014),
                                                       Atom('H ', -0.47810,  1.04781,  0.00015)])
make_graph(ch3cl)


def test_attack():

    reactant = ReactantComplex(nh3, ch3cl)
    subst_centre = SubstitutionCentre(a_atom_idx=0, c_atom_idx=5, x_atom_idx=4, a_atom_nn_idxs=[1, 2, 3])
    subst_centre.r0_ac = 1.38

    cost = attack_cost(reactant=reactant, subst_centres=[subst_centre], attacking_mol_idx=0,
                       a=1, b=1, c=1)

    term_a = (3.0 - 3 * 1.38)**2

    nh3_coords = nh3.get_coordinates()
    attack_vec = (1.0 / 3.0) * (nh3_coords[1] - nh3_coords[0] +
                                nh3_coords[2] - nh3_coords[0] +
                                nh3_coords[3] - nh3_coords[0])

