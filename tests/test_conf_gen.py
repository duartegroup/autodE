from autode.atoms import Atom
from autode.mol_graphs import make_graph
from autode.conformers import conf_gen
from autode.molecule import Molecule
from autode.config import Config
from autode.geom import get_krot_p_q
from scipy.spatial import distance_matrix
import numpy as np
import os

here = os.path.dirname(os.path.abspath(__file__))

butane = Molecule(name='butane', charge=0, mult=1, atoms=[Atom('C', -0.63938, -0.83117,  0.06651),
                                                          Atom('C',  0.89658, -0.77770,  0.06222),
                                                          Atom('H', -0.95115, -1.71970,  0.65729),
                                                          Atom('H', -1.01425, -0.95802, -0.97234),
                                                          Atom('C', -1.28709,  0.40256,  0.69550),
                                                          Atom('H',  1.27330, -1.74033, -0.34660),
                                                          Atom('H',  1.27226, -0.67376,  1.10332),
                                                          Atom('C',  1.46136,  0.35209, -0.79910),
                                                          Atom('H',  1.10865,  0.25011, -1.84737),
                                                          Atom('H',  2.57055,  0.30082, -0.79159),
                                                          Atom('H',  1.16428,  1.34504, -0.40486),
                                                          Atom('H', -0.93531,  0.53113,  1.74115),
                                                          Atom('H', -2.38997,  0.27394,  0.70568),
                                                          Atom('H', -1.05698,  1.31807,  0.11366)])

methane = Molecule(name='methane', charge=0, mult=1, atoms=[Atom('C', 0.70879, 0.95819, -0.92654),
                                                            Atom('H', 1.81819, 0.95820, -0.92655),
                                                            Atom('H', 0.33899, 0.14642, -0.26697),
                                                            Atom('H', 0.33899, 0.79287, -1.95935),
                                                            Atom('H', 0.33899, 1.93529, -0.55331)])


def test_conf_gen():

    atoms = conf_gen.get_simanl_atoms(species=methane)
    assert len(atoms) == 5

    # Ensure the new graph is identical
    regen = Molecule(name='regenerated_methane', atoms=atoms)

    assert regen.graph.edges == methane.graph.edges
    assert regen.graph.nodes == methane.graph.nodes


def test_conf_gen_dist_const():

    hydrogen = Molecule(name='H2', charge=0, mult=1, atoms=[Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.0),
                                                            Atom(atomic_symbol='H', x=0.0, y=0.0, z=0.7)])

    # H2 at a bond length (r) of 0.7 Å has a bond
    assert len(hydrogen.graph.edges) == 1

    # H2 at r = 1.5 Å is definitely not bonded
    atoms = conf_gen.get_simanl_atoms(species=hydrogen, dist_consts={(0, 1): 1.5})

    long_hydrogen = Molecule(name='H2', atoms=atoms, charge=0, mult=1)
    assert long_hydrogen.n_atoms == 2
    assert len(long_hydrogen.graph.edges) == 0


def test_chiral_rotation():

    chiral_ethane = Molecule(name='chiral_ethane', charge=0, mult=1,
                             atoms=[Atom('C', -0.26307, 0.59858, -0.07141),
                                    Atom('C', 1.26597, 0.60740, -0.09729),
                                    Atom('Cl', -0.91282, 2.25811, 0.01409),
                                    Atom('F', -0.72365, -0.12709, 1.01313),
                                    Atom('H', -0.64392, 0.13084, -1.00380),
                                    Atom('Cl', 1.93888, 1.31880, 1.39553),
                                    Atom('H', 1.61975, 1.19877, -0.96823),
                                    Atom('Br', 1.94229, -1.20011, -0.28203)])

    chiral_ethane.graph.nodes[0]['stereo'] = True
    chiral_ethane.graph.nodes[1]['stereo'] = True

    atoms = conf_gen.get_simanl_atoms(chiral_ethane)
    regen = Molecule(name='regenerated_ethane', charge=0, mult=1, atoms=atoms)

    regen_coords = regen.get_coordinates()
    coords = chiral_ethane.get_coordinates()

    # Atom indexes of the C(C)(Cl)(F)(H) chiral centre
    ccclfh = [0, 1, 2, 3, 4]

    # Atom indexes of the C(C)(Cl)(Br)(H) chiral centre
    ccclbrh = [1, 0, 5, 7, 6]

    for centre_idxs in [ccclfh, ccclbrh]:
        # Ensure the fragmented centres map almost identically
        rot_mat, p, q = get_krot_p_q(template_coords=coords[centre_idxs], coords_to_fit=regen_coords[centre_idxs])
        fitted_centre1 = np.array([np.matmul(rot_mat, coord - p) + q for coord in regen_coords[centre_idxs]])

        # RMSD on the 5 atoms should be < 0.1 Å
        assert np.sqrt(np.average(np.square(fitted_centre1 - coords[centre_idxs]))) < 1E-1


def test_butene():

    butene = Molecule(name='z-but-2-ene', charge=0, mult=1,
                      atoms=[Atom('C', -1.69185, -0.28379, -0.01192),
                             Atom('C', -0.35502, -0.40751,  0.01672),
                             Atom('C', -2.39437,  1.04266, -0.03290),
                             Atom('H', -2.13824,  1.62497,  0.87700),
                             Atom('H', -3.49272,  0.88343, -0.05542),
                             Atom('H', -2.09982,  1.61679, -0.93634),
                             Atom('C',  0.57915,  0.76747,  0.03048),
                             Atom('H',  0.43383,  1.38170, -0.88288),
                             Atom('H',  1.62959,  0.40938,  0.05452),
                             Atom('H',  0.39550,  1.39110,  0.93046),
                             Atom('H', -2.29700, -1.18572, -0.02030),
                             Atom('H',  0.07422, -1.40516,  0.03058)])

    butene.graph.nodes[0]['stereo'] = True
    butene.graph.nodes[1]['stereo'] = True

    # Conformer generation should retain the stereochemistry
    atoms = conf_gen.get_simanl_atoms(species=butene)
    regen = Molecule(name='regenerated_butene', atoms=atoms, charge=0, mult=1)

    # TODO finish this
    # regen.print_xyz_file()
