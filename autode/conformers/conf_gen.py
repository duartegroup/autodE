from copy import deepcopy
from itertools import combinations
import numpy as np
import os
from scipy.optimize import minimize
from time import time
from cconf_gen import v
from cconf_gen import dvdr
from cconf_gen import stereo
from cconf_gen import dstereodr
from autode.bond_lengths import get_ideal_bond_length_matrix
from autode.config import Config
from autode.input_output import xyz_file_to_atoms
from autode.input_output import atoms_to_xyz_file
from autode.log import logger
from autode.mol_graphs import split_mol_across_bond
from autode.exceptions import CannotSplitAcrossBond

import itertools
from autode.geom import calc_rmsd


def get_coords_minimised_v(coords, bonds, k, c, d0, tol, fixed_bonds, td_stereo, pi_stereo):
    # TODO divide and conquer?

    n_atoms = len(coords)
    os.environ['OMP_NUM_THREADS'] = str(1)

    init_coords = coords.flatten()
    stereo_res = minimize(stereo, x0=init_coords, args=(td_stereo, pi_stereo), method='CG', jac=dstereodr)
    full_res = minimize(v, x0=stereo_res.x, args=(bonds, k, d0, c, fixed_bonds, td_stereo, pi_stereo), method='CG', tol=tol, jac=dvdr)

    return full_res.x.reshape(n_atoms, 3)


def get_atoms_from_generated_file(species, xyz_filename):
    """Get the atoms from a previously generated  .xyz file, if the atoms match"""

    if not os.path.exists(xyz_filename):
        return None

    atoms = xyz_file_to_atoms(filename=xyz_filename)

    # Ensure the xyz file has the correct atoms
    all_atoms_match = all(atoms[i].label == species.atoms[i].label for i in range(species.n_atoms))

    if len(atoms) == species.n_atoms and all_atoms_match:
        logger.info('Conformer has already been generated')
        return atoms

    return None


def get_ordered_td_stereochem_list(species):
    stereochem = []

    for i in range(species.n_atoms):
        if species.graph.nodes[i]['stereo'] == '@td':
            stereochem.append(sorted(species.graph.neighbors(i)))
        elif species.graph.nodes[i]['stereo'] == '@@td':
            neighbours = sorted(species.graph.neighbors(i))
            order = [1, 0, 2, 3]
            stereochem.append([neighbours[j] for j in order])

    return np.asarray(stereochem).flatten()


def get_simanl_atoms(species, dist_consts={}, conf_n=0):
    """V(r) = Σ_bonds k(d - d0)^2 + Σ_ij c/d^4

    Arguments:
        species (autode.species.Species): Species, Molecule, TSguess, TS
        dist_consts (dict): Key = tuple of atom indexes, Value = distance
        conf_n (int): Number of this conformer generated

    Returns:
        (np.ndarray): Coordinates of the generated conformer
    """
    xyz_filename = f'{species.name}_conf{conf_n}_siman.xyz'

    # saved_atoms = get_atoms_from_generated_file(species, xyz_filename)
    # if saved_atoms is not None:
    #     return saved_atoms

    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = np.random.RandomState()
    atoms = deepcopy(species.atoms)

    # Add the distance constraints as fixed bonds
    d0 = get_ideal_bond_length_matrix(atoms=species.atoms, bonds=species.graph.edges())

    constrained_bonds = []
    for bond, length in dist_consts.items():
        i, j = bond
        d0[i, j] = length
        d0[j, i] = length
        constrained_bonds.append(bond)

    td_stereo = get_ordered_td_stereochem_list(species)
    print(td_stereo)
    #pi_stereo = np.array([0, 4, 3, 5])
    pi_stereo = np.array([])
    # Randomise coordinates that aren't fixed by shifting a maximum of autode.Config.max_atom_displacement in x, y, z
    factor = Config.max_atom_displacement / np.sqrt(3)

    # correct_coords = np.array([[0.86947,  0.75269, -1.21051],
    #                            [1.07036,  0.16546, -0.02600],
    #                            [-0.10872,  -0.21419, 0.69418],
    #                            [0.10112,  -0.79789, 1.87891],
    #                            [2.00585,  0.04914, 0.27092],
    #                            [-1.04746,  -0.07925, 0.41600]])
    correct_coords = np.array(([-1.11571,   -0.36960,   0.76471],
                               [-0.41454,   0.90856,   -0.45644],
                               [-1.29744,   0.29667,   -1.82383],
                               [1.15442,   -0.44057,   -0.84362],
                               [-0.09010,   1.80038,   -0.18172]))

    failed = 0
    start = time()

    for i in range(1):
        [atom.translate(vec=factor * rand.uniform(-1, 1, 3)) for atom in atoms]

        logger.info('Minimising species...')
        st = time()

        coords = get_coords_minimised_v(coords=np.array([atom.coord for atom in atoms]), bonds=species.graph.edges,
                                        k=1.0, c=0.01, d0=d0, tol=species.n_atoms/5E4, fixed_bonds=constrained_bonds,
                                        td_stereo=td_stereo, pi_stereo=pi_stereo)
        logger.info(f'                  ... ({time()-st:.3f} s)')

        stereo_energy = stereo(coords.flatten(), td_stereo, pi_stereo)

        print(calc_rmsd(coords, correct_coords))
        if calc_rmsd(coords, correct_coords) > 0.1:
            failed += 1
            break

    print(f'{failed} failed out of 10000 in {time()-start:.3f} s')

    # Set the coordinates of the new atoms
    for i, atom in enumerate(atoms):
        atom.coord = coords[i]

    # Print an xyz file so rerunning will read the file
    atoms_to_xyz_file(atoms=atoms, filename=xyz_filename)

    exit()

    return atoms
