import os
import numpy as np
from autode.log import logger
from autode.geom import xyz2coord
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix
from autode.config import Config
from autode.solvent.qmmm import QMMM
from multiprocessing import Pool
from math import ceil, floor


def add_solvent_molecules(solute, solvent, n_solvent_mols):
    """Add a specific number of solvent molecules around a solute

    Arguments:
        solute (mol obj) -- solute molecule
        solvent (mol obj) -- solvent molecule
        n_solvent_mols (int) -- number of solvent molecules desired

    Returns:
        list, list, list, list -- solute_xyzs, solvent_xyzs, solvent_bonds, solvent_charges
    """
    np.random.seed()

    logger.info(f'Adding solvent molecules around {solute.name}')

    # centre solvent
    solvent_coords = xyz2coord(solvent.xyzs)
    solvent_centre = np.average(solvent_coords, axis=0)
    solvent_coords -= solvent_centre
    max_solvent_cart_values = np.max(solvent_coords, axis=0)
    min_solvent_cart_values = np.min(solvent_coords, axis=0)
    solvent_size = np.linalg.norm(max_solvent_cart_values - min_solvent_cart_values)
    if solvent_size == 0:
        solvent_size = 1

    # centre solute
    solute_coords = xyz2coord(solute.xyzs)
    max_solute_cart_values = np.max(solute_coords, axis=0)
    min_solute_cart_values = np.min(solute_coords, axis=0)
    solute_centre = (max_solute_cart_values + min_solute_cart_values) / 2
    solute_coords -= solute_centre
    centered_solute_xyzs = coords2xyzs(solute_coords, solute.xyzs)

    radius = (np.linalg.norm(max_solute_cart_values - min_solute_cart_values))
    if radius < 2:
        radius = 2
    solvent_mol_area = (0.9*solvent_size)**2 * np.pi

    solvent_bonds = list(solvent.graph.edges())
    solvent_coords_on_sphere = []
    all_solvent_xyzs = []
    all_solvent_bonds = []
    all_solvent_charges = []

    distances = []
    i = 1
    while len(solvent_coords_on_sphere) <= n_solvent_mols:
        solvent_coords_on_sphere += add_solvent_on_sphere(solvent_coords, radius, solvent_mol_area, i)
        i += 1
    for i, solvent_coords in enumerate(solvent_coords_on_sphere):
        distances.append(np.linalg.norm(solvent_coords))
    # only take closest solvent molecules
    sorted_by_dist_coords = [x for _, x in sorted(zip(distances, solvent_coords_on_sphere))]
    for i, solvent_coords in enumerate(sorted_by_dist_coords):
        if i == n_solvent_mols:
            break
        solvent_xyzs = coords2xyzs(solvent_coords, solvent.xyzs)
        for xyz in solvent_xyzs:
            all_solvent_xyzs.append(xyz)
        for bond in solvent_bonds:
            all_solvent_bonds.append((bond[0] + i * len(solvent_xyzs), bond[1] + i * len(solvent_xyzs)))
        all_solvent_charges += solvent.charges

    return centered_solute_xyzs, all_solvent_xyzs, all_solvent_bonds, all_solvent_charges


def add_solvent_on_sphere(solvent_coords, radius, solvent_mol_area, radius_mult):
    """Packs solvent molecules semi-evenly on a sphere around the solvent molecule

    Arguments:
        solvent_coords (np.array) -- solvent molecule coords
        radius (float) -- radius of the solute molecule
        solvent_mol_area (float) -- rough top down area of the molecule
        radius_mult (int) -- multiplier to use on the radius to get the radius of this sphere of solvent

    Returns:
        np.array -- coords of the solvent molecules on the sphere
    """
    rad_to_use = (radius * radius_mult * 0.8) + 0.4
    solvent_coords_on_sphere = []
    fit_on_sphere = ceil((4 * np.pi * rad_to_use**2) / solvent_mol_area)
    d = fit_on_sphere**(4/5)
    m_theta = ceil(d/np.pi)
    total_circum = 0
    for m in range(0, m_theta):
        total_circum += 2 * np.pi * np.sin(np.pi * (m+0.5)/m_theta)
    for m in range(0, m_theta):
        theta = np.pi * (m+0.5)/m_theta
        circum = 2 * np.pi * np.sin(theta)
        n_on_ring = int(round(circum * fit_on_sphere / total_circum))
        for n in range(0, n_on_ring):
            if m % 2 == 0:
                phi = (2 * np.pi * n/n_on_ring) + 0.7*np.pi*(np.random.rand()-0.5)/(n_on_ring)
            else:
                phi = (2 * np.pi * (n+0.5)/n_on_ring) + 0.7*np.pi*(np.random.rand()-0.5)/(n_on_ring)
            # add a little bit of randomness to the positioning
            rand_theta = theta + 0.35*np.pi*(np.random.rand()-0.5)/(m_theta-1)
            rand_add = 0.4*radius * (np.random.rand()-0.5)
            x = (rad_to_use + rand_add) * np.sin(rand_theta) * np.cos(phi)
            y = (rad_to_use + rand_add) * np.sin(rand_theta) * np.sin(phi)
            z = (rad_to_use + rand_add) * np.cos(rand_theta)
            position = [x, y, z]
            new_solvent_mol_coords = random_rot_solvent(solvent_coords.copy()) + position
            solvent_coords_on_sphere.append(new_solvent_mol_coords)
    return solvent_coords_on_sphere


def random_rot_solvent(coords):
    """rotate the solvent molecule randomly

    Arguments:
        coords (np.array) -- coords of solvent molecule

    Returns:
        np.array -- rotated coords of solvent molecule
    """
    axis = np.random.rand(3)
    theta = np.random.rand() * np.pi * 2
    rot_matrix = calc_rotation_matrix(axis, theta)
    for i in range(len(coords)):
        coords[i] = np.matmul(rot_matrix, coords[i])

    return coords


def run(solute, solvent, n_qm_solvent_mols, method, i):
    if os.path.exists(f'{solute.name}_qmmm_{i}.out'):
        lines = [line for line in open(f'{solute.name}_qmmm_{i}.out', 'r', encoding="utf-8")]
        xyzs_section = False
        xyzs = []
        for line in lines:
            if 'XYZS' in line:
                xyzs_section = True
            if 'Energy' in line:
                xyzs_section = False
                qmmm_energy = float(line.split()[2])
            if xyzs_section and len(line.split()) == 4:
                label, x, y, z = line.split()
                xyzs.append([label, float(x), float(y), float(z)])
    else:
        completed_qmmm = False
        while not completed_qmmm:
            try:
                for filename in os.listdir(os.getcwd()):
                    if f'{solute.name}_qmmm_{i}_step_' in filename:
                        os.remove(filename)
                n_solvent_mols = 700
                solute_xyzs, solvent_xyzs, solvent_bonds, solvent_charges = add_solvent_molecules(solute, solvent, n_solvent_mols)
                solute.xyzs = solute_xyzs
                os.environ['OPENMM_CPU_THREADS'] = str(1)
                os.environ['OMP_NUM_THREADS '] = str(1)
                qmmm = QMMM(solute, n_solvent_mols, solvent_xyzs, solvent_bonds, solvent_charges, n_qm_solvent_mols, i, method)
                qmmm.simulate()
                xyzs = qmmm.final_xyzs
                qmmm_energy = qmmm.final_energy
                completed_qmmm = True
            except:
                pass
        for filename in os.listdir(os.getcwd()):
            if f'{solute.name}_qmmm_{i}_step_' in filename:
                os.remove(filename)
        with open(f'{solute.name}_qmmm_{i}.out', 'w') as out_file:
            print('XYZS', file=out_file)
            [print('{:<3} {:^10.5f} {:^10.5f} {:^10.5f}'.format(*line), file=out_file) for line in xyzs]
            print(f'Energy = {qmmm_energy}', file=out_file)

    return xyzs, qmmm_energy


def do_explicit_solvent_qmmm(solute, solvent, method, n_confs=192, n_qm_solvent_mols=50):
    """Run explicit solvent qmmm calculations to find the lowest energy of the solvated molecule

    Arguments:
        solute (mol obj) -- molecule to be solvated, all coords will be fixed
        solvent (mol obj) -- solvating molecule, will be randomly placed around the solute the optimized with qmmm calculations
        method (ESW method) -- method to use for QM calculations

    Keyword Arguments:
        n_confs (int) -- number of differenct solvent configurations to calculate (default: {192})
        n_qm_solvent_mols (int) -- number of solvent molecules to place around the solute (default: {30})

    Returns:
        float, list, int -- energy, xyzs, n_qm_atoms
    """
    qmmm_energies = []
    qmmm_xyzs = []

    logger.info(f'Splitting calculation into {Config.n_cores} threads')
    with Pool(processes=Config.n_cores) as pool:
        results = [pool.apply_async(run, (solute, solvent, n_qm_solvent_mols, method, i)) for i in range(n_confs)]
        xyzs_and_energies = [res.get(timeout=None) for res in results]

    for xyzs, qmmm_energy in xyzs_and_energies:
        qmmm_energies.append(qmmm_energy)
        qmmm_xyzs.append(xyzs)

    min_e = min(qmmm_energies)
    lowest_energy_index = qmmm_energies.index(min_e)
    lowest_energy_qmmm_xyzs = qmmm_xyzs[lowest_energy_index]
    qmmm_n_qm_atoms = (n_qm_solvent_mols * len(solvent.xyzs)) + len(solute.xyzs)

    q = 0

    # get a bolztmann weighting of the energy
    boltzmann_qmmm_energy = 0
    for e in qmmm_energies:
        energy = e - min_e
        q += np.exp(-1052.58*energy)
        boltzmann_qmmm_energy = energy * np.exp(-1052.58*energy)
    boltzmann_qmmm_energy = (boltzmann_qmmm_energy / q) + min_e

    return boltzmann_qmmm_energy, lowest_energy_qmmm_xyzs, qmmm_n_qm_atoms
