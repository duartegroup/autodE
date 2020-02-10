import os
import numpy as np
from scipy.optimize import minimize
from autode.config import Config
from autode.log import logger
from autode.geom import xyz2coord
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix
from multiprocessing import Pool
from csolvent import do_md
from csolvent import v


def get_coords_minimised_v(solute_coords, solvent_coords, solvent_bonds, solute_charges, solvent_charges, k, d0, tol):
    n_atoms = len(solvent_coords)
    os.environ['OMP_NUM_THREADS'] = str(1)
    init_coords = solvent_coords.reshape(3 * n_atoms, 1)
    flat_solute_coords = solute_coords.reshape(3*len(solute_coords), 1)
    res = minimize(v, x0=init_coords, args=(flat_solute_coords, solvent_bonds, solute_charges, solvent_charges, k, d0), method='BFGS', tol=tol)
    final_coords = res.x.reshape(n_atoms, 3)

    return final_coords


def simanl(solute_xyzs, solvent_xyzs, solvent_bonds, solute_charges, solvent_charges, solvent_size):

    n_steps = 100000
    temp = 1
    dt = 0.001
    k = 1000

    n_atoms = len(solvent_xyzs)
    d0 = np.zeros((n_atoms, n_atoms))
    solvent_coords = xyz2coord(solvent_xyzs)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if (i, j) in solvent_bonds or (j, i) in solvent_bonds:
                d0[i, j] = np.linalg.norm(solvent_coords[i] - solvent_coords[j])

    solute_coords = xyz2coord(solute_xyzs)

    logger.info('Running high temp MD')
    solvent_coords = do_md(solute_xyzs, solvent_xyzs, solvent_bonds, solute_charges, solvent_charges, n_steps, temp, dt, k, d0)

    logger.info('Minimising with BFGS')

    solvent_coords = get_coords_minimised_v(solute_coords, solvent_coords, solvent_bonds, solute_charges, solvent_charges, k, d0, len(solvent_xyzs)/5)

    solvent_xyzs = coords2xyzs(solvent_coords, solvent_xyzs)

    return solvent_xyzs


def add_solvent_molecules(solvent_charges, solvent_xyzs, solvent_bonds, solute_charges, solute_xyzs):

    np.random.seed()

    # centre solvent
    solvent_coords = xyz2coord(solvent_xyzs)
    solvent_centre = np.average(solvent_coords, axis=0)
    solvent_coords -= solvent_centre
    max_solvent_cart_values = np.max(solvent_coords, axis=0)
    min_solvent_cart_values = np.min(solvent_coords, axis=0)
    solvent_size = np.linalg.norm(max_solvent_cart_values - min_solvent_cart_values)

    # centre solute
    solute_coords = xyz2coord(solute_xyzs)
    max_solute_cart_values = np.max(solute_coords, axis=0)
    min_solute_cart_values = np.min(solute_coords, axis=0)
    solute_centre = (max_solute_cart_values + min_solute_cart_values) / 2
    solute_coords -= solute_centre
    centered_solute_xyzs = coords2xyzs(solute_coords, solute_xyzs)

    solvent_coords_on_sphere = []

    # place solvent molecules on a sphere around the centre of the solute
    radius = np.linalg.norm(max_solute_cart_values - min_solute_cart_values) + solvent_size + 3
    area_per_solvent_mol = (4 * np.pi * radius**2) / (300 * solvent_size)
    d = np.sqrt(area_per_solvent_mol)
    m_theta = int(np.pi/d)
    d_theta = np.pi / m_theta
    d_phi = area_per_solvent_mol/d_theta
    for m in range(0, m_theta):
        theta = np.pi * (m+0.5)/m_theta
        m_phi = int(2 * np.pi * np.sin(theta)/d_phi)
        for n in range(0, m_phi):
            phi = 2 * np.pi * n/m_phi
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            position = [x, y, z]
            new_solvent_mol_coords = random_rot_solvent(solvent_coords.copy()) + position
            clashing = False
            for coord in new_solvent_mol_coords:
                if any(np.linalg.norm(coord - other_coord) < solvent_size for other_coords in solvent_coords_on_sphere for other_coord in other_coords):
                    clashing = True
                    break
            if not clashing:
                solvent_coords_on_sphere.append(new_solvent_mol_coords)

    n_solvent_mols = 0
    all_solvent_xyzs = []
    bond_list = []
    all_charges = []

    # create solvent bond and charges lists
    for solvent_coords in solvent_coords_on_sphere:
        solvent_xyzs = coords2xyzs(solvent_coords, solvent_xyzs)
        for xyz in solvent_xyzs:
            all_solvent_xyzs.append(xyz)
        for bond in solvent_bonds:
            bond_list.append((bond[0] + n_solvent_mols * len(solvent_xyzs), bond[1] + n_solvent_mols * len(solvent_xyzs)))
        all_charges += solvent_charges
        n_solvent_mols += 1

    final_solvent_xyzs = simanl(centered_solute_xyzs, all_solvent_xyzs, bond_list, solute_charges, all_charges, solvent_size)

    return centered_solute_xyzs + final_solvent_xyzs


def random_rot_solvent(coords):
    axis = np.random.rand(3)
    theta = np.random.rand() * np.pi * 2
    rot_matrix = calc_rotation_matrix(axis, theta)
    for i in range(len(coords)):
        coords[i] = np.matmul(rot_matrix, coords[i])

    return coords


def gen_simanl_solvent_xyzs(name, solute_xyzs, solute_charges, solvent_xyzs, solvent_charges, solvent_bonds, n_simanls=4):

    logger.info('Looking for previously generated conformers')

    all_solvent_conf_xyzs = []

    no_conf = []
    for i in range(n_simanls):
        xyz_file_name_start = f'{name}_conf{i}'
        has_conf = False
        for filename in os.listdir(os.getcwd()):
            if filename.startswith(xyz_file_name_start) and filename.endswith('.xyz'):
                xyzs = []
                with open(filename, 'r') as file:
                    for line_no, line in enumerate(file):
                        if line_no > 1:
                            atom_label, x, y, z = line.split()
                            xyzs.append([atom_label, float(x), float(y), float(z)])
                all_solvent_conf_xyzs.append(xyzs)
                has_conf = True
                break
        if not has_conf:
            no_conf.append(i)

    logger.info(f'Found {len(all_solvent_conf_xyzs)} previously generated conformers')

    simanls_left = n_simanls - len(all_solvent_conf_xyzs)

    logger.info(f'Have {simanls_left} conformers left to generate')

    if simanls_left == 1:
        solvent_conf_xyzs = add_solvent_molecules(solvent_charges, solvent_xyzs, solvent_bonds, solute_charges, solute_xyzs)
        all_solvent_conf_xyzs.insert(no_conf[i], solvent_conf_xyzs)

    elif simanls_left > 1:
        logger.info(f'Splitting calculation into {Config.n_cores} threads')
        with Pool(processes=Config.n_cores) as pool:
            results = [pool.apply_async(add_solvent_molecules, (solvent_charges, solvent_xyzs, solvent_bonds, solute_charges, solute_xyzs))
                       for i in range(simanls_left)]
            solvent_conf_xyzs = [res.get(timeout=None) for res in results]

        for i in range(simanls_left):
            all_solvent_conf_xyzs.insert(no_conf[i], solvent_conf_xyzs[i])

    return all_solvent_conf_xyzs
