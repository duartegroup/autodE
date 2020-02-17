import os
import numpy as np
from autode.log import logger
from autode.geom import xyz2coord
from autode.geom import coords2xyzs
from autode.geom import calc_rotation_matrix
from autode.solvent.qmmm import QMMM
from multiprocessing import Pool
from math import ceil, floor

from autode.input_output import xyzs2xyzfile


def add_solvent_molecules(solute, solvent):

    np.random.seed()

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
    if radius == 0:
        radius = 1
    solvent_mol_area = (2*solvent_size)**2 * np.pi

    n_qm_mols = 20
    solvent_bonds = list(solvent.graph.edges())

    qm_solvent_xyzs = []
    qm_solvent_bonds = []

    qm_mol_radius = find_radii(0.1, solvent_mol_area, n_qm_mols, True)
    if qm_mol_radius < radius:
        qm_mol_add = radius - qm_mol_radius
    else:
        qm_mol_add = 0
    solvent_coords_on_sphere = add_solvent_on_sphere(solvent_coords, qm_mol_radius, solvent_mol_area, qm_mol_add)
    for i, solvent_coords in enumerate(solvent_coords_on_sphere):
        if i == n_qm_mols:
            break
        solvent_xyzs = coords2xyzs(solvent_coords, solvent.xyzs)
        for xyz in solvent_xyzs:
            qm_solvent_xyzs.append(xyz)
        for bond in solvent_bonds:
            qm_solvent_bonds.append((bond[0] + i * len(solvent_xyzs), bond[1] + i * len(solvent_xyzs)))

    # TODO get right number of qm solvent mols

    n_mm_mols = 500
    solvent_coords_on_sphere = []
    radii = find_radii(qm_mol_radius + qm_mol_add, solvent_mol_area, n_mm_mols, False)
    radii_add = []
    for i, rad in enumerate(radii):
        if i == 0:
            radii_add.append(0)
        else:
            prev_rad = radii[i-1] + radii_add[i-1]
            if rad - prev_rad < radius:
                radii_add.append(radius + prev_rad - rad)
            else:
                radii_add.append(0)
    for rad, rad_add in zip(radii, radii_add):
        solvent_coords_on_sphere += add_solvent_on_sphere(solvent_coords, rad, solvent_mol_area, rad_add)

    mm_solvent_xyzs = []
    mm_solvent_bonds = []
    mm_charges = []

    for i, solvent_coords in enumerate(solvent_coords_on_sphere):
        if i == n_mm_mols:
            break
        solvent_xyzs = coords2xyzs(solvent_coords, solvent.xyzs)
        for xyz in solvent_xyzs:
            mm_solvent_xyzs.append(xyz)
        for bond in solvent_bonds:
            mm_solvent_bonds.append((bond[0] + i * len(solvent_xyzs), bond[1] + i * len(solvent_xyzs)))
        mm_charges += solvent.charges

    return centered_solute_xyzs, qm_solvent_xyzs, qm_solvent_bonds, mm_solvent_xyzs, mm_solvent_bonds, mm_charges


def find_radii(radius, solvent_mol_area, target, single_sphere):
    n_solvent_mols = []
    radii = []
    achieved_target = False
    radius_add = 0
    while not achieved_target:
        n_solvent_mols.append(floor((4 * np.pi * (radius+radius_add)**2) / solvent_mol_area))
        radii.append(radius+radius_add)
        if single_sphere:
            if target <= n_solvent_mols[-1] <= target + 3:
                return radii[-1]
        else:
            len_n_solvent_mols = len(n_solvent_mols)
            if len_n_solvent_mols > 2:
                for i in range(len_n_solvent_mols):
                    for j in range(len_n_solvent_mols):
                        if i < j:
                            for k in range(len_n_solvent_mols):
                                if j < k:
                                    if n_solvent_mols[i] + n_solvent_mols[j] + n_solvent_mols[k] == target:
                                        achieved_target = True
                                        radii_to_use = [radii[i], radii[j], radii[k]]
            if len_n_solvent_mols > 3 and not achieved_target:
                for i in range(len_n_solvent_mols):
                    for j in range(len_n_solvent_mols):
                        if i < j:
                            for k in range(len_n_solvent_mols):
                                if j < k:
                                    for l in range(len_n_solvent_mols):
                                        if k < l:
                                            if n_solvent_mols[i] + n_solvent_mols[j] + n_solvent_mols[k] + n_solvent_mols[l] == target:
                                                achieved_target = True
                                                radii_to_use = [radii[i], radii[j], radii[k], radii[l]]
            if len_n_solvent_mols > 4 and not achieved_target:
                for i in range(len_n_solvent_mols):
                    for j in range(len_n_solvent_mols):
                        if i < j:
                            for k in range(len_n_solvent_mols):
                                if j < k:
                                    for l in range(len_n_solvent_mols):
                                        if k < l:
                                            for m in range(len_n_solvent_mols):
                                                if l < m:
                                                    if n_solvent_mols[i] + n_solvent_mols[j] + n_solvent_mols[k] + n_solvent_mols[l] + n_solvent_mols[m] == target:
                                                        achieved_target = True
                                                        radii_to_use = [radii[i], radii[j], radii[k], radii[l], radii[m]]
        radius_add += 0.1*radius
    return radii_to_use


def add_solvent_on_sphere(solvent_coords, radius, solvent_mol_area, radius_add):
    solvent_coords_on_sphere = []
    fit_on_sphere = ceil((4 * np.pi * radius**2) / solvent_mol_area)
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
            rand_theta = theta + 0.35*np.pi*(np.random.rand()-0.5)/(m_theta-1)
            x = 0.7*(radius + radius_add + 0.5*(np.random.rand()-0.5)) * np.sin(rand_theta) * np.cos(phi)
            y = 0.7*(radius + radius_add + 0.5*(np.random.rand()-0.5)) * np.sin(rand_theta) * np.sin(phi)
            z = 0.7*(radius + radius_add + 0.5*(np.random.rand()-0.5)) * np.cos(rand_theta)
            position = [x, y, z]
            new_solvent_mol_coords = random_rot_solvent(solvent_coords.copy()) + position
            solvent_coords_on_sphere.append(new_solvent_mol_coords)
    return solvent_coords_on_sphere


def random_rot_solvent(coords):
    axis = np.random.rand(3)
    theta = np.random.rand() * np.pi * 2
    rot_matrix = calc_rotation_matrix(axis, theta)
    for i in range(len(coords)):
        coords[i] = np.matmul(rot_matrix, coords[i])

    return coords


def do_explicit_solvent_qmmm(solute, solvent, method, distance_constraints=None, hlevel=False, fix_qm=False, n=4):
    for _ in range(n):
        solute_xyzs, qm_solvent_xyzs, qm_solvent_bonds, mm_solvent_xyzs, mm_solvent_bonds, mm_charges = add_solvent_molecules(solute, solvent)
        solute.xyzs = solute_xyzs
        xyzs2xyzfile(solute_xyzs + qm_solvent_xyzs, 'qm.xyz')
        xyzs2xyzfile(solute_xyzs + qm_solvent_xyzs + mm_solvent_xyzs, 'full.xyz')
        qmmm = QMMM(solute, qm_solvent_xyzs, qm_solvent_bonds, mm_solvent_xyzs,
                    mm_solvent_bonds, mm_charges, method, distance_constraints, hlevel, fix_qm)
        qmmm.simulate()
        return qmmm.final_energy, qmmm.final_xyzs, qmmm.n_qm_atoms
