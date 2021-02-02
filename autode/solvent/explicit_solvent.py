from copy import deepcopy
from math import ceil
import numpy as np
from autode.log import logger


def add_solvent_molecules(species, n_qm_solvent_mols, n_solvent_mols):
    """Add a specific number of solvent molecules around a solute"""
    # Initialise a new random seed and make a copy of the species' atoms. RandomState is thread safe
    rand = np.random.RandomState()

    logger.info(f'Adding solvent molecules around {species.name}')

    solvent_n_atoms = species.solvent_mol.n_atoms
    total_n_solvent_atoms = n_solvent_mols * solvent_n_atoms

    centre_species(species.solvent_mol)
    solvent_coords = species.solvent_mol.coordinates
    solvent_size = np.linalg.norm(np.max(solvent_coords, axis=0) - np.min(solvent_coords, axis=0))
    solvent_size = 1 if solvent_size < 1 else solvent_size

    centre_species(species)
    solute_coords = species.coordinates
    radius = np.linalg.norm(np.max(solute_coords, axis=0) - np.min(solute_coords, axis=0))
    radius = 2 if radius < 2 else radius

    solvent_area = (0.9*solvent_size) ** 2 * np.pi

    i = 1
    all_solvent_atoms = []
    while len(all_solvent_atoms) <= total_n_solvent_atoms:
        add_solvent_on_sphere(species, all_solvent_atoms, radius, solvent_area, i, rand)
        i += 1

    # TODO make this nicer?
    # Only take closest solvent molecules
    distances = []
    for i in range(int(len(all_solvent_atoms)/solvent_n_atoms)):
        solvent_mol_atoms = all_solvent_atoms[i*solvent_n_atoms:(i+1)*solvent_n_atoms]
        solvent_mol_coords = [atom.coord for atom in solvent_mol_atoms]
        distances.append(np.linalg.norm(np.average(solvent_mol_coords, axis=0)))

    species.qm_solvent_atoms = []
    species.mm_solvent_atoms = []
    sorted_distances = sorted(distances)
    for i in range(n_solvent_mols):
        original_index = distances.index(sorted_distances[i])
        solvent_mol_atoms = all_solvent_atoms[original_index*solvent_n_atoms:(original_index+1)*solvent_n_atoms]
        if i < n_qm_solvent_mols:
            species.qm_solvent_atoms += solvent_mol_atoms
        else:
            species.mm_solvent_atoms += solvent_mol_atoms

    return None


def centre_species(species):
    """Translates a species so its centre is at (0,0,0)"""
    species_coords = species.coordinates
    species_centre = np.average(species_coords, axis=0)
    species.translate(-species_centre)


def add_solvent_on_sphere(species, solvent_atoms, radius, solvent_mol_area, radius_mult, rand):
    """Packs solvent molecules semi-evenly on a sphere around the solvent molecule"""
    rad_to_use = (radius * radius_mult * 0.8) + 0.4
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
                phi = (2 * np.pi * n/n_on_ring) + 0.7*np.pi*(rand.rand()-0.5)/(n_on_ring)
            else:
                phi = (2 * np.pi * (n+0.5)/n_on_ring) + 0.7*np.pi*(rand.rand()-0.5)/(n_on_ring)
            # Add a little bit of randomness to the positioning
            rand_theta = theta + 0.35*np.pi*(rand.rand()-0.5)/(m_theta-1)
            rand_add = 0.4*radius * (rand.rand()-0.5)
            x = (rad_to_use + rand_add) * np.sin(rand_theta) * np.cos(phi)
            y = (rad_to_use + rand_add) * np.sin(rand_theta) * np.sin(phi)
            z = (rad_to_use + rand_add) * np.cos(rand_theta)
            position = [x, y, z]
            species.solvent_mol.rotate(axis=rand.uniform(-1.0, 1.0, 3), theta=2*np.pi*rand.rand())
            for atom in species.solvent_mol.atoms:
                new_atom = deepcopy(atom)
                new_atom.translate(position)
                solvent_atoms.append(new_atom)

    return None
