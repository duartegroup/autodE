from autode.log import logger
from autode.constants import Constants
from autode.geom import get_krot_p_q
import numpy as np


def get_unique_confs(conformers, energy_threshold_kj=1):
    """
    For a list of conformers return those that are unique based on an energy threshold

    Arguments:
        conformers (list(autode.conformer.Conformer)):
        energy_threshold_kj (float): Energy threshold in kJ mol-1

    Returns:
        (list(autode.conformers.conformers.Conformer)):
    """
    logger.info(f'Stripping conformers with energy ∆E < {energy_threshold_kj} kJ mol-1 to others')

    n_conformers = len(conformers)
    delta_e = energy_threshold_kj / Constants.ha2kJmol   # Conformer.energy is in Hartrees

    # The first conformer must be unique
    unique_conformers = conformers[:1]

    for i in range(1, n_conformers):

        if conformers[i].energy is None:
            logger.error('Conformer had no energy. Stripping')
            continue

        # Iterate through all the unique conformers already found and check that the energy is not similar
        unique = True
        for j in range(len(unique_conformers)):
            if conformers[i].energy - delta_e < conformers[j].energy < conformers[i].energy + delta_e:
                unique = False
                break

        if unique:
            unique_conformers.append(conformers[i])

    n_unique_conformers = len(unique_conformers)
    logger.info(f'Stripped {n_conformers - n_unique_conformers} conformer(s) from a total of {n_conformers}')

    return unique_conformers


def check_rmsd(conf, conf_list, rmsd_tol=0.5):
    for other_conf in conf_list:
        conf_coords = conf.get_coordinates()
        other_conf_coords = other_conf.get_coordinates()
        rot_mat, p, q = get_krot_p_q(template_coords=conf_coords, coords_to_fit=other_conf_coords)
        fitted_other_conf_coords = np.array([np.matmul(rot_mat, coord - p) + q for coord in other_conf_coords])

        rmsd = np.sqrt(np.average(np.square(fitted_other_conf_coords - conf_coords)))
        if rmsd < rmsd_tol:
            return False

    return True
