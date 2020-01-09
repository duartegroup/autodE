from autode.log import logger

# A set of reasonable valances for anionic/neutral/cationic atoms
valid_valances = {'H': [0, 1],
                  'B': [3, 4],
                  'C': [2, 3, 4],
                  'N': [2, 3, 4],
                  'O': [1, 2, 3],
                  'F': [0, 1],
                  'Si': [2, 3, 4],
                  'P': [2, 3, 4, 5, 6],
                  'S': [2, 3, 4, 5, 6],
                  'Cl': [0, 1, 2, 3, 4],
                  'Br': [0, 1, 2, 3, 4],
                  'I': [0, 1, 2, 3, 4, 5, 6],
                  'Rh': [0, 1, 2, 3, 4, 5, 6]
                  }

atomic_weights = {'H': 1,
                  'He': 4,
                  'Li': 7,
                  'Be': 9,
                  'B': 11,
                  'C': 12,
                  'N': 14,
                  'O': 16,
                  'F': 19,
                  'Ne': 20,
                  'Na': 23,
                  'Mg': 24,
                  'Al': 27,
                  'Si': 28,
                  'P': 31,
                  'S': 32,
                  'Cl': 36,
                  'Ar': 40,
                  'K': 39,
                  'Ca': 40,
                  'Sc': 45,
                  'Ti': 48,
                  'V': 50,
                  'Cr': 52,
                  'Mn': 55,
                  'Fe': 56,
                  'Co': 59,
                  'Ni': 59,
                  'Cu': 64,
                  'Zn': 65
                  }

# Van der Waals radii in Å taken from http://www.webelements.com/periodicity/van_der_waals_radius/

vdw_radii = {
    'H': 1.20,
    'He': 1.40,
    'Li': 1.82,
    'Be': 1.53,
    'B': 1.92,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'Ne': 1.54,
    'Na': 2.27,
    'Mg': 1.73,
    'Al': 1.84,
    'Si': 2.10,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Ar': 1.88,
    'K': 2.75,
    'Ca': 2.31,
    'Ni': 1.63,
    'Cu': 1.40,
    'Zn': 1.39,
    'Ga': 1.87,
    'Ge': 2.11,
    'As': 1.85,
    'Se': 1.90,
    'Br': 1.85,
    'Kr': 2.02,
    'Rb': 3.03,
    'Sr': 2.49,
    'Pd': 1.63,
    'Ag': 1.72,
    'Cd': 1.58,
    'In': 1.93,
    'Sn': 2.17,
    'Sb': 2.06,
    'Te': 2.06,
    'I': 1.98,
    'Xe': 2.16,
    'Cs': 3.43,
    'Ba': 2.49,
    'Pt': 1.75,
    'Au': 1.66}


def get_maximal_valance(atom_label):
    """Get the maximum valance of an atom

    Arguments:
        atom_label (str): atom label e.g. C or Pd

    Returns:
        int: maximal valence of the atom
    """

    if atom_label in valid_valances.keys():
        return valid_valances[atom_label][-1]
    else:
        logger.warning(f'Could not find a valid valance for {atom_label}. Guessing at 6')
        return 6


def get_atomic_weight(atom_label):
    """Get the atomic weight of an atom

    Arguments:
        atom_label (str): atom label e.g. C or Pd

    Returns:
        int: atomic weight of the atom
    """

    if atom_label in atomic_weights.keys():
        return atomic_weights[atom_label]
    else:
        logger.warning(f'Could not find a valid weight for {atom_label}. Guessing at 70')
        return 70


def get_vdw_radii(atom_label):
    """Get the van der waal's radius of an atom

    Arguments:
        atom_label (str): atom label e.g. C or Pd

    Returns:
        float: van der waal's radius of the atom
    """
    if atom_label in vdw_radii.keys():
        vdv_radii = vdw_radii[atom_label]
    else:
        logger.error(f'Couldn\'t find the VdV radii for {atom_label}. Guessing at 1.5')
        vdv_radii = 1.5

    return vdv_radii
