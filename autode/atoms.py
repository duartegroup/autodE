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
                  'Cl': 35.5,
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


def get_maximal_valance(atom_label):
    """
    Get the maximum valance of an atom
    :param atom_label: (str) atom label e.g. C or Pd
    :return: (int)
    """

    if atom_label in valid_valances.keys():
        return valid_valances[atom_label][-1]
    else:
        logger.warning(
            'Could not find a valid valance for {}. Guessing at 6'.format(atom_label))
        return 6


def get_atomic_weight(atom_label):
    """
    Get the atomic weight of an atom
    :param atom_label: (str) atom label e.g. C or Pd
    :return: (int)
    """

    if atom_label in atomic_weights.keys():
        return atomic_weights[atom_label]
    else:
        logger.warning(
            'Could not find a valid weight for {}. Guessing at 70'.format(atom_label))
        return 70
