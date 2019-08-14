from .log import logger


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


def get_maximal_valance(atom_label):
    """
    Get the maximum valance of an atom
    :param atom_label: (str) atom label e.g. C or Pd
    :return: (int)
    """

    if atom_label in valid_valances.keys():
        return valid_valances[atom_label][-1]
    else:
        logger.warning('Could not find a valid valance for {}'.format(atom_label))
        return 6
