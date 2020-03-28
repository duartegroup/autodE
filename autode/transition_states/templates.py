"""
The idea with templating is to avoid needless PES scans when finding TSs for which similar have already been found.

For instance the TS for the addition of CN- to acetone is going to be perturbed only slightly by modifying a methyl
for a CH2CH3 group. It is more efficient to know the forming C–C bond distance in the previous TS, fix it and run
a constrained optimisation which will hopefully be a good guess of the TS
"""
import pickle
import os
from autode.log import logger
from autode.config import Config
from autode.mol_graphs import is_isomorphic


def get_ts_templates(folder_path=Config.ts_template_folder_path):
    """Get all the transition state templates from a folder, or the default if folder path is None

    Keyword Arguments:
        folder_path (str): /path/to/the/ts/template/library

    Returns:
        (list(autode.transition_states.templates.TStemplate))

    """

    if folder_path is None:
        logger.info('Folder path is not set – getting TS templates from the default path')
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')

    logger.info(f'Getting TS templates from {folder_path}')

    if not os.path.exists(folder_path):
        logger.error('Folder does not exist')
        return []

    obj_filenames = [filename for filename in os.listdir(folder_path) if filename.endswith('.obj')]
    objects = [pickle.load(open(os.path.join(folder_path, filename), 'rb')) for filename in obj_filenames]

    logger.info(f'Have {len(objects)} TS templates')
    return objects


def template_matches(reactant, truncated_graph, ts_template):
    """
    Determine if a transition state template matches a truncated graph. The truncated graph includes all the active
    bonds in the reaction and the nearest neighbours to those atoms e.g. for a Diels-Alder reaction
         H        H
          \      /
        H-C----C-H          where the dotted lines represent active bonds
          .    .
      H  .      .   H
       \.        . /
    H - C        C - H
         \      /
          C    C

    Arguments:
        reactant (autode.complex.ReactantComplex):
        truncated_graph (nx.Graph):
        ts_template (autode.transition_states.templates.TStemplate):
    """

    if reactant.charge != ts_template.charge or reactant.mult != ts_template.mult:
        return False

    if reactant.solvent != ts_template.solvent:
        return False

    if is_isomorphic(truncated_graph, ts_template.graph):
        logger.info('Found matching TS template')
        return True

    return False


class TStemplate:

    def save_object(self, basename='template', folder_path=None):
        """Save the TS template object"""

        if folder_path is None:
            logger.info('Saving TS template to default directory')
            folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')

        if not os.path.exists(folder_path):
            logger.info(f'Making directory {folder_path}')
            os.mkdir(folder_path)

        # Iterate i until the templatei.obj file doesn't exist
        name, i = basename + '0', 0
        while True:
            if not os.path.exists(os.path.join(folder_path, f'{name}.obj')):
                break
            name = basename + str(i)
            i += 1

        file_path = os.path.join(folder_path, f'{name}.obj')
        logger.info(f'Saving the template as {file_path}')

        with open(file_path, 'wb') as pickled_file:
            pickle.dump(self, file=pickled_file)

        return None

    def __init__(self, graph, charge, mult, solvent=None):
        """Construct a TS template object

        Arguments:
            graph (nx.Graph): Active bonds in the TS are represented by the edges with attribute active=True, going out
                              to nearest bonded neighbours

        Keyword Arguments:
            solvent (autode.solvent.solvents.Solvent):  (default: {None})
            charge (int):
            mult (int):
        """
        self.graph = graph
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
