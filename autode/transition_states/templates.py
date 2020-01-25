"""
The idea with templating is to avoid needless PES scans when finding TSs for which similar have already been found.

For instance the TS for the addition of CN- to acetone is going to be perturbed only slightly by modifying a methyl
for a CH2F group. There it is more efficient to know the forming C–C bond distance in the previous TS, fix it and run
a constrained optimisation which will hopefully be a good guess of the TS

--------------------------------------------------------------------------------------------------------------------

In this file we have the classes which will be saved and then loaded at runtime. The idea being that lib/ gets populated
with TS templates which can be searched through..
"""
import pickle
import os
from autode.log import logger
from autode.mol_graphs import is_subgraph_isomorphic


def get_ts_templates(reaction_class, folder_path=None):
    logger.info(f'Getting TS templates from lib/{reaction_class.__name__}')

    if folder_path is None:
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', reaction_class.__name__)

    if os.path.exists(folder_path):
        obj_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.obj')]
        return [pickle.load(open(os.path.join(folder_path, filename), 'rb')) for filename in obj_files]
    return []


def template_matches(mol, ts_template, mol_graph):

    if mol.charge == ts_template.charge and mol.mult == ts_template.mult:
        if mol.solvent == ts_template.solvent:
            if is_subgraph_isomorphic(larger_graph=mol_graph, smaller_graph=ts_template.graph):
                logger.info('Found matching TS template')
                return True

    return False


class TStemplate:

    def save_object(self, basename='template', folder_path=None):

        if folder_path is not None:
            logger.info(f'Saving template to user-defined directory: {folder_path}')

            if not os.path.exists(folder_path):
                logger.critical('Cannot save TS templates to a directory that doesn\'t exist')
                exit()

        # If the folder_path is None then save to the default location
        else:
            base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
            if not os.path.exists(base_path):
                logger.info(f'Making base directory {base_path}')
                os.mkdir(base_path)

            folder_path = os.path.join(base_path, self.reaction_class.__name__)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

        # Iterate i until the i.obj file doesn't exist
        name, i = basename + '0', 0
        while True:
            if not os.path.exists(os.path.join(folder_path, name + '.obj')):
                break
            name = basename + str(i)
            i += 1

        file_path = os.path.join(folder_path, name + '.obj')
        logger.info(f'Saving the template as {file_path}')
        with open(file_path, 'wb') as pickled_file:
            pickle.dump(self, file=pickled_file)

        return None

    def __init__(self, graph, reaction_class, solvent=None, charge=0, mult=1):
        """Construct a TS template object

        Arguments:
            graph (nx.Graph): Active bonds in the TS are represented by the edges with attribute active=True, going out to nearest bonded neighbours
            reaction_class (object): Reaction class (reactions.py)

        Keyword Arguments:
            solvent (str): solvent (default: {None})
            charge (int): charge (default: {0})
            mult (int): multiplicity of the molecule (default: {1})
        """

        self.graph = graph
        self.reaction_class = reaction_class
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
