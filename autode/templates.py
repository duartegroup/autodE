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
from .log import logger
from .mol_graphs import is_subgraph_isomorphic


def get_ts_templates(reaction_class):
    logger.info('Getting TS templates from {}/{}'.format('lib', reaction_class.__name__))

    folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', reaction_class.__name__)
    if os.path.exists(folder_name):
        return [pickle.load(open(os.path.join(folder_name, filename), 'rb')) for filename in os.listdir(folder_name)]
    return []


def template_matches(mol, ts_template):

    if mol.charge == ts_template.charge and mol.mult == ts_template.mult:
        if mol.solvent == ts_template.solvent:
            if is_subgraph_isomorphic(larger_graph=mol.graph, smaller_graph=ts_template.graph):
                logger.info('Found matching TS template')
                return True

    return False


class TStemplate(object):

    def save_object(self, basename='template'):

        name, i = basename + '0', 0
        folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', self.reaction_class.__name__)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        while True:
            if not os.path.exists(os.path.join(folder_name, name + '.obj')):
                break
            name = basename + str(i)
            i += 1

        with open(os.path.join(folder_name, name + '.obj'), 'wb') as pickled_file:
            pickle.dump(self, file=pickled_file)

    def __init__(self, graph, reaction_class, solvent=None, charge=0, mult=1):
        """
        Construct a TS template object
        :param graph: (object) networkx graph object. Active bonds in the TS are represented by the edges with
        attribute active=True, going out to nearest bonded neighbours
        :param reaction_class; (object) Addition/Dissociation/Elimination/Rearrangement/Substitution reaction
        :param solvent: (str)
        :param charge: (int)
        :param mult: (int)
        """

        self.graph = graph
        self.reaction_class = reaction_class
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
