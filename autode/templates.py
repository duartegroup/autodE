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


def get_ts_templates(reaction_class):
    logger.info('Getting TS templates from {}/{}'.format('lib', reaction_class.__name__))

    folder_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib', reaction_class.__name__)
    if os.path.exists(folder_name):
        return [pickle.load(open(file_name, 'rb')) for file_name in os.listdir(folder_name)]
    return []


def get_matching_ts_template(mol, mol_aaenvs, ts_guess_templates):

    print(ts_guess_templates)
    print(ts_guess_templates[0].solvent)

    for ts_guess_template in ts_guess_templates:
        print('here1')

        if ts_guess_template.solvent == mol.solvent:
            print('here2')

            if ts_guess_template.charge == mol.charge:
                print('here3')

                if ts_guess_template.mult == mol.mult:
                    print('here4')
                    if ts_guess_template.aaenv_dists.keys() == mol_aaenvs:
                        logger.info('Found TS template')
                        return ts_guess_template
    return None


class ActiveAtomEnvironment(object):

    def __init__(self, atom_label, bonded_atom_labels):
        """
        Generate an atom surrounded by it's environment (bonded)
        :param atom_label: (str)
        :param bonded_atom_labels: (list(str))
        """

        self.atom_label = atom_label
        self.bonded_atom_labels = bonded_atom_labels


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

    def __init__(self, aaenv_dists, reaction_class, solvent=None, charge=0, mult=1):
        """
        Construct a TS template object
        :param aaenv_dists: list(dict) List of dictionaries keyed with ActiveAtomEnvironment object pairs with the value
        of the distance found previously at the TS
        :param reaction_class; (object) Addition/Dissociation/Elimination/Rearrangement/Substitution reaction
        :param solvent: (str)
        :param charge: (int)
        :param mult: (int)
        """

        self.aaenv_dists = aaenv_dists
        self.reaction_class = reaction_class
        self.solvent = solvent
        self.charge = charge
        self.mult = mult
