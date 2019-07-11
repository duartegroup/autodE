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
        folder_name = self.reaction_class.__name__
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


if __name__ == '__main__':

    from autode import reactions

    tmp_ts_template = TStemplate(aaenv_dists=[{(ActiveAtomEnvironment('C', ['C', 'C', 'O']),
                                                ActiveAtomEnvironment('C', ['N'])): 2.061}],
                                 reaction_class=reactions.Addition,
                                 solvent='water',
                                 charge=-1)

    # tmp_ts_template.save_object()

    test_obj = pickle.load(open('Addition/template0.obj', 'rb'))
    print(test_obj.solvent)
