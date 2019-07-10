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
        while True:
            if not os.path.exists(name + '.obj'):
                break
            name = basename + str(i)
            i += 1

        with open(name + '.obj', 'wb') as pickled_file:
            pickle.dump(self, file=pickled_file)

    def __init__(self, aaenv_dists, solvent=None, charge=0, mult=1):
        """
        Construct a TS template object
        :param aaenv_dists: list(dict) List of dictionaries keyed with ActiveAtomEnvironment object pairs with the value
        of the distance found previously at the TS
        :param solvent: (str)
        :param charge: (int)
        :param mult: (int)
        """

        self.aaenv_dists = aaenv_dists
        self.solvent = solvent
        self.charge = charge
        self.mult = mult


if __name__ == '__main__':

    tmp_ts_template = TStemplate(aaenv_dists=[{(ActiveAtomEnvironment('C', ['C', 'C', 'O']),
                                                ActiveAtomEnvironment('C', ['N'])): 2.061}],
                                 solvent='water',
                                 charge=-1)

    tmp_ts_template.save_object()
