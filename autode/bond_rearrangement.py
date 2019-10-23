from autode.geom import get_neighbour_list


class BondRearrangement(object):

    def get_active_atom_neighbour_lists(self, mol, depth):

        if self.active_atom_nl is None:
            self.active_atom_nl = [get_neighbour_list(atom_i=atom, mol=mol)[
                :depth] for atom in self.active_atoms]

        return self.active_atom_nl

    def __eq__(self, other):
        return self.fbonds == other.fbonds and self.bbonds == other.bbonds

    def __init__(self, forming_bonds=None, breaking_bonds=None):

        self.fbonds = forming_bonds if forming_bonds is not None else []
        self.bbonds = breaking_bonds if breaking_bonds is not None else []

        self.n_fbonds = len(self.fbonds)
        self.n_bbonds = len(self.bbonds)

        self.all = self.fbonds + self.bbonds
        self.active_atoms = [atom_id for bond in self.all for atom_id in bond]
        self.active_atom_nl = None
