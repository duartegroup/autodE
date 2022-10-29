class ScannedBond:
    def __str__(self):
        i, j = self.atom_indexes
        return f"{i}-{j}"

    def __getitem__(self, item):
        return self.atom_indexes[item]

    @property
    def dr(self):
        """Change in distance for this bond (∆r / Å)"""
        if self.curr_dist is None or self.final_dist is None:
            return 0

        return self.final_dist - self.curr_dist

    def __init__(self, atom_indexes):
        """
        Bond with a current and final distance which will be scanned over

        -----------------------------------------------------------------------
        Arguments:
            atom_indexes (tuple(int)): Atom indexes that make this
            'bond' e.g. (0, 1)
        """
        assert len(atom_indexes) == 2

        self.atom_indexes = atom_indexes

        self.curr_dist = None
        self.final_dist = None

        self.forming = False
        self.breaking = False


class FormingBond(ScannedBond):
    def __init__(self, atom_indexes, species, final_species=None):
        """
        Forming bond with current and final distances

        -----------------------------------------------------------------------
        Arguments:
            atom_indexes (tuple(int)):

            species (autode.species.Species):
        """
        super().__init__(atom_indexes)
        self.forming = True

        i, j = self.atom_indexes
        self.curr_dist = species.distance(i=i, j=j)

        if final_species is None:
            self.final_dist = species.atoms.eqm_bond_distance(i, j)
        else:
            self.final_dist = final_species.distance(*atom_indexes)


class BreakingBond(ScannedBond):
    def __init__(self, atom_indexes, species, final_species=None):
        """
        Form a breaking bond with current and final distances

        -----------------------------------------------------------------------
        Arguments:
            atom_indexes (tuple(int)):

            species (autode.species.Species):

            final_species (autode.species.Species | None):
        """
        super().__init__(atom_indexes)
        self.breaking = True

        self.curr_dist = species.distance(*self.atom_indexes)

        if final_species is None:
            self.final_dist = 2.0 * self.curr_dist

        else:
            # Take the smallest possible final distance, thus the shortest
            # path to traverse
            self.final_dist = min(
                final_species.distance(*self.atom_indexes),
                2.0 * self.curr_dist,
            )
