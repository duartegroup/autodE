from autode.exceptions import AtomsNotFound
from autode.log import logger
from autode.species.species import Species


def get_conformer(species, name):
    """
    Generate a conformer of a species given a new name but with the same
    atoms

    Arguments:
        species (autode.species.Species):
        name (str):

    Returns:
        (autode.conformers.Conformer): Conformer
    """
    conformer = Conformer(name=name, atoms=species.atoms,
                          charge=species.charge, mult=species.mult)

    # Conformer should be in the same solvent
    conformer.solvent = species.solvent

    return conformer


class Conformer(Species):

    def __repr__(self):
        return self._repr(prefix='Conformer')

    def single_point(self, method, keywords=None, n_cores=None):
        """
        Calculate a single point and default to a low level single point method

        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            keywords (autode.wrappers.keywords.Keywords):

            n_cores (int | None):
        """
        keywords = method.keywords.low_sp if keywords is None else keywords
        return super().single_point(method, keywords, n_cores=n_cores)

    def optimise(self, method=None, reset_graph=False, calc=None, keywords=None,
                 n_cores=None):
        """
        Optimise the geometry of this conformer

        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            reset_graph (bool):
            calc (autode.calculation.Calculation):
            keywords (autode.wrappers.keywords.Keywords):
            n_cores (int | None):
        """
        try:
            if keywords is None and method is not None:
                keywords = method.keywords.low_opt

            super().optimise(method,
                             keywords=keywords, calc=calc, n_cores=n_cores)

        except AtomsNotFound:
            logger.error(f'Atoms not found for {self.name} but not critical')
            self.atoms = None

        return None

    def __init__(self, name='conf', atoms=None, solvent_name=None,
                 charge=0, mult=1, dist_consts=None):
        super().__init__(name, atoms, charge, mult, solvent_name=solvent_name)

        self.dist_consts = dist_consts
