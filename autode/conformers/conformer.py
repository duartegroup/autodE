from autode.calculation import Calculation
from autode.config import Config
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

    def single_point(self, method, keywords=None):
        """
        Calculate a single point and default to a low level single point method

        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            keywords (autode.wrappers.keywords.Keywords):
        """
        keywords = method.keywords.low_sp if keywords is None else keywords
        return super().single_point(method, keywords)

    def optimise(self, method=None, reset_graph=False, calc=None, keywords=None):
        """
        Optimise the geometry of this conformer

        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            reset_graph (bool):
            calc (autode.calculation.Calculation):
            keywords (autode.wrappers.keywords.Keywords):
        """
        logger.info(f'Running optimisation of {self.name}')

        if calc is not None or reset_graph:
            raise NotImplementedError

        opt = Calculation(name=f'{self.name}_opt', molecule=self, method=method,
                          keywords=method.keywords.low_opt,
                          n_cores=Config.n_cores,
                          distance_constraints=self.dist_consts)
        opt.run()
        self.energy = opt.get_energy()

        try:
            self.atoms = opt.get_final_atoms()

        except AtomsNotFound:
            logger.error(f'Atoms not found for {self.name} but not critical')
            self.atoms = None

        return None

    def __init__(self, name='conf', atoms=None, solvent_name=None,
                 charge=0, mult=1, dist_consts=None):
        super(Conformer, self).__init__(name, atoms, charge, mult,
                                        solvent_name=solvent_name)

        self.dist_consts = dist_consts
