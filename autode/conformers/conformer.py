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
        (autode.conformers.Conformer)
    """
    conformer = Conformer(name=name, atoms=species.atoms,
                          charge=species.charge, mult=species.mult)

    # Conformer should be in the same solvent
    conformer.solvent = species.solvent

    return conformer


class Conformer(Species):

    def optimise(self, method, reset_graph=False, calc=None):
        logger.info(f'Running optimisation of {self.name}')

        opt = Calculation(name=f'{self.name}_opt', molecule=self, method=method,
                          keywords=method.keywords.low_opt,
                          n_cores=Config.n_cores,
                          distance_constraints=self.dist_consts)
        opt.run()
        self.energy = opt.get_energy()

        try:
            self.set_atoms(atoms=opt.get_final_atoms())

        except AtomsNotFound:
            logger.error(f'Atoms not found for {self.name} but not critical')
            self.set_atoms(atoms=None)

        return None

    def __init__(self, name='conf', atoms=None, solvent_name=None, charge=0, mult=1, dist_consts=None):
        super(Conformer, self).__init__(name, atoms, charge, mult, solvent_name=solvent_name)

        self.dist_consts = dist_consts
