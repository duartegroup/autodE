from autode.species import Species
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.calculation import Calculation
from autode.exceptions import AtomsNotFound
from autode.config import Config


class Conformer(Species):

    def optimise(self, method):
        logger.info(f'Running optimisation of {self.name}')

        opt = Calculation(name=f'{self.name}_opt', molecule=self, method=method, keywords_list=method.keywords.low_opt,
                          n_cores=Config.n_cores, opt=True, distance_constraints=self.dist_consts)
        opt.run()
        self.energy = opt.get_energy()

        try:
            self.set_atoms(atoms=opt.get_final_atoms())

        except AtomsNotFound:
            logger.error(f'Atoms not found for {self.name} but not critical')
            self.set_atoms(atoms=None)

    def __init__(self, name='conf', atoms=None, solvent_name=None, charge=0, mult=1, dist_consts=None):
        super(Conformer, self).__init__(name, atoms, charge, mult, solvent_name=solvent_name)

        self.dist_consts = dist_consts
