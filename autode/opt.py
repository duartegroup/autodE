from copy import deepcopy
from autode.log import logger
from autode.config import Config
from autode.ts_guess import TSguess
from autode.calculation import Calculation
from autode.wrappers.ORCA import ORCA


def get_ts_guess_constrained_opt(mol, keywords, name, distance_consts, reaction_class):
    """
    Get a TS guess from a constrained optimisation with the active atoms fixed at values defined in distance_consts

    :param mol: (object)
    :param keywords: (list)
    :param name: (str)
    :param distance_consts: (dict) distance constraints to impose on the calculation keyed with tuples of atom ids
    and values of required constrained value
    :param reaction_class: (object)

    :return: TSguess object
    """

    logger.info('Getting TS guess from ORCA relaxed potential energy scan')

    opt_mol_with_const = deepcopy(mol)
    const_opt = Calculation(name=mol.name + '_orca_constrained_opt', molecule=opt_mol_with_const, method=mol.method,
                            keywords=keywords, n_cores=Config.n_cores, max_core_mb=Config.max_core,
                            distance_constraints=distance_consts)
    const_opt.run()

    return TSguess(name=name, reaction_class=reaction_class, molecule=opt_mol_with_const,
                   active_bonds=list(distance_consts.keys()))
