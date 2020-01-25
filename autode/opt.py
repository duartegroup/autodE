from copy import deepcopy
from autode.log import logger
from autode.config import Config
from autode.transition_states.ts_guess import TSguess
from autode.calculation import Calculation


def get_ts_guess_constrained_opt(mol, keywords, name, distance_consts, reaction_class, product):
    """Get a TS guess from a constrained optimisation with the active atoms fixed at values defined in distance_consts

    Arguments:
        mol (molecule object): molecule to opt
        keywords (list): keywords to use in the calc
        name (name): ts guess name
        distance_consts (dict): keys = tuple of atom ids for a bond to be kept at fixed length, value = length to be fixed at
        reaction_class (object): reaction type (reactions.py)
        product (molecule object): product complex

    Returns:
        ts guess object: ts guess
    """

    logger.info('Getting TS guess from constrained optimisation')

    opt_mol_with_const = deepcopy(mol)
    const_opt = Calculation(name=name + '_constrained_opt', molecule=opt_mol_with_const, method=mol.method,
                            keywords=keywords, n_cores=Config.n_cores, max_core_mb=Config.max_core,
                            distance_constraints=distance_consts)
    const_opt.run()
    opt_mol_with_const.xyzs = const_opt.get_final_xyzs()
    opt_mol_with_const.energy = const_opt.get_energy()

    return TSguess(name=name, reaction_class=reaction_class, molecule=opt_mol_with_const,
                   active_bonds=list(distance_consts.keys()), reactant=mol, product=product)
