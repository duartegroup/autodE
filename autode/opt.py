from .log import logger
from .config import Config
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca
from .ORCAio import get_orca_opt_xyzs_energy
from .ORCAio import get_orca_energy
from .ORCAio import get_orca_opt_final_xyzs
from .ts_guess import TSguess


def get_orca_ts_guess_constrained_opt(mol, orca_keywords, name, distance_constraints, reaction_class):
    """

    :return: TSguess object
    """

    logger.info('Getting TS guess from ORCA relaxed potential energy scan')

    inp_filename = name + '_orca_constrained_opt.inp'
    gen_orca_inp(inp_filename, orca_keywords, mol.xyzs, mol.charge, mol.mult, mol.solvent, Config.n_cores,
                 distance_constraints=distance_constraints)

    orca_out_lines = run_orca(inp_filename, out_filename=inp_filename.replace('.inp', '.out'))
    ts_guess_xyzs = get_orca_opt_final_xyzs(orca_out_lines)

    return TSguess(name=name, reaction_class=reaction_class, xyzs=ts_guess_xyzs, solvent=mol.solvent,
                   charge=mol.charge, mult=mol.mult, active_bonds=list(distance_constraints.keys()))


def get_opt_xyzs_energy(mol, keywords, n_cores):
    """
    Optimise a Molecule using ORCA, using the optimisation level specified in config.py
    :param mol: Molecule object
    :param keywords: (list) ORCA keywords to use in the optimisation
    :param n_cores: (int) number of cores to perform the optimsiation with
    :return:
    """
    logger.info('Optimising {} with ORCA'.format(mol.name))

    inp_filename = mol.name + '_orca_opt.inp'
    gen_orca_inp(inp_filename, keywords, mol.xyzs, mol.charge, mol.mult, mol.solvent, n_cores)
    orca_output = run_orca(inp_filename, out_filename=inp_filename.replace('.inp', '.out'))
    if mol.n_atoms == 1:
        return mol.xyzs, get_orca_energy(orca_output)
    else:
        return get_orca_opt_xyzs_energy(orca_output)
