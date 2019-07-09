from .log import logger
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca
from .ORCAio import get_orca_opt_xyzs_energy
from .ORCAio import get_orca_energy


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
