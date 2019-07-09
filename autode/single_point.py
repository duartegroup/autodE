from .log import logger
from .ORCAio import gen_orca_inp
from .ORCAio import run_orca
from .ORCAio import get_orca_energy


def get_single_point_energy(mol, keywords, n_cores):
    logger.info('Calculating single point energy for {}'.format(mol.name))
    inp_filename = mol.name + '_orca_sp.inp'
    gen_orca_inp(inp_filename, keywords, mol.xyzs, mol.charge, mol.mult, mol.solvent, n_cores)
    orca_output = run_orca(inp_filename, out_filename=inp_filename.replace('.inp', '.out'))

    return get_orca_energy(orca_output)
