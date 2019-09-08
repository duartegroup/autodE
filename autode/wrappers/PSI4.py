import numpy as np
from copy import deepcopy
from autode.log import logger
from autode.config import Config
from autode.constants import Constants
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods

pcmsolver_solvents = ['water', 'propylene carbonate', 'dimethylsulfoxide', 'nitromethane', 'acetonitrile',
                      'acetonitrile', 'methanol', 'ethanol', 'Acetone', '1,2-dichloroethane', 'methylenechloride',
                      'tetrahydrofurane', 'aniline', 'chlorobenzene', 'chloroform', 'toluene', '1,4-dioxane',
                      'benzene', 'carbon tetrachloride', 'cyclohexane', 'n-heptane']

PSI4 = ElectronicStructureMethod(name='psi4', path=Config.PSI4.path,
                                 aval_solvents=pcmsolver_solvents,
                                 scan_keywords=Config.PSI4.scan_keywords,
                                 conf_opt_keywords=Config.PSI4.conf_opt_keywords,
                                 opt_keywords=Config.PSI4.opt_keywords,
                                 opt_ts_keywords=Config.PSI4.opt_ts_keywords,
                                 hess_keywords=Config.PSI4.hess_keywords,
                                 opt_ts_block=Config.PSI4.opt_ts_block,
                                 sp_keywords=Config.PSI4.sp_keywords)


def generate_input(calc):
    calc.input_filename = calc.name + '_psi4.in'
    calc.output_filename = calc.name + '_psi4.out'
    calc.flags = ['-n', str(calc.n_cores)]

    with open(calc.input_filename, 'w') as in_file:
        print('memory', Config.max_core, ' mb', file=in_file)

        # Python cannot handel things with pluses or minuses in so remove them
        name = calc.name
        if '-' in name:
            name = name.replace('-', '')
        if '+' in name:
            name = name.replace('+', '')

        print('molecule', name, '{', file=in_file)
        print(calc.charge, calc.mult, file=in_file)
        [print('{:<3}{:^12.8f}{:^12.8f}{:^12.8f}'.format(*line), file=in_file) for line in calc.xyzs]
        print('}', file=in_file)

        # Can't do an optimisation of a single atom to alter the keywords to only evaluate a SP
        eval_keywords = []
        if calc.n_atoms == 1 and calc.opt is True:
            for keyword_line in calc.keywords:
                if 'optimize' in keyword_line:
                    eval_keywords.append(keyword_line.replace('optimize', 'energy'))
                else:
                    eval_keywords.append(keyword_line)
        else:
            eval_keywords = deepcopy(calc.keywords)

        # Add the block appropriate for an implicit solvent model using UFF radii
        if calc.solvent is not None:
            if calc.optts_block is not None:
                logger.warning('PSI4 doesn\'t have analytical second derivatives for PCM. This will be slow')

            eval_keywords.insert(0, 'set pcm true\nset pcm_scf_type total')
            print('pcm = {\n'
                  '    Units = Angstrom\n'
                  '    Medium {\n'
                  '    SolverType = IEFPCM\n'
                  '    Solvent =', calc.solvent,
                  '    }\n    Cavity{\n'
                  '    RadiiSet = UFF\n'
                  '    Type = GePol\n'
                  '    Scaling = False\n'
                  '    Area = 0.3\n'
                  '    Mode = Implicit\n'
                  '    }\n'
                  '}', file=in_file)

        if calc.optts_block:
            print(calc.optts_block, file=in_file)

        if calc.bond_ids_to_add:
            # PSI4 seeming doesn't allow for specific internal coordinates to be added..
            print('set add_auxiliary_bonds true', file=in_file)

        if calc.distance_constraints:
            print('set optking {'
                  '\n    fixed_distance = ("', file=in_file)
            for bond_ids in calc.distance_constraints.keys():
                print('      ', bond_ids[0], ' ', bond_ids[1], np.round(calc.distance_constraints[bond_ids], 3),
                      file=in_file)
            print('     ")\n}', file=in_file)

        if calc.mult != 1:
            print('set reference uks', file=in_file)
        print(*eval_keywords, sep='\n', file=in_file)

    return None


def calculation_terminated_normally(calc):

    for n_line, line in enumerate(calc.rev_output_file_lines):
        if '*** Psi4 exiting successfully. Buy a developer a beer!' in line:
            return True
        if 'Optimization has failed!' in line:
            return True
        if n_line > 150:
            return False

    return False


def get_energy(calc):
    for line in calc.rev_output_file_lines:
        if 'Total Energy =' in line:
            return float(line.split()[3])


def optimisation_converged(calc):
    for line in calc.rev_output_file_lines:
        if 'Optimization is completeOptimization is complete' in line:
            return True
    return False


def optimisation_nearly_converged(calc):

    opt_didnt_converge, opt_summary_block = False, False
    for n_line, line in enumerate(calc.output_file_lines):
        if 'Optimization has failed!' in line:
            opt_didnt_converge = True
        if opt_didnt_converge and 'Optimization Summary' in line:
            opt_summary_block = True
        if opt_summary_block:
            # look for the end of the block that looks like this
            # 50    -157.024501806968   0.001593233313   0.22347783   0.05638238   0.04367953   0.01250046  ~
            #   ------------------------------------------------------------------------------------------- ~
            if len(line.split()) == 2 and calc.output_file_lines[n_line-1].split()[-2][-1].isdigit():
                if calc.output_file_lines[n_line-1].split()[2]*Constants.ha2kcalmol < 1:
                    return True
                else:
                    return False

    return False


def get_imag_freqs(calc):
    raise NotImplementedError


def get_normal_mode_displacements(calc, mode_number):
    raise NotImplementedError


def get_final_xyzs(calc):
    if calc.n_atoms == 1:
        return calc.xyzs

    opt_done, xyzs = False, []
    xyz_lines = None
    for n_line, line in enumerate(calc.output_file_lines):
        if 'OPTKING Finished Execution' in line:
            opt_done = True
        if opt_done and 'Geometry (in Angstrom)' in line:
            xyz_lines = calc.output_file_lines[n_line+2:n_line+calc.n_atoms+2]
            opt_done = False

    if xyz_lines is None:
        return []

    for line in xyz_lines:
        atom_label, x, y, z = line.split()
        xyzs.append([atom_label.title(), float(x), float(y), float(z)])

    return xyzs


# Bind all the required functions to the class definition
[setattr(PSI4, method, globals()[method]) for method in req_methods]
