from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.geom import get_shifted_xyzs_linear_interp
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import req_methods


solvents = ['acetic acid', 'acetone', 'acetonitrile', 'benzene', '1-butanol', '2-butanone', 'methyl ethyl ketone',
            'carbon tetrachloride', 'chlorobenzene', 'chloroform', 'cyclohexane', '1,2-dichlorobenzene',
            'dichloromethane', 'n,n-dimethylacetamide', 'n,n-dimethylformamide', 'dmp', '1,4-dioxane', 'ether',
            'ethyl acetate', 'ethyl alcohol', 'heptane', 'hexane', 'pentane', '1-propanol', 'pyridine',
            'tetrahydrofuran', 'thf', 'toluene', 'water']

# Dielectric from Physical Properties of Solvents from SIGMA-ALDRICH
solvents_and_dielectrics = {"acetic acid": 6.15, "acetone": 20.7, "acetonitrile": 37.5, "benzene": 2.28,
                            "1-butanol": 17.8, "2-butanone": 18.5, "methyl ethyl ketone": 18.5,
                            "carbon tetrachloride": 2.24, "chlorobenzene": 2.71, "chloroform": 4.81,
                            "cyclohexane": 2.02, "1,2-dichlorobenzene": 9.93, "dichloromethane": 9.08,
                            "n,n-dimethylacetamide": 37.8, "n,n-dimethylformamide": 36.7, "dmp": 36.7,
                            "1,4-dioxane": 2.21, "ether": 4.34, "ethyl acetate": 6.02, "ethyl alcohol": 24.55,
                            "heptane": 1.92, "hexane": 1.89, "pentane": 1.84, "1-propanol": 20.1, "pyridine": 12.3,
                            "tetrahydrofuran": 7.6, "thf": 7.6, "toluene": 2.4, "water": 78.54}

MOPAC = ElectronicStructureMethod(name='mopac',
                                  path=Config.MOPAC.path,
                                  req_licence=True,
                                  path_to_licence=Config.MOPAC.licence_path,
                                  aval_solvents=solvents)


def generate_input(calc):
    logger.info(f'Generating MOPAC input for {calc.name}')

    calc.input_filename = calc.name + '_mopac.mop'
    calc.output_filename = calc.input_filename.replace('.mop', '.out')

    keywords = Config.MOPAC.keywords.copy()

    if not calc.opt:
        keywords.append('1SCF')

    if calc.solvent is not None:
        keywords.append('EPS=' + str(solvents_and_dielectrics[calc.solvent]))

    keywords.append(f'CHARGE={calc.charge}')

    if calc.mult != 1:
        if calc.mult == 2:
            keywords.append('DOUBLET')
        elif calc.mult == 3:
            keywords.append('TRIPLET')
        elif calc.mult == 4:
            keywords.append('QUARTET')
        else:
            logger.critical('Unsupported spin multiplicity')
            exit()

    with open(calc.input_filename, 'w') as input_file:
        print(*keywords, '\n\n', file=input_file)

        if calc.distance_constraints is not None:
            # MOPAC seemingly doesn't have the capability to defined constrained bond lengths, so perform a linear
            # interpolation to the xyzs then fix the Cartesians

            xyzs = get_shifted_xyzs_linear_interp(xyzs=calc.xyzs,
                                                  bonds=list(calc.distance_constraints.keys()),
                                                  final_distances=list(calc.distance_constraints.values()))

            # Populate a flat list of atom ids to fix
            fixed_atoms = [i for bond in calc.distance_constraints.keys() for i in bond]

        else:
            xyzs = calc.xyzs
            fixed_atoms = []

        if calc.cartesian_constraints is not None:
            fixed_atoms += calc.cartesian_constraints

        for i, xyz_line in enumerate(xyzs):
            if i in fixed_atoms:
                print('{:<3}{:^10.5f} 0 {:^10.5f} 0 {:^10.5f} 0'.format(*xyz_line), file=input_file)
            else:
                print('{:<3}{:^10.5f} 1 {:^10.5f} 1 {:^10.5f} 1'.format(*xyz_line), file=input_file)

    return None


def calculation_terminated_normally(calc):

    for n_line, line in enumerate(calc.rev_output_file_lines):
        if 'JOB ENDED NORMALLY' in line:
            return True
        if n_line > 50:
            # MOPAC will have a     * JOB ENDED NORMALLY *  line close to the end if terminated normally
            return False

    return False


def get_energy(calc):
    for n_line, line in enumerate(calc.output_file_lines):
        if 'TOTAL ENERGY' in line:
            # e.g.     TOTAL ENERGY            =       -476.93072 EV
            return Constants.eV2ha * float(line.split()[-2])

    return None


def optimisation_converged(calc):

    for line in calc.rev_output_file_lines:
        if 'GRADIENT' in line and 'IS LESS THAN CUTOFF' in line:
            return True

    return False


def optimisation_nearly_converged(calc):
    raise NotImplementedError


def get_imag_freqs(calc):
    raise NotImplementedError


def get_normal_mode_displacements(calc, mode_number):
    raise NotImplementedError


def get_final_xyzs(calc):

    xyzs = []

    for n_line, line in enumerate(calc.output_file_lines):
        if 'CARTESIAN COORDINATES' in line and len(calc.output_file_lines[n_line+3].split()) == 5:
            #                              CARTESIAN COORDINATES
            #
            #    1    C        1.255660629     0.020580974    -0.276235553

            xyzs = []
            xyz_lines = calc.output_file_lines[n_line+2:n_line+2+calc.n_atoms]
            for xyz_line in xyz_lines:
                atom_label, x, y, z = xyz_line.split()[1:]
                xyzs.append([atom_label, float(x), float(y), float(z)])

    return xyzs


# Bind all the required functions to the class definition
[setattr(MOPAC, method, globals()[method]) for method in req_methods]
