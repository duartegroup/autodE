from autode.config import Config
from autode.log import logger
from autode.constants import Constants
from autode.geom import get_shifted_xyzs_linear_interp
from autode.wrappers.base import ElectronicStructureMethod
from autode.exceptions import UnsuppportedCalculationInput
import numpy as np


# dielectrics from Gaussian solvent_name list
solvents_and_dielectrics = {'acetic acid': 6.25, 'acetone': 20.49, 'acetonitrile': 35.69, 'benzene': 2.27, '1-butanol': 17.33,
                            '2-butanone': 18.25, 'carbon tetrachloride': 2.23, 'chlorobenzene': 5.70, 'chloroform': 4.71,
                            'cyclohexane': 2.02, '1,2-dichlorobenzene': 9.99, 'dichloromethane': 8.93, 'n,n-dimethylacetamide': 37.78,
                            'n,n-dimethylformamide': 37.22, '1,4-dioxane': 2.21, 'ether': 4.24, 'ethyl acetate': 5.99, 'tce': 3.42,
                            'ethyl alcohol': 24.85, 'heptane': 1.91, 'hexane': 1.88, 'pentane': 1.84, '1-propanol': 20.52, 'pyridine': 12.98,
                            'tetrahydrofuran': 7.43, 'toluene': 2.37, 'water': 78.36, 'cs2': 2.61, 'dmso': 46.82, 'methanol': 32.61,
                            '2-butanol': 15.94, 'acetophenone': 17.44, 'aniline': 6.89, 'anisole': 4.22, 'benzaldehyde': 18.22,
                            'benzonitrile': 25.59, 'benzyl chloride': 6.72, 'isobutyl bromide': 7.78, 'bromobenzene': 5.40, 'bromoethane': 9.01,
                            'bromoform': 4.25, 'bromooctane': 5.02, 'bromopentane': 6.27, 'butanal': 13.45, '1,1,1-trichloroethane': 7.08,
                            'cyclopentane': 1.96, '1,1,2-trichloroethane': 7.19, 'cyclopentanol': 16.99, '1,2,4-trimethylbenzene': 2.37,
                            'cyclopentanone': 13.58, '1,2-dibromoethane': 4.93, '1,2-dichloroethane': 10.13, 'cis-decalin': 2.21,
                            'trans-decalin': 2.18, 'decalin': 2.20, '1,2-ethanediol': 40.25, 'decane': 1.98, 'dibromomethane': 7.23,
                            'dibutylether': 3.05, 'z-1,2-dichloroethene': 9.20, 'e-1,2-dichloroethene': 2.14, '1-bromopropane': 8.05,
                            '2-bromopropane': 9.36, '1-chlorohexane': 5.95, '1-ChloroPentane': 6.50, '1-chloropropane': 8.35,
                            'diethylamine': 3.58, 'decanol': 7.53, 'diiodomethane': 5.32, '1-fluorooctane': 3.89, 'heptanol': 11.32,
                            'cisdmchx': 2.06, 'diethyl sulfide': 5.73, 'diisopropyl ether': 3.38, 'hexanol': 12.51, 'hexene': 2.07,
                            'hexyne': 2.62, 'iodobutane': 6.17, '1-iodohexadecane': 3.53, 'diphenylether': 3.73, '1-iodopentane': 5.70,
                            '1-iodopropane': 6.96, 'dipropylamine': 2.91, 'dodecane': 2.01, '1-nitropropane': 23.73, 'ethanethiol': 6.67,
                            'nonanol': 8.60, 'octanol': 9.86, 'pentanol': 15.13, 'pentene': 1.99, 'ethylbenzene': 2.43, 'tbp': 8.18,
                            '2,2,2-trifluoroethanol': 26.73, 'fluorobenzene': 5.42, '2,2,4-trimethylpentane': 1.94, 'formamide': 108.94,
                            '2,4-dimethylpentane': 1.89, '2,4-dimethylpyridine': 9.41, '2,6-dimethylpyridine': 7.17, 'hexadecane': 2.04,
                            'dimethyl disulfide': 9.60, 'ethyl methanoate': 8.33, 'phentole': 4.18, 'formic acid': 51.1, 'hexanoic acid': 2.6,
                            '2-chlorobutane': 8.39, '2-heptanone': 11.66, '2-hexanone': 14.14, '2-methoxyethanol': 17.20, 'isobutanol': 16.78,
                            'tertbutanol': 12.47, '2-methylpentane': 1.89, '2-methylpyridine': 9.95, '2-nitropropane': 25.65, '2-octanone': 9.47,
                            '2-pentanone': 15.20, 'iodobenzene': 4.55, 'iodoethane': 7.62, 'iodomethane': 6.87, 'isopropylbenzene': 2.37,
                            'p-cymene': 2.23, 'mesitylene': 2.27, 'methyl benzoate': 6.74, 'methyl butanoate': 5.56, 'methyl acetate': 6.86,
                            'methyl formate': 8.84, 'methyl propanoate': 6.08, 'n-methylaniline': 5.96, 'methylcyclohexane': 2.02, 'nmfmixtr': 181.56,
                            'nitrobenzene': 34.81, 'nitroethane': 28.29, 'nitromethane': 36.56, 'o-nitrotoluene': 25.67, 'n-nonane': 1.96,
                            'n-octane': 1.94, 'n-pentadecane': 2.03, 'pentanal': 10.00, 'pentanoic acid': 2.69, 'pentyl acetate': 4.73,
                            'pentylamine': 4.20, 'perfluorobenzene': 2.03, 'propanal': 18.50, 'propanoic acid': 3.44, 'cyanoethane': 29.32,
                            'propyl acetate': 5.52, 'propylamine': 4.99, 'tetrachloroethene': 2.27, 'sulfolane': 43.96, 'tetralin': 2.77,
                            'thiophene': 2.73, 'thiophenol': 4.27, 'triethylamine': 2.38, 'n-undecane': 1.99, 'xylene mix': 3.29, 'm-xylene': 2.35,
                            'o-xylene': 2.55, 'p-xylene': 2.27, '2-propanol': 19.26, '2-propen-1-ol': 19.01, 'e-2-pentene': 2.05,
                            '3-methylpyridine': 11.65, '3-pentanone': 16.78, '4-heptanone': 12.26, 'mibk': 12.88, '4-methylpyridine': 11.96,
                            '5-nonanone': 10.6, 'benzyl alcohol': 12.46, 'butanoic acid': 2.99, 'butanenitrile': 24.29, 'butyl acetate': 4.99,
                            'butylamine': 4.62, 'n-butylbenzene': 2.36, 's-butylbenzene': 2.34, 't-butylbenzene': 2.34, 'o-chlorotoluene': 4.63,
                            'm-cresol': 12.44, 'o-cresol': 6.76, 'cyclohexanone': 15.62, 'isoquinoline': 11.00, 'quinoline': 9.16, 'argon': 1.43,
                            'krypton': 1.52, 'xenon': 1.70}


class MOPAC(ElectronicStructureMethod):

    def generate_input(self, calc):
        logger.info(f'Generating mopac input for {calc.name}')

        calc.input_filename = calc.name + '_mopac.mop'
        calc.output_filename = calc.input_filename.replace('.mop', '.out')

        keywords = Config.MOPAC.keywords.copy()

        if not calc.opt:
            keywords.append('1SCF')

        if calc.grad:
            keywords.append('GRAD')

        if calc.charges:
            keywords.append('QMMM')

        if calc.solvent_keyword is not None:
            keywords.append('EPS=' + str(solvents_and_dielectrics[calc.solvent_keyword]))

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
                raise UnsuppportedCalculationInput

        with open(calc.input_filename, 'w') as input_file:
            print(*keywords, '\n\n', file=input_file)

            if calc.distance_constraints is not None:
                # mopac seemingly doesn't have the capability to defined constrained bond lengths, so perform a linear
                # interpolation to the xyzs then fix the Cartesians

                xyzs = get_shifted_xyzs_linear_interp(xyzs=calc.xyzs,
                                                      bonds=list(calc.distance_constraints.keys()),
                                                      final_distances=list(calc.distance_constraints.values()))

                # Populate a flat list of atom ids to fix
                fixed_atoms = [i for bond in calc.distance_constraints.keys()
                               for i in bond]

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

        if calc.charges:
            potentials = []
            for xyz in calc.xyzs:
                potential = 0
                coord = np.asarray(xyz[1:])
                for charge in calc.charges:
                    charge_coords = np.asarray(charge[1:4])
                    distance = np.linalg.norm(coord - charge_coords)
                    potential += charge[4] / distance
                potentials.append(322*potential)
            with open(f'{calc.name}_mol.in', 'w') as pc_file:
                print(f'\n{len(calc.xyzs)} 0', file=pc_file)
                [print(f'0 0 0 0 {potential}', file=pc_file) for potential in potentials]
            calc.additional_input_files.append((f'{calc.name}_mol.in', 'mol.in'))

        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(calc.rev_output_file_lines):
            if 'JOB ENDED NORMALLY' in line:
                return True
            if n_line > 50:
                # mopac will have a     * JOB ENDED NORMALLY *  line close to the end if terminated normally
                return False

        return False

    def get_energy(self, calc):
        for line in calc.output_file_lines:
            if 'TOTAL ENERGY' in line:
                # e.g.     TOTAL ENERGY            =       -476.93072 EV
                return Constants.eV2ha * float(line.split()[-2])

        return None

    def optimisation_converged(self, calc):

        for line in calc.rev_output_file_lines:
            if 'GRADIENT' in line and 'IS LESS THAN CUTOFF' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        raise NotImplementedError

    def get_imag_freqs(self, calc):
        raise NotImplementedError

    def get_normal_mode_displacements(self, calc, mode_number):
        raise NotImplementedError

    def get_final_xyzs(self, calc):

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

    def get_atomic_charges(self, calc):
        raise NotImplementedError

    def get_gradients(self, calc):
        gradients_section = False
        gradients = []
        for line in calc.output_file_lines:

            if 'FINAL  POINT  AND  DERIVATIVES' in line:
                gradients_section = True

            if gradients_section and 'ATOM   CHEMICAL' in line:
                gradients_section = False

            if gradients_section and len(line.split()) == 8:
                _, _, _, _, _, _, value, _ = line.split()
                gradients.append(value)
        grad_array = np.asarray(gradients)
        grad_array *= Constants.a02ang/Constants.ha2kcalmol
        grad_array.reshape((calc.n_atoms, 3))

        return grad_array.tolist()

    def __init__(self):
        super().__init__(name='mopac', path=Config.MOPAC.path, keywords=Config.MOPAC.keywords)


mopac = MOPAC()
