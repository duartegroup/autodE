from copy import copy
import numpy as np
from autode.wrappers.base import ElectronicStructureMethod
from autode.wrappers.base import execute
from autode.wrappers.keywords import Keywords
from autode.wrappers.keywords import SinglePointKeywords
from autode.wrappers.keywords import GradientKeywords
from autode.atoms import Atom
from autode.config import Config
from autode.constants import Constants
from autode.exceptions import UnsuppportedCalculationInput
from autode.geom import get_atoms_linear_interp
from autode.log import logger
from autode.utils import work_in_tmp_dir
import os


# dielectrics from Gaussian solvent list
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


def get_keywords(calc_input, molecule):
    """Get the keywords to use for a MOPAC calculation"""
    # To determine if there is an optimisation or single point the keywords
    # needs to be a subclass of Keywords
    assert isinstance(calc_input.keywords, Keywords)

    keywords = copy(calc_input.keywords)
    if isinstance(calc_input.keywords, SinglePointKeywords):
        # Single point calculation add the 1SCF keyword to prevent opt
        if not any('1scf' in kw.lower() for kw in keywords):
            keywords.append('1SCF')

    if isinstance(calc_input.keywords, GradientKeywords):
        # Gradient calculation needs GRAD
        if not any('grad' in kw.lower() for kw in keywords):
            keywords.append('GRAD')

    if calc_input.point_charges is not None:
        keywords.append('QMMM')

    if calc_input.solvent is not None:
        dielectric = solvents_and_dielectrics[calc_input.solvent]
        keywords.append(f'EPS={dielectric}')

    # Add the charge and multiplicity
    keywords.append(f'CHARGE={molecule.charge}')

    if molecule.mult != 1:
        if molecule.mult == 2:
            keywords.append('DOUBLET')
        elif molecule.mult == 3:
            keywords.append('TRIPLET')
        elif molecule.mult == 4:
            keywords.append('QUARTET')
        else:
            logger.critical('Unsupported spin multiplicity')
            raise UnsuppportedCalculationInput

    return keywords


def get_atoms_and_fixed_atom_indexes(molecule):
    """
    MOPAC seemingly doesn't have the capability to defined constrained bond
    lengths, so perform a linear interpolation to the atoms then fix the
    Cartesians

    Arguments:
        molecule (any):

    Returns:
        (tuple): List of non-fixed atoms and fixed atoms
    """
    fixed_atoms = []

    if molecule.constraints.distance is None:
        return molecule.atoms, fixed_atoms

    bonds = list(molecule.constraints.distance.keys())
    distances = list(molecule.constraints.distance.values())

    # Get a set of atoms that have been shifted using a linear interpolation
    atoms = get_atoms_linear_interp(atoms=molecule.atoms, bonds=bonds,
                                    final_distances=distances)

    # Populate a flat list of atom ids to fix
    fixed_atoms = [i for bond in bonds for i in bond]

    return atoms, fixed_atoms


def print_atoms(inp_file, atoms, fixed_atom_idxs):
    """Print the atoms to the input file depending on whether they are fixed"""

    for i, atom in enumerate(atoms):
        x, y, z = atom.coord

        if i in fixed_atom_idxs:
            line = f'{atom.label:<3}{x:^10.5f} 0 {y:^10.5f} 0 {z:^10.5f} 0'
        else:
            line = f'{atom.label:<3}{x:^10.5f} 1 {y:^10.5f} 1 {z:^10.5f} 1'

        print(line, file=inp_file)
    return


def print_point_charges(calc, atoms):
    """Print a point charge file if there are point charges"""

    if calc.input.point_charges is None:
        return

    potentials = []
    for atom in atoms:
        potential = 0
        coord = atom.coord
        for point_charge in calc.point_charges:
            # V = q/r_ij
            potential += (point_charge.charge
                          / np.linalg.norm(coord - point_charge.coord))

        # Distance in Å need to be converted to a0 and then the energy
        # Ha e^-1 to kcal mol-1 e^-1
        potentials.append(Constants.ha2kcalmol * Constants.a02ang * potential)

    with open(f'{calc.name}_mol.in', 'w') as pc_file:
        print(f'\n{len(atoms)} 0', file=pc_file)

        for potential in potentials:
            print(f'0 0 0 0 {potential}', file=pc_file)

    calc.additional_input_files.append(f'{calc.name}_mol.in')
    return


class MOPAC(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):

        with open(calc.input.filename, 'w') as input_file:
            keywords = get_keywords(calc.input, molecule)
            print(*keywords, '\n\n', file=input_file)

            atoms, fixed_atom_idxs = get_atoms_and_fixed_atom_indexes(molecule)

            if molecule.constraints.cartesian is not None:
                fixed_atom_idxs += calc.cartesian_constraints

            print_atoms(input_file, atoms, fixed_atom_idxs)
            print_point_charges(calc, atoms)

        return None

    def get_input_filename(self, calc):
        return f'{calc.name}_mopac.mop'

    def get_output_filename(self, calc):
        return f'{calc.name}_mopac.out'

    def execute(self, calc):

        @work_in_tmp_dir(filenames_to_copy=calc.input.get_input_filenames(),
                         kept_file_exts=('.mop', '.out'))
        def execute_mopac():
            logger.info(f'Setting the number of OMP threads to {calc.n_cores}')
            os.environ['OMP_NUM_THREADS'] = str(calc.n_cores)
            execute(calc, params=[calc.method.path, calc.input.filename])

        execute_mopac()
        return None

    def clean_up(self, calc):
        pass

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if 'JOB ENDED NORMALLY' in line:
                return True
            if n_line > 50:
                # Normal termination string is close to the end of the file
                return False

        return False

    def get_enthalpy(self, calc):
        raise NotImplementedError

    def get_free_energy(self, calc):
        raise NotImplementedError

    def get_energy(self, calc):
        for line in calc.output.file_lines:
            if 'TOTAL ENERGY' in line:
                # e.g.     TOTAL ENERGY            =       -476.93072 EV
                return Constants.eV2ha * float(line.split()[-2])

        return None

    def optimisation_converged(self, calc):

        for line in reversed(calc.output.file_lines):
            if 'GRADIENT' in line and 'IS LESS THAN CUTOFF' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        raise NotImplementedError

    def get_imaginary_freqs(self, calc):
        raise NotImplementedError

    def get_normal_mode_displacements(self, calc, mode_number):
        raise NotImplementedError

    def get_final_atoms(self, calc):

        atoms = []

        for n_line, line in enumerate(calc.output.file_lines):

            if n_line == len(calc.output.file_lines) - 3:
                # At the end of the file
                break

            line_length = len(calc.output.file_lines[n_line+3].split())

            if 'CARTESIAN COORDINATES' in line and line_length == 5:
                #                              CARTESIAN COORDINATES
                #
                #    1    C        1.255660629     0.020580974    -0.276235553

                atoms = []
                xyz_lines = calc.output.file_lines[n_line+2:n_line+2+calc.molecule.n_atoms]
                for xyz_line in xyz_lines:
                    atom_label, x, y, z = xyz_line.split()[1:]
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))

        return atoms

    def get_atomic_charges(self, calc):
        raise NotImplementedError

    def get_gradients(self, calc):
        gradients_section = False
        gradients = []
        for line in calc.output.file_lines:

            if 'FINAL  POINT  AND  DERIVATIVES' in line:
                gradients_section = True

            if gradients_section and 'ATOM   CHEMICAL' in line:
                gradients_section = False

            if gradients_section and len(line.split()) == 8:
                _, _, _, _, _, _, value, _ = line.split()
                gradients.append(value)
        grad_array = np.asarray(gradients)
        grad_array *= Constants.a02ang/Constants.ha2kcalmol
        grad_array.reshape((calc.molecule.n_atoms, 3))

        return grad_array.tolist()

    def __init__(self):
        super().__init__(name='mopac', path=Config.MOPAC.path, keywords_set=Config.MOPAC.keywords)


mopac = MOPAC()
