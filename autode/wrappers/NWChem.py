import numpy as np
import autode.wrappers.keywords as kws
from autode.wrappers.base import ElectronicStructureMethod
from autode.utils import run_external_monitored
from autode.atoms import Atom
from autode.geom import symm_matrix_from_ltril
from autode.config import Config
from autode.exceptions import UnsuppportedCalculationInput, CouldNotGetProperty
from autode.log import logger
from autode.constants import Constants
from autode.utils import work_in_tmp_dir


def ecp_block(molecule, keywords):
    """
    Generate a block of input for any effective core potentials to add

    Arguments:
        molecule (autode.species.Species):
        keywords (autode.wrappers.keywords.Keywords):

    Returns:
        (str):
    """
    ecp_kwd = keywords.ecp

    if ecp_kwd is None:
        return ""   # No ECP is defined in these keywords

    # Set of unique atomic symbols that require an ECP
    ecp_elems = set(atom.label for atom in molecule.atoms
                    if atom.atomic_number >= ecp_kwd.min_atomic_number)

    if len(ecp_elems) == 0:
        return ""   # No atoms require an ECP

    ecp_str = '\necp\n'
    ecp_str += '\n'.join(f'  {label}   library  {ecp_kwd.nwchem}'
                         for label in ecp_elems)
    ecp_str += '\nend'

    return ecp_str


def get_keywords(calc_input, molecule):
    """Generate a keywords list and adding solvent"""

    new_keywords = []
    scf_block = False

    for keyword in calc_input.keywords:

        if isinstance(keyword, kws.Functional):
            keyword = f'dft\n  maxiter 100\n  xc {keyword.nwchem}\nend'

        elif isinstance(keyword, kws.BasisSet):
            keyword = f'basis\n  *   library {keyword.nwchem}\nend'
            keyword += ecp_block(molecule, keywords=calc_input.keywords)

        elif isinstance(keyword, kws.ECP):
            # ECPs are added to the basis block
            continue

        elif isinstance(keyword, kws.Keyword):
            keyword = keyword.nwchem

        if 'opt' in keyword.lower() and molecule.n_atoms == 1:
            logger.warning('Cannot do an optimisation for a single atom')

            # Replace any 'opt' containing word in this keyword with energy
            words = []
            for word in keyword.split():
                if 'opt' in word:
                    words.append('energy')
                else:
                    words.append(word)

            new_keywords.append(' '.join(words))

        elif keyword.lower().startswith('dft'):
            lines = keyword.split('\n')
            lines.insert(1, f'  mult {molecule.mult}')
            new_keyword = '\n'.join(lines)
            new_keywords.append(new_keyword)

        elif keyword.lower().startswith('scf'):
            if calc_input.solvent:
                logger.critical('nwchem only supports solvent for DFT calcs')
                raise UnsuppportedCalculationInput

            scf_block = True
            lines = keyword.split('\n')
            lines.insert(1, f'  nopen {molecule.mult - 1}')
            new_keyword = '\n'.join(lines)
            new_keywords.append(new_keyword)

        elif (any(st in keyword.lower() for st in ['ccsd', 'mp2'])
              and not scf_block):

            if calc_input.solvent.keyword is not None:
                logger.critical('nwchem only supports solvent for DFT calcs')
                raise UnsuppportedCalculationInput

            new_keywords.append(f'scf\n  nopen {molecule.mult - 1}\nend')
            new_keywords.append(keyword)
        else:
            new_keywords.append(keyword)

    return new_keywords


def print_added_internals(inp_file, calc_input, molecule):
    """Add internal coordinates to the input file for added internals and
    distance constraints"""

    if calc_input.added_internals is None and molecule.constraints.distance is None:
        return

    print('  zcoord', file=inp_file)

    # NWChem indexes atoms from 1
    if calc_input.added_internals is not None:
        for (i, j) in calc_input.added_internals:
            print(f'    bond {i+1} {j+1}', file=inp_file)

    if molecule.constraints.distance is not None:
        for (i, j) in molecule.constraints.distance.keys():
            print(f'    bond {i + 1} {j + 1}', file=inp_file)

    print('  end', file=inp_file)
    return


def print_constraints(inp_file, molecule, force_constant=20):
    """Add distance and cartesian constraints to the input file"""

    if molecule.constraints.distance is None and molecule.constraints.cartesian is None:
        return

    print('constraints', file=inp_file)

    # nwchem indexes atoms from 1 so increment atom ids by 1
    if molecule.constraints.distance is not None:
        for (i, j), dist in molecule.constraints.distance.items():
            dist_a0 = dist / Constants.a0_to_ang  # Constraints are in Bohr

            print(f'  spring bond {i+1} {j+1} {force_constant} {dist_a0:.3f}',
                  file=inp_file)

    if molecule.constraints.cartesian is not None:

        const_atom_idxs = [i + 1 for i in molecule.constraints.cartesian]
        list_of_ranges, used_atoms = [], []

        for i in const_atom_idxs:
            rang = []
            if i not in used_atoms:
                while i in const_atom_idxs:
                    used_atoms.append(i)
                    rang.append(i)
                    i += 1
                if len(rang) in (1, 2):
                    list_of_ranges += rang
                else:
                    list_of_ranges.append(f'{rang[0]}:{rang[-1]}')

        print('  fix atom', end=' ', file=inp_file)
        print(*list_of_ranges, sep=' ', file=inp_file)

    print('end', file=inp_file)
    return


class NWChem(ElectronicStructureMethod):

    def generate_input(self, calc, molecule):
        # TODO impliment partial hessian
        keywords = get_keywords(calc.input, molecule)

        with open(calc.input.filename, 'w') as inp_file:

            print(f'start {calc.name}_nwchem\necho', file=inp_file)

            if calc.input.solvent is not None:
                print(f'cosmo\n '
                      f'do_cosmo_smd true\n '
                      f'solvent {calc.input.solvent}\n'
                      f'end', file=inp_file)

            print('geometry', end=' ', file=inp_file)
            if molecule.constraints.distance or molecule.constraints.cartesian:
                print('noautoz', file=inp_file)
            else:
                print('', file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}',
                      file=inp_file)

            print_added_internals(inp_file, calc.input, molecule)
            print('end', file=inp_file)

            print(f'charge {calc.molecule.charge}', file=inp_file)
            print_constraints(inp_file, molecule)

            if calc.input.point_charges is not None:
                print('bq')
                for charge, x, y, z in calc.point_charges:
                    print(f'{x:^12.8f} {y:^12.8f} {z:^12.8f} {charge:^12.8f}',
                          file=inp_file)
                print('end')

            print(f'memory {Config.max_core} mb', file=inp_file)

            print(*keywords, sep='\n', file=inp_file)

            # Will used partial an ESP initialisation to generate partial
            # atomic charges - more accurate than the standard Mulliken
            # analysis (or at least less sensitive to the method)
            print('task esp', file=inp_file)

        return None

    def get_input_filename(self, calc):
        return f'{calc.name}.nw'

    def get_output_filename(self, calc):
        return f'{calc.name}.out'

    def get_version(self, calc):
        """Get the NWChem version from the output file"""
        for line in calc.output.file_lines:

            if '(NWChem)' in line:
                # e.g. Northwest Computational Chemistry Package (NWChem) 6.6
                return line.split()[-1]

        logger.warning('Could not find the NWChem version')
        return '???'

    def execute(self, calc):

        @work_in_tmp_dir(filenames_to_copy=calc.input.filenames,
                         kept_file_exts=('.nw', '.out'))
        def execute_nwchem():
            params = ['mpirun', '-np', str(calc.n_cores), calc.method.path,
                      calc.input.filename]

            run_external_monitored(params, calc.output.filename,
                                   break_words=['Received an Error', 'MPI_ABORT'])
        execute_nwchem()
        return None

    def calculation_terminated_normally(self, calc):

        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if any(substring in line for substring in['CITATION',
                                                      'Failed to converge in maximum number of steps or available time']):
                logger.info('nwchem terminated normally')
                return True
            if 'MPI_ABORT' in line:
                return False

            if n_line > 500:
                return False

        return False

    def get_enthalpy(self, calc):
        raise NotImplementedError

    def get_free_energy(self, calc):
        raise NotImplementedError

    def get_energy(self, calc):

        wf_strings = ['Total CCSD energy', 'Total CCSD(T) energy',
                      'Total SCS-MP2 energy', 'Total MP2 energy',
                      'Total RI-MP2 energy']

        for line in reversed(calc.output.file_lines):
            if any(string in line for string in ['Total DFT energy', 'Total SCF energy']):
                return float(line.split()[4])

            if any(string in line for string in wf_strings):
                return float(line.split()[3])

        raise CouldNotGetProperty(name='energy')

    def optimisation_converged(self, calc):

        for line in reversed(calc.output.file_lines):
            if 'Optimization converged' in line:
                return True

        return False

    def optimisation_nearly_converged(self, calc):
        if self.optimisation_converged(calc):
            return False

        for j, line in enumerate(reversed(calc.output.file_lines)):
            if '@' in line and 'ok' in calc.output.file_lines[-j-1]:
                return True

        return False

    def get_final_atoms(self, calc):

        xyzs_section = False
        atoms = []

        for line in calc.output.file_lines:
            if 'Output coordinates in angstroms' in line:
                xyzs_section = True
                atoms = []

            if 'Atomic Mass' in line:
                xyzs_section = False

            if xyzs_section and len(line.split()) == 6:
                if line.split()[0].isdigit():
                    _, atom_label, _, x, y, z = line.split()
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))

        return atoms

    def get_atomic_charges(self, calc):
        """
        e.g.
         Atom              Coordinates                           Charge

                                                  ESP


        1 C    -0.000814    0.000010    0.001095   -0.266058
        . .       .            .            .          .
        """
        charges_section = False
        charges = []

        for line in calc.output.file_lines:
            if len(line.split()) == 3 and 'Atom' in line and 'Coordinates' in line and 'Charge' in line:
                charges_section = True
                charges = []

            if charges_section and len(line.split()) == 6:
                charge = line.split()[-1]
                charges.append(float(charge))

            if charges_section and '------------' in line:
                charges_section = False

        print(charges)
        return charges

    def get_gradients(self, calc):

        gradients_section = False
        gradients = []
        for line in calc.output.file_lines:
            if 'DFT ENERGY GRADIENTS' in line:
                gradients_section = True
                gradients = []

            if '----------------------------------------' in line and gradients_section:
                gradients_section = False

            if gradients_section and len(line.split()) == 8:
                x, y, z = line.split()[5:]
                gradients.append(np.array([float(x), float(y), float(z)]))

        # Convert from Ha a0^-1 to Ha A-1
        gradients = [grad / Constants.a0_to_ang for grad in gradients]
        return np.array(gradients)

    @staticmethod
    def _atom_masses_from_hessian(calc):
        """
        Grab the atomic masses from the 'atom information' section, which
        should be present from a Hessian calculation. Block looks like::

            ---------------------------- Atom information ----------------
             atom    #      X           Y          Z            mass
            --------------------------------------------------------------
            O        1  0.0000D+00  0.000D+00  2.26367D-01  1.5994910D+01
            H        2  1.4235D+00  0.000D+00 -9.05466D-01  1.0078250D+00
            H        3 -1.4435D+00  0.000D+00 -9.05466D-01  1.0078250D+00

        Returns:
            (list(float)):
        """
        n_atoms, file_lines = calc.molecule.n_atoms, calc.output.file_lines
        atom_lines = None

        for i, line in enumerate(reversed(file_lines)):

            if 'Atom information' not in line:
                continue

            atom_lines = file_lines[-i+2:-i+2+n_atoms]
            break

        if atom_lines is None:
            raise CouldNotGetProperty('No masses found in output file')

        # Replace double notation for standard 'E' and float all the final
        # entries, which should be the masses in amu
        return [float(line.split()[-1].replace('D', 'E')) for line in atom_lines]

    def get_hessian(self, calc):
        """
        Get the un-mass weighted Hessian matrix from the calculation. Block
        looks like::

           ----------------------------------------------------
          MASS-WEIGHTED NUCLEAR HESSIAN (Hartree/Bohr/Bohr/Kamu)
          ----------------------------------------------------


                       1            2           .....
           ----- ----- ----- ----- -----
            1    4.25381D+01
            2   -8.96428D-10 -4.68356D-04
            .        .             .           .

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (np.ndarray):
        """
        hess_lines = None

        for i, line in enumerate(calc.output.file_lines):

            if 'MASS-WEIGHTED NUCLEAR HESSIAN' not in line:
                continue

            start_line = i+6
            n = 3*calc.molecule.n_atoms

            hess_lines = calc.output.file_lines[start_line:start_line+n]
            break

        if hess_lines is None:
            raise CouldNotGetProperty('Hessian not found in the output file')

        atom_masses = self._atom_masses_from_hessian(calc)

        # Construct a flat list of hessian elements from the lower triangular
        # elements printed
        hess_values = []
        for line in hess_lines:
            line = line.replace('D', 'E')
            hess_values += [float(val) for val in line.split()[1:]]

        hess = symm_matrix_from_ltril(array=hess_values)

        # Un-mass weight from Kamu^-1 to 1
        mass_arr = np.repeat(atom_masses,  repeats=3, axis=np.newaxis) * 1E-3
        hess *= np.sqrt(np.outer(mass_arr, mass_arr))

        # and convert from atomic units (Ha/a0^2) to base units (Ha/Å^2)
        return hess / Constants.a0_to_ang**2

    def __init__(self):
        super().__init__('nwchem', path=Config.NWChem.path,
                         keywords_set=Config.NWChem.keywords,
                         implicit_solvation_type=Config.NWChem.implicit_solvation_type,
                         doi='10.1063/5.0004997')


nwchem = NWChem()
