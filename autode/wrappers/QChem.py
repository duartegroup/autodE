import numpy as np
import autode.wrappers.keywords as kws
from typing import List
from autode.config import Config
from autode.log import logger
from autode.atoms import Atom
from autode.exceptions import CouldNotGetProperty
from autode.wrappers.base import ElectronicStructureMethod
from autode.utils import run_external, work_in_tmp_dir


class QChem(ElectronicStructureMethod):
    """
    QChem Electronic Structure package.

    Tested versions:
        5.4.1

    Website: https://www.q-chem.com/
    User manual: https://manual.q-chem.com/5.1/index.html
    """

    def __init__(self):
        super().__init__('qchem',
                         path=Config.QChem.path,
                         keywords_set=Config.QChem.keywords,
                         implicit_solvation_type=Config.QChem.implicit_solvation_type,
                         doi='10.1080/00268976.2014.952696')

    def __repr__(self):
        return f'QChem(available = {self.available})'

    def generate_input(self, calc, molecule):
        """Generate a valid QChem input"""

        if calc.input.filename is None:
            raise ValueError(f'Cannot generate an input for {calc}. Input '
                             'filename was undefined')

        with _InputFileWriter(filename=calc.input.filename) as inp_file:
            inp_file.add_molecule_block(molecule)
            inp_file.add_rem_block(calc)
            # TODO: other

        return None

    def get_output_filename(self, calc) -> str:
        return f'{calc.name}.out'

    def get_input_filename(self, calc) -> str:
        return f'{calc.name}.in'

    def get_version(self, calc) -> str:
        """QChem version from a completed output file"""

        if not calc.output.exists:
            return '???'

        for line in calc.output.file_lines:

            if 'Q-Chem' in line and len(line.split()) > 1:
                # e.g.  Q-Chem 5.4.1 for Intel X86 EM64T Linux
                str0, str1 = line.split()[:2]

                if str0 == 'Q-Chem' and '.' in str1 and ',' not in str1:
                    return str1

        return '???'

    def execute(self, calc) -> None:
        """Execute a qchem calculation"""

        @work_in_tmp_dir(filenames_to_copy=calc.input.filenames,
                         kept_file_exts=('.in', '.out'))
        def execute_qchem():
            params = [self.path, '-nt', str(calc.n_cores), calc.input.filename]
            run_external(params, output_filename=calc.output.filename)

        execute_qchem()
        return None

    def calculation_terminated_normally(self, calc) -> bool:
        """Did the calculation terminate normally?"""

        if not calc.output.exists:
            logger.warning('QChem output did not exist - calculation '
                           'did not terminate')
            return False

        calc_started = False

        for line in calc.output.file_lines:
            if 'Q-Chem begins' in line:
                calc_started = True

            if 'MAXIMUM OPTIMIZATION CYCLES REACHED' in line:
                logger.info('Maximum number of optimisation steps reached')
                return True

            if 'fatal error' in line:
                logger.error('Fatal error in QChem calculation. Final lines:')
                calc.print_final_output_lines(n=50)
                return False

        return True if calc_started else False

    def optimisation_converged(self, calc) -> bool:
        return any('OPTIMIZATION CONVERGED' in line
                   for line in calc.output.file_lines)

    def optimisation_nearly_converged(self, calc) -> bool:
        pass

    def get_final_atoms(self, calc) -> List[Atom]:

        atoms = []

        for i, line in enumerate(calc.output.file_lines):

            if 'Coordinates (Angstroms)' not in line:
                continue

            """e.g.
            
                               Coordinates (Angstroms)
             ATOM                X               Y               Z
              1  O         0.0003489977   -0.1403224128    0.0000000000
              2  H        -0.7524338562    0.4527672831    0.0000000000
              3  H         0.7551329498    0.4500625364    0.0000000000
            Point Group: cs    Number of degrees of freedom:     3
            """

            start_idx = i+2
            end_idx = start_idx + calc.molecule.n_atoms

            atoms = []
            for cline in calc.output.file_lines[start_idx:end_idx]:
                atoms.append(Atom(cline.split()[1], *cline.split()[2:5]))

        return atoms

    def get_atomic_charges(self, calc) -> List:
        raise NotImplementedError

    def get_energy(self, calc) -> float:
        """Get the total electronic energy from the calculation"""

        for line in reversed(calc.output.file_lines):

            if 'Total energy' in line:
                try:
                    return float(line.split()[-1])

                except (TypeError, ValueError, IndexError):
                    break

        raise CouldNotGetProperty('energy')

    def get_gradients(self, calc) -> np.ndarray:
        pass

    def get_hessian(self, calc) -> np.ndarray:
        pass


class _InputFileWriter:

    def __init__(self, filename):
        self.file = open(filename, 'w')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def write(self, string, end='\n') -> None:
        print(string, file=self.file, end=end)

    def add_molecule_block(self, molecule) -> None:
        """Print the cartesian coordinates of a molecule to the input file"""

        self.write('$molecule\n'
                   f'{molecule.charge} {molecule.mult}')

        for atom in molecule.atoms:
            x, y, z = atom.coord
            self.write(f'{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}')

        self.write('$end')
        return None

    def add_rem_block(self, calc) -> None:
        """Add the $rem block"""
        keywords = calc.input.keywords

        if any('$' in word.lower() for word in keywords):
            raise NotImplementedError('Cannot add $rem block - additional bloc'
                                      f'ks included in keywords: {keywords}')

        self.write('$rem')
        self._write_job_type(keywords)
        self._write_keywords(keywords, molecule=calc.molecule)
        self.write('$end')

        return None

    def _write_ecp(self, ecp_kwd, molecule) -> None:
        """Write the effective core potential (ECP) block, if required"""

        ecp_elems = set(atom.label for atom in molecule.atoms
                        if atom.atomic_number >= ecp_kwd.min_atomic_number)

        if len(ecp_elems) > 0:
            logger.info(f'Writing ECP block for atoms {ecp_elems}')
            self.write(f'ecp {ecp_kwd.qchem}')

        return None

    def _write_keywords(self, keywords, molecule) -> None:

        for word in keywords:

            if isinstance(word, kws.BasisSet):
                self.write(f'basis {word.qchem}')

            elif isinstance(word, kws.Functional):
                self.write(f'method {word.qchem}')

            elif isinstance(word, kws.DispersionCorrection):
                self.write(f'dft_d {word.qchem}')

            elif isinstance(word, kws.MaxOptCycles):
                self.write(f'geom_opt_max_cycle {word}')

            elif isinstance(word, kws.ECP):
                self._write_ecp(word, molecule=molecule)

            else:
                self.write(word)

        return None

    def _write_job_type(self, keywords) -> None:

        if any('jobtype' in word.lower() for word in keywords):
            logger.info('QChem *jobtype* already defined - not appending')

        elif isinstance(keywords, kws.OptKeywords):
            self.write('jobtype opt')

        elif isinstance(keywords, kws.HessianKeywords):
            self.write('jobtype freq')

        elif isinstance(keywords, kws.GradientKeywords):
            self.write('jobtype force')

        if (isinstance(keywords, kws.OptKeywords)
                or isinstance(keywords, kws.HessianKeywords)):
            # Print the Hessian
            self.write('geom_opt_print 4')

        return None
