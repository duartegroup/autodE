import numpy as np
from typing import List
from autode.config import Config
import autode.wrappers.keywords as kws
from autode.wrappers.base import ElectronicStructureMethod
from autode.utils import run_external, work_in_tmp_dir


class QChem(ElectronicStructureMethod):
    """
    QChem Electronic Structure package

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
        raise NotImplementedError

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
        pass

    def optimisation_converged(self, calc) -> bool:
        pass

    def optimisation_nearly_converged(self, calc) -> bool:
        pass

    def get_final_atoms(self, calc) -> List:
        pass

    def get_atomic_charges(self, calc) -> List:
        pass

    def get_energy(self, calc) -> float:
        pass

    def get_gradients(self, calc) -> np.ndarray:
        pass

    def get_hessian(self, calc) -> np.ndarray:
        pass