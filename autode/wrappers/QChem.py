import numpy as np
import autode.wrappers.keywords as kws
import autode.wrappers.methods
from typing import List, TYPE_CHECKING

from autode.config import Config
from autode.values import PotentialEnergy, Gradient, Coordinates
from autode.log import logger
from autode.opt.optimisers.base import ExternalOptimiser
from autode.hessians import Hessian
from autode.utils import run_external, work_in_tmp_dir
from autode.exceptions import (
    CouldNotGetProperty,
    NotImplementedInMethod,
    UnsupportedCalculationInput,
)

if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor
    from autode.opt.optimisers.base import BaseOptimiser


class QChem(autode.wrappers.methods.ExternalMethodOEGH):
    """
    QChem Electronic Structure package.

    Tested versions:
        5.4.1

    Website: https://www.q-chem.com/
    User manual: https://manual.q-chem.com/5.1/index.html
    """

    def __init__(self):
        super().__init__(
            executable_name="qchem",
            path=Config.QChem.path,
            keywords_set=Config.QChem.keywords,
            implicit_solvation_type=Config.QChem.implicit_solvation_type,
            doi_list=["10.1080/00268976.2014.952696"],
        )

    def __repr__(self):
        return f"QChem(available = {self.is_available})"

    def generate_input_for(self, calc: "CalculationExecutor") -> None:
        """Generate a QChem input file"""
        assert calc.input.keywords is not None, "Must have input keywords"
        molecule = calc.molecule

        if calc.input.filename is None:
            raise ValueError(
                f"Cannot generate an input for {calc}. Input "
                "filename was undefined"
            )

        if (
            molecule.is_implicitly_solvated
            and not self._keywords_contain(calc, "solvent_method")
            and self.implicit_solvation_type is not None
        ):
            calc.input.keywords.append(self.implicit_solvation_type)

        if calc.input.point_charges is not None:
            raise NotImplementedError(
                "Point charges within QChem "
                "calculations are not yet supported"
            )

        with self._InputFileWriter(filename=calc.input.filename) as inp_file:
            inp_file.add_molecule_block(molecule)

            if self._is_ts_opt(calc):
                # TS optimisations require an initial frequency calculation
                inp_file.add_freq(calc)
                inp_file.add_calculation_seperator()

            inp_file.add_rem_block(calc)
            inp_file.add_solvent_block(calc)
            inp_file.add_constraints(calc)

            if self._is_ts_opt(calc):
                inp_file.add_molecule_read()

                # TS optimisation also require a final frequency calculation
                inp_file.add_calculation_seperator()
                inp_file.add_freq(calc)
                inp_file.add_molecule_read()

        return None

    @staticmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.out"

    @staticmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.in"

    def version_in(self, calc: "CalculationExecutor") -> str:
        """QChem version from a completed output file"""

        if not calc.output.exists:
            return "???"

        for line in calc.output.file_lines:
            if "Q-Chem" in line and len(line.split()) > 1:
                # e.g.  Q-Chem 5.4.1 for Intel X86 EM64T Linux
                str0, str1 = line.split()[:2]

                if str0 == "Q-Chem" and "." in str1 and "," not in str1:
                    return str1

        return "???"

    def execute(self, calc) -> None:
        """Execute a qchem calculation"""

        @work_in_tmp_dir(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=(".in", ".out"),
        )
        def execute_qchem():
            params = [self.path, "-nt", str(calc.n_cores), calc.input.filename]
            run_external(params, output_filename=calc.output.filename)

        execute_qchem()
        return None

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        """Did the calculation terminate normally?"""

        if not calc.output.exists:
            logger.warning(
                "QChem output did not exist - calculation " "did not terminate"
            )
            return False

        calc_started = False

        for line in calc.output.file_lines:
            if "Q-Chem begins" in line:
                calc_started = True

            if "MAXIMUM OPTIMIZATION CYCLES REACHED" in line:
                logger.info("Maximum number of optimisation steps reached")
                return True

            if "fatal error" in line or "input file has failed" in line:
                logger.error("Fatal error in QChem calculation. Final lines:")
                calc.output.try_to_print_final_lines(n=50)
                return False

        return True if calc_started else False

    def optimiser_from(self, calc: "CalculationExecutor") -> "BaseOptimiser":
        return QChemOptimiser(output_lines=calc.output.file_lines)

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        if not isinstance(calc.input.keywords, kws.OptKeywords):
            logger.warning(
                "Non-optimisation calculation performed - no change"
                " to geometry"
            )
            return calc.molecule.coordinates

        if calc.molecule.n_atoms == 1:
            # Coordinate of a single atom will not change
            return calc.molecule.coordinates

        coords: list[list[float]] = []

        for i, line in enumerate(calc.output.file_lines):
            if "Coordinates (Angstroms)" in line:
                start_idx = i + 2
            elif "Standard Nuclear Orientation (Angstroms)" in line:
                start_idx = i + 3
            else:
                continue

            """e.g.
                               Coordinates (Angstroms)
             ATOM                X               Y               Z
              1  O         0.0003489977   -0.1403224128    0.0000000000
              2  H        -0.7524338562    0.4527672831    0.0000000000
              3  H         0.7551329498    0.4500625364    0.0000000000
            Point Group: cs    Number of degrees of freedom:     3
            """

            end_idx = start_idx + calc.molecule.n_atoms
            coords = []
            for cline in calc.output.file_lines[start_idx:end_idx]:
                x, y, z = cline.split()[2:5]
                coords.append([float(x), float(y), float(z)])

        return Coordinates(coords, units="Å")

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        raise NotImplementedInMethod

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        """Get the total electronic energy from the calculation"""

        for line in reversed(calc.output.file_lines):
            if "Total energy" in line:
                try:
                    return PotentialEnergy(line.split()[-1], units="Ha")

                except (TypeError, ValueError, IndexError):
                    break

        raise CouldNotGetProperty("energy")

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        """Gradient of the potential energy"""

        try:
            gradients = self._raw_opt_gradient(calc)

        except CouldNotGetProperty:  # Failed to get gradient from optimisation
            gradients = self._raw_scf_grad(calc)

        return Gradient(gradients, units="Ha a0^-1").to("Ha Å^-1")

    def hessian_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> Hessian:
        """Extract the mass-weighted non projected Hessian matrix
        NOTE: Required $rem vibman_print 4 $end in the input"""
        assert calc.input.keywords is not None, "Must have keywords"

        hessian = self._extract_mass_weighted_hessian(calc)
        atom_masses = self._extract_atomic_masses(calc)

        # Un-mass weight
        mass_arr = np.repeat(atom_masses, repeats=3, axis=np.newaxis)
        hessian *= np.sqrt(np.outer(mass_arr, mass_arr))

        return Hessian(
            hessian,
            atoms=calc.molecule.atoms,
            functional=calc.input.keywords.functional,
            units="Ha a0^-2",
        ).to("Ha Å^-2")

    @staticmethod
    def _raw_opt_gradient(calc) -> list:
        grad = []

        for i, line in enumerate(calc.output.file_lines):
            if "Cartesian Gradient" not in line:
                continue

            """e.g.

                        Cartesian Gradient (au)
             ATOM              X           Y           Z
            1  O           0.000005   -0.000002    0.000000
            2  H           0.000017    0.000001    0.000000
            3  H          -0.000021    0.000001    0.000000
            """

            start_idx = i + 2
            end_idx = start_idx + calc.molecule.n_atoms

            grad = [
                [float(val) for val in _l.split()[2:]]
                for _l in calc.output.file_lines[start_idx:end_idx]
            ]

        if len(grad) == 0:
            raise CouldNotGetProperty("gradient")

        return grad

    @staticmethod
    def _raw_scf_grad(calc):
        grad = []
        n_grad_lines = (calc.molecule.n_atoms // 6 + 1) * 4

        for i, line in enumerate(calc.output.file_lines):
            if "Gradient of SCF Energy" not in line:
                continue

            """e.g.

            -----------------------------------------------------------------
            Calculating analytic gradient of the SCF energy
            Gradient of SCF Energy
                       1           2           3           4           5
               1  -0.0108562  -0.0095972   0.0087634   0.0032518  -0.0040093
            """

            start_idx = i + 1
            end_idx = start_idx + n_grad_lines
            lines_slice = calc.output.file_lines[start_idx:end_idx]

            grad = []

            for j in range(len(lines_slice) // 4):
                x_line = lines_slice[4 * j + 1]
                y_line = lines_slice[4 * j + 2]
                z_line = lines_slice[4 * j + 3]

                for k in range(1, len(x_line.split())):
                    grad.append(
                        [
                            float(x_line.split()[k]),
                            float(y_line.split()[k]),
                            float(z_line.split()[k]),
                        ]
                    )

        if len(grad) == 0:
            raise CouldNotGetProperty("gradient")

        return grad

    @staticmethod
    def _extract_atomic_masses(calc) -> np.ndarray:
        masses = []
        for line in calc.output.file_lines:
            if "Has Mass" in line:
                # e.g.
                #   Atom    1 Element O  Has Mass   15.99491

                mass = float(line.split()[-1])
                masses.append(mass)

        # Only return the final n_atoms masses
        return np.array(masses[-calc.molecule.n_atoms :])

    @staticmethod
    def _extract_mass_weighted_hessian(calc) -> np.ndarray:
        """Extract the mass weighted Hessian as a 3Nx3N matrix (N = n_atoms)"""

        n_atoms = calc.molecule.n_atoms
        lines = calc.output.file_lines

        hess = []

        def correct_shape(_hess):
            """Is the Hessian the correct shape? 3N x 3N"""
            return len(_hess) == 3 * n_atoms and all(
                len(row) == 3 * n_atoms for row in _hess
            )

        for i, line in enumerate(lines):
            if "Mass-Weighted Hessian Matrix" not in line:
                continue

            start_idx = i + 3
            end_idx = start_idx + 3 * n_atoms

            hess = [
                [float(val) for val in _l.split()]
                for _l in lines[start_idx:end_idx]
            ]

            while not correct_shape(hess):
                try:
                    start_idx = end_idx + 2
                    end_idx = start_idx + 3 * n_atoms
                    lines_slice = lines[start_idx:end_idx]

                    if len(lines_slice) == 0:
                        raise AssertionError

                    for j, _l in enumerate(lines_slice):
                        hess[j] += [float(val) for val in _l.split()]

                except (TypeError, ValueError, AssertionError):
                    raise CouldNotGetProperty("Hessian")

        if not correct_shape(hess):
            raise CouldNotGetProperty("Hessian")

        return np.array(hess)

    @staticmethod
    def _is_ts_opt(calc) -> bool:
        """Is the calculation a QChem TS optimisation?"""
        return any(
            "jobtype" in word.lower() and "ts" in word.lower()
            for word in calc.input.keywords
        )

    @staticmethod
    def _keywords_contain(calc, string) -> bool:
        return any(string in w.lower() for w in calc.input.keywords)

    class _InputFileWriter:
        def __init__(self, filename):
            self.file = open(filename, "w")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.file.close()

        def write(self, string, end="\n") -> None:
            print(string, file=self.file, end=end)

        def add_freq(self, calc) -> None:
            """Add a frequency calculation"""

            freq_calc = calc.copy()
            kwds = kws.HessianKeywords(
                [
                    kwd
                    for kwd in freq_calc.input.keywords
                    if "jobtype" not in kwd.lower()
                ]
            )
            freq_calc.input.keywords = kwds

            self.add_rem_block(freq_calc)
            self.add_solvent_block(freq_calc)

            return None

        def add_calculation_seperator(self) -> None:
            return self.write("\n@@@\n")

        def add_molecule_read(self) -> None:
            return self.write("$molecule\n    read\n$end")

        def add_constraints(self, calc) -> None:
            """Add cartesian and distance constraints"""

            if not isinstance(calc.input.keywords, kws.OptKeywords):
                # Constraints are only needed for optimisations
                return None

            constraints = calc.molecule.constraints

            if calc.input.added_internals is None and not constraints.any:
                return None

            self.write("$opt")

            if constraints.distance is not None:
                self.write("CONSTRAINT")

                for (i, j), dist in constraints.distance.items():
                    self.write(f'stre {i+1} {j+1} {dist.to("Å"):.5f}')

                self.write("ENDCONSTRAINT")

            if constraints.cartesian is not None:
                self.write("FIXED")

                for i in constraints.cartesian:
                    self.write(f"{i+1} XYZ")  # where i is an atom index

                self.write("ENDFIXED")

            if calc.input.added_internals is not None:
                self.write("CONNECT")

                for i, j in calc.input.added_internals:
                    self.write(f"{i+1} 1 {j+1}")

                self.write("ENDCONNECT")

            self.write("$end\n")
            return None

        def add_solvent_block(self, calc) -> None:
            """Add the solvent section, appropriate for an SMx solvent model"""
            if calc.molecule.solvent is None:
                # calculation is in the gas phase
                return None

            self.write(
                "$smx\n" f"solvent {calc.molecule.solvent.qchem}\n" f"$end\n"
            )

            return None

        def add_molecule_block(self, molecule) -> None:
            """Print molecular cartesian coordinates to the input file"""

            self.write("$molecule\n" f"{molecule.charge} {molecule.mult}")

            for atom in molecule.atoms:
                x, y, z = atom.coord
                self.write(f"{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}")

            self.write("$end\n")
            return None

        def add_rem_block(self, calc) -> None:
            """Add the calculation specification in a $rem block"""
            keywords = calc.input.keywords

            if any("$" in word.lower() for word in keywords):
                raise NotImplementedError(
                    "Cannot add $rem block - already " f"present in {keywords}"
                )

            self.write("$rem")

            if calc.molecule.n_atoms > 1:  # Defaults to a single point
                self._write_job_type(keywords)

            self._write_keywords(keywords, molecule=calc.molecule)

            if not isinstance(calc.input.keywords, kws.SinglePointKeywords):
                self.write("symmetry False")
                self.write("sym_ignore True")

            total_memory_in_mb = int(Config.max_core.to("MB") * calc.n_cores)
            self.write(f"mem_total {total_memory_in_mb}")

            self.write("$end\n")

            return None

        def _write_ecp(self, ecp_kwd, molecule) -> None:
            """Write the effective core potential (ECP) block, if required"""

            ecp_elems = set(
                atom.label
                for atom in molecule.atoms
                if atom.atomic_number >= ecp_kwd.min_atomic_number
            )

            if len(ecp_elems) > 0:
                logger.info(f"Writing ECP block for atoms {ecp_elems}")
                self.write(f"ecp {ecp_kwd.qchem}")

            return None

        def _write_keywords(self, keywords, molecule) -> None:
            for word in keywords:
                if isinstance(word, kws.BasisSet):
                    self.write(f"basis {word.qchem}")

                elif isinstance(word, kws.Functional):
                    self.write(f"method {word.qchem}")

                elif isinstance(word, kws.DispersionCorrection):
                    self.write(f"dft_d {word.qchem}")

                elif isinstance(word, kws.MaxOptCycles):
                    self.write(f"geom_opt_max_cycles {word}")

                elif isinstance(word, kws.ECP):
                    self._write_ecp(word, molecule=molecule)

                elif isinstance(word, kws.ImplicitSolventType):
                    if word.lower() != "smd":
                        err = f"Only SMD solvent is supported. Had: {word}"
                        raise UnsupportedCalculationInput(err)

                    self.write("solvent_method smd")

                elif "jobtype" in word.lower():
                    if molecule.n_atoms == 1 and "opt" in word.lower():
                        logger.warning("Cannot optimise a single atom")

                    elif " ts" in word.lower():
                        self.write(word)
                        # A completed Hessian calculation must be present
                        self.write("geom_opt_hessian read")

                    else:
                        self.write(word)

                else:
                    self.write(word)

            return None

        def _write_job_type(self, keywords) -> None:
            if any("jobtype" in word.lower() for word in keywords):
                logger.info("QChem *jobtype* already defined - not appending")

            elif isinstance(keywords, kws.OptKeywords):
                self.write("jobtype opt")

            elif isinstance(keywords, kws.HessianKeywords):
                self.write("jobtype freq")

            elif isinstance(keywords, kws.GradientKeywords):
                self.write("jobtype force")

            if isinstance(keywords, kws.OptKeywords) or isinstance(
                keywords, kws.HessianKeywords
            ):
                # Print the Hessian
                self.write("geom_opt_print 4\n" "vibman_print 4")

            return None


class QChemOptimiser(ExternalOptimiser):
    def __init__(self, output_lines: List[str]):
        self._lines = output_lines

    @property
    def converged(self) -> bool:
        return any("OPTIMIZATION CONVERGED" in line for line in self._lines)

    @property
    def last_energy_change(self) -> "PotentialEnergy":
        raise NotImplementedError
