import os
import shutil
import numpy as np
import autode.wrappers.methods

from autode.values import Coordinates, Gradient, PotentialEnergy
from autode.utils import run_external
from autode.wrappers.keywords import OptKeywords, GradientKeywords
from autode.config import Config
from autode.opt.optimisers.base import ExternalOptimiser
from autode.exceptions import AtomsNotFound, CouldNotGetProperty
from autode.utils import work_in_tmp_dir, run_in_tmp_environment
from autode.log import logger


class XTB(autode.wrappers.methods.ExternalMethodOEG):
    def __init__(self):
        super().__init__(
            executable_name="xtb",
            path=Config.XTB.path,
            keywords_set=Config.XTB.keywords,
            implicit_solvation_type=Config.XTB.implicit_solvation_type,
            doi_list=["10.1002/wcms.1493"],
        )

        self.force_constant = Config.XTB.force_constant

    def __repr__(self):
        return f"XTB(available = {self.is_available})"

    def print_distance_constraints(self, inp_file, molecule):
        """Add distance constraints to the input file"""

        if molecule.constraints.distance is None:
            return None

        for (i, j), dist in molecule.constraints.distance.items():
            # XTB counts from 1 so increment atom ids by 1
            print(
                f"$constrain\n"
                f"force constant={self.force_constant}\n"
                f"distance:{i+1}, {j+1}, {dist:.4f}\n$",
                file=inp_file,
            )
        return None

    def print_cartesian_constraints(self, inp_file, molecule):
        """Add cartesian constraints to an xtb input file"""

        if molecule.constraints.cartesian is None:
            return None

        atom_idxs = list(
            sorted(int(i) + 1 for i in molecule.constraints.cartesian)
        )
        list_of_ranges = []

        for atom_idx in atom_idxs:
            last_range = (
                list_of_ranges[-1] if len(list_of_ranges) > 0 else None
            )
            if last_range is not None and atom_idx - 1 == last_range[-1]:
                last_range.append(atom_idx)
            else:
                list_of_ranges.append([atom_idx])

        list_of_ranges_str = [
            f"{idxs[0]}-{idxs[-1]}" if len(idxs) > 1 else str(idxs[0])
            for idxs in list_of_ranges
        ]
        print(
            f"$constrain\n"
            f"force constant={self.force_constant}\n"
            f'atoms: {",".join(list_of_ranges_str)}\n'
            f"$",
            file=inp_file,
        )
        return None

    @staticmethod
    def print_point_charge_file(calc):
        """Generate a point charge file"""

        if calc.input.point_charges is None:
            return None

        with open(f"{calc.name}_xtb.pc", "w") as pc_file:
            print(len(calc.input.point_charges), file=pc_file)

            for point_charge in calc.input.point_charges:
                x, y, z = point_charge.coord
                charge = point_charge.charge
                print(
                    f"{charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f}",
                    file=pc_file,
                )

        calc.input.additional_filenames.append(f"{calc.name}_xtb.pc")
        return None

    def print_xcontrol_file(self, calc, molecule):
        """Print an XTB input file with constraints and point charges"""

        xcontrol_filename = f"xcontrol_{calc.name}"
        with open(xcontrol_filename, "w") as xcontrol_file:

            self.print_distance_constraints(xcontrol_file, molecule)
            self.print_cartesian_constraints(xcontrol_file, molecule)

            if calc.input.point_charges is not None:
                self.print_point_charge_file(calc)
                print(
                    f"$embedding\n"
                    f"input={calc.name}_xtb.pc\n"
                    f"input=orca\n"
                    f"$end",
                    file=xcontrol_file,
                )

        calc.input.additional_filenames.append(xcontrol_filename)
        return

    def generate_input_for(self, calc):

        molecule = calc.molecule
        calc.molecule.print_xyz_file(filename=calc.input.filename)

        if molecule.constraints.any or calc.input.point_charges:
            self.print_xcontrol_file(calc, molecule)

        return None

    @staticmethod
    def input_filename_for(calc):
        return f"{calc.name}.xyz"

    @staticmethod
    def output_filename_for(calc):
        return f"{calc.name}.out"

    def version_in(self, calc):
        """Get the XTB version from the output file"""

        for line in calc.output.file_lines:
            if "xtb version" in line and len(line.split()) >= 4:
                # e.g.   * xtb version 6.2.3 (830e466) compiled by ....
                return line.split()[3]

        logger.warning("Could not find the XTB version in the output file")
        return "???"

    @staticmethod
    def _remove_xtbopt_xyz_file() -> None:
        if os.path.exists("xtbopt.xyz"):
            os.remove("xtbopt.xyz")

        return None

    def execute(self, calc):
        """Execute an XTB calculation using the runtime flags"""
        # XTB calculation keywords must be a class

        flags = [
            "--chrg",
            str(calc.molecule.charge),
            "--uhf",
            str(calc.molecule.mult - 1),
        ]

        if isinstance(calc.input.keywords, OptKeywords):
            if calc.input.keywords.max_opt_cycles is not None:
                logger.warning("Switching off optimisation cycle limit")
                calc.input.keywords.max_opt_cycles = None

        if len(calc.input.keywords) != 0:
            flags += list(calc.input.keywords)

        elif isinstance(calc.input.keywords, OptKeywords):
            flags.append("--opt")

        elif isinstance(calc.input.keywords, GradientKeywords):
            flags.append("--grad")

        if calc.molecule.solvent is not None:
            flags += ["--gbsa", calc.molecule.solvent.xtb]

        if len(calc.input.additional_filenames) > 0:
            # XTB allows for an additional xcontrol file, which should be the
            # last file in the list
            flags += ["--input", calc.input.additional_filenames[-1]]

        @work_in_tmp_dir(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=(".xyz", ".out", ".pc", ".grad"),
            use_ll_tmp=True,
        )
        @run_in_tmp_environment(
            OMP_NUM_THREADS=calc.n_cores, GFORTRAN_UNBUFFERED_ALL=1
        )
        def execute_xtb():

            logger.info(f'Running XTB with: {" ".join(flags)}')
            run_external(
                params=[calc.method.path, calc.input.filename] + flags,
                output_filename=calc.output.filename,
            )

            if os.path.exists("gradient"):
                shutil.move("gradient", f"{calc.name}_OLD.grad")

            self._remove_xtbopt_xyz_file()

        execute_xtb()
        return None

    def terminated_normally_in(self, calc):

        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if "ERROR" in line:
                return False
            if n_line > 20:
                # With xtb we will search for there being no '#ERROR!' in the
                # last few lines
                return True

        return False

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:

        for line in reversed(calc.output.file_lines):
            if "total E" in line:
                return PotentialEnergy(line.split()[-1], units="Ha")
            if "TOTAL ENERGY" in line:
                return PotentialEnergy(line.split()[-3], units="Ha")

        raise CouldNotGetProperty(name="energy")

    @staticmethod
    def converged_line_in_output(calc):

        for line in reversed(calc.output.file_lines):
            if "GEOMETRY OPTIMIZATION CONVERGED" in line:
                return True

        return False

    def optimiser_from(
        self, calc: "CalculationExecutor"
    ) -> "autode.opt.optimisers.base.BaseOptimiser":
        return XTBOptimiser(converged=self.converged_line_in_output(calc))

    @staticmethod
    def _get_final_coords_6_2_above(calc):
        """
        e.g.

        ================
         final structure:
        ================
        5
         xtb: 6.2.3 (830e466)
        Cl        1.62694523673790    0.09780349799138   -0.02455489507427
        C        -0.15839164427314   -0.00942638308615    0.00237760557913
        H        -0.46867957388620   -0.59222865914178   -0.85786049981721
        H        -0.44751262498645   -0.49575975568264    0.92748366742968
        H        -0.55236139359212    0.99971129991918   -0.04744587811734
        """
        matrix = []

        for i, line in enumerate(calc.output.file_lines):
            if "final structure" in line:
                n_atoms = int(calc.output.file_lines[i + 2].split()[0])

                for xyz_line in calc.output.file_lines[
                    i + 4 : i + 4 + n_atoms
                ]:
                    _, x, y, z = xyz_line.split()
                    matrix.append([float(x), float(y), float(z)])
                break

        return Coordinates(matrix, units="Å")

    @staticmethod
    def _get_final_coords_old(calc):
        """
        e.g.

        ================
         final structure:
        ================
        $coord
            2.52072290250473   -0.04782551206377   -0.50388676977877      C
                    .                 .                    .              .
        """
        matrix = []
        geom_section = False

        for line in calc.output.file_lines:

            if "$coord" in line:
                geom_section = True

            if "$end" in line and geom_section:
                geom_section = False

            if len(line.split()) == 4 and geom_section:
                x, y, z, _ = line.split()
                matrix.append([float(x), float(y), float(z)])

        return Coordinates(matrix, units="a0").to("Å")

    def coordinates_from(self, calc):

        for i, line in enumerate(calc.output.file_lines):

            # XTB 6.2.x have a slightly different way of printing the atoms
            if (
                "xtb version" in line
                or "Version" in line
                and len(line.split()) >= 4
            ):

                if line.split()[3] == "6.2.2" or "6.1" in line.split()[2]:
                    return self._get_final_coords_old(calc)

                else:
                    return self._get_final_coords_6_2_above(calc)

            # Version is not recognised if we're 50 lines into the output file
            # - try and use the old version
            if i > 50:
                return self._get_final_coords_old(calc)

        raise AtomsNotFound(
            "Failed to find any coordinates in XTB " "output file"
        )

    def partial_charges_from(self, calc):
        charges_sect = False
        charges = []
        for line in calc.output.file_lines:
            if "Mol." in line:
                charges_sect = False
            if charges_sect and len(line.split()) == 7:
                charges.append(float(line.split()[4]))
            if "covCN" in line:
                charges_sect = True
        return charges

    def gradient_from(self, calc):
        raw = []

        if os.path.exists(f"{calc.name}_xtb.grad"):
            grad_file_name = f"{calc.name}_xtb.grad"
            with open(grad_file_name, "r") as grad_file:
                for line in grad_file:
                    x, y, z = line.split()
                    raw.append(np.array([float(x), float(y), float(z)]))

        elif os.path.exists(f"{calc.name}_OLD.grad"):
            with open(f"{calc.name}_OLD.grad", "r") as grad_file:
                for i, line in enumerate(grad_file):
                    if i > 1 and len(line.split()) == 3:
                        x, y, z = line.split()
                        vec = [
                            float(x.replace("D", "E")),
                            float(y.replace("D", "E")),
                            float(z.replace("D", "E")),
                        ]

                        raw.append(np.array(vec))

            os.remove(f"{calc.name}_OLD.grad")

            with open(f"{calc.name}_xtb.grad", "w") as new_grad_file:
                [
                    print(
                        "{:^12.8f} {:^12.8f} {:^12.8f}".format(*line),
                        file=new_grad_file,
                    )
                    for line in raw
                ]

        if len(raw) == 0:
            raise CouldNotGetProperty(name="gradient")

        return Gradient(raw, units="Ha a0^-1").to("Ha Å^-1")


class XTBOptimiser(ExternalOptimiser):
    def __init__(self, converged: bool):
        self._converged = converged

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def last_energy_change(self) -> "PotentialEnergy":
        raise NotImplementedError


xtb = XTB()
