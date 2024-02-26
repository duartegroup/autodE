import numpy as np
import os
import autode.wrappers.keywords as kws
import autode.wrappers.methods
from typing import List, TYPE_CHECKING

from autode.utils import run_external
from autode.hessians import Hessian
from autode.opt.optimisers.base import ExternalOptimiser
from autode.values import PotentialEnergy, Gradient, Coordinates
from autode.input_output import xyz_file_to_atoms
from autode.config import Config
from autode.utils import work_in_tmp_dir
from autode.log import logger
from autode.exceptions import (
    UnsupportedCalculationInput,
    CouldNotGetProperty,
    NoCalculationOutput,
    XYZfileWrongFormat,
    AtomsNotFound,
)

if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor
    from autode.opt.optimisers.base import BaseOptimiser

vdw_gaussian_solvent_dict = {
    "water": "Water",
    "acetone": "Acetone",
    "acetonitrile": "Acetonitrile",
    "benzene": "Benzene",
    "carbon tetrachloride": "CCl4",
    "dichloromethane": "CH2Cl2",
    "chloroform": "Chloroform",
    "cyclohexane": "Cyclohexane",
    "n,n-dimethylformamide": "DMF",
    "dimethylsulfoxide": "DMSO",
    "ethanol": "Ethanol",
    "n-hexane": "Hexane",
    "methanol": "Methanol",
    "1-octanol": "Octanol",
    "pyridine": "Pyridine",
    "tetrahydrofuran": "THF",
    "toluene": "Toluene",
}


def print_added_internals(inp_file, calc_input):
    """Print the added internal coordinates"""

    if calc_input.added_internals is None:
        return

    for i, j in calc_input.added_internals:
        print(
            "%geom\n" "modify_internal\n" "{ B",
            i,
            j,
            "A } end\n" "end",
            file=inp_file,
        )
    return


def print_distance_constraints(inp_file, molecule):
    """Print the distance constraints to the input file"""
    if molecule.constraints.distance is None:
        return

    print("%geom Constraints", file=inp_file)
    for (i, j), dist in molecule.constraints.distance.items():
        print("{ B", i, j, dist, "C }", file=inp_file)
    print("    end\nend", file=inp_file)

    return


def print_cartesian_constraints(inp_file, molecule):
    """Print the Cartesian constraints to the input file"""

    if molecule.constraints.cartesian is None:
        return

    print("%geom Constraints", file=inp_file)
    for i in molecule.constraints.cartesian:
        print("{ C", i, "C }", file=inp_file)
    print("    end\nend", file=inp_file)

    return


def print_num_optimisation_steps(inp_file, molecule, calc_input):
    """If there are relatively few atoms increase the number of opt steps"""

    if not isinstance(calc_input.keywords, kws.OptKeywords):
        return  # Not an optimisation so no need to increase steps

    if calc_input.keywords.max_opt_cycles is not None:
        print(
            f"%geom MaxIter {int(calc_input.keywords.max_opt_cycles)} end",
            file=inp_file,
        )
        return

    if molecule.n_atoms > 33:
        return  # Use default behaviour

    return


def print_point_charges(inp_file, calc_input):
    """Print a point charge file and add the name to the input file"""

    if calc_input.point_charges is None:
        return

    filename = calc_input.filename.replace(".inp", ".pc")
    with open(filename, "w") as pc_file:
        print(len(calc_input.point_charges), file=pc_file)
        for pc in calc_input.point_charges:
            x, y, z = pc.coord
            print(
                f"{pc.charge:^12.8f} {x:^12.8f} {y:^12.8f} {z:^12.8f}",
                file=pc_file,
            )

    calc_input.additional_filenames.append(filename)

    print(f'% pointcharges "{filename}"', file=inp_file)
    return


def print_default_params(inp_file):
    """Print some useful default parameters to the input file"""

    print(
        "%output \nxyzfile=True \nend ",
        "%scf \nmaxiter 250 \nend",
        "%output\nPrint[P_Hirshfeld] = 1\nend",
        "% maxcore",
        int(Config.max_core.to("MB")),
        sep="\n",
        file=inp_file,
    )
    return


def print_coordinates(inp_file, molecule):
    """Print the coordinates to the input file in the correct format"""

    print("*xyz", molecule.charge, molecule.mult, file=inp_file)
    for atom in molecule.atoms:
        x, y, z = atom.coord
        print(
            f"{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}", file=inp_file
        )
    print("*", file=inp_file)

    return


class ORCA(autode.wrappers.methods.ExternalMethodOEGH):
    def __init__(self):
        super().__init__(
            executable_name="orca",
            path=Config.ORCA.path,
            keywords_set=Config.ORCA.keywords,
            implicit_solvation_type=Config.ORCA.implicit_solvation_type,
            doi_list=["10.1002/wcms.81", "10.1002/wcms.1327"],
        )

    def __repr__(self):
        return f"ORCA(available = {self.is_available})"

    def generate_input_for(self, calc: "CalculationExecutor") -> None:
        assert calc.input.filename is not None

        keywords = self.get_keywords(calc.input, calc.molecule)
        assert len(keywords) > 0

        with open(calc.input.filename, "w") as inp_file:
            print("!", *keywords, file=inp_file)

            self.print_solvent(inp_file, calc.molecule, keywords)
            print_added_internals(inp_file, calc.input)
            print_distance_constraints(inp_file, calc.molecule)
            print_cartesian_constraints(inp_file, calc.molecule)
            print_num_optimisation_steps(inp_file, calc.molecule, calc.input)
            print_point_charges(inp_file, calc.input)
            print_default_params(inp_file)

            if calc.n_cores > 1:
                print(f"%pal nprocs {calc.n_cores}\nend", file=inp_file)

            print_coordinates(inp_file, calc.molecule)

        return None

    @staticmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.inp"

    @staticmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.out"

    def version_in(self, calc: "CalculationExecutor") -> str:
        """Get the version of ORCA used to execute this calculation"""

        if not calc.output.exists:
            return self._get_version_no_output()

        for line in calc.output.file_lines:
            if "Program Version" in line and len(line.split()) >= 3:
                return line.split()[2]

        logger.warning("Could not find the ORCA version number")
        return "???"

    def execute(self, calc):
        @work_in_tmp_dir(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=Config.ORCA.copied_output_exts,
        )
        def execute_orca():
            run_external(
                params=[calc.method.path, calc.input.filename],
                output_filename=calc.output.filename,
            )

        execute_orca()
        return None

    def optimiser_from(self, calc: "CalculationExecutor") -> "BaseOptimiser":
        return ORCAOptimiser(output_lines=calc.output.file_lines)

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        termination_strings = [
            "$end",  # at the end of a .hess file
            "ORCA TERMINATED NORMALLY",
            "The optimization did not converge",
        ]

        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if any(substring in line for substring in termination_strings):
                logger.info("orca terminated normally")
                return True

            if n_line > 30:
                # The above lines are pretty close to the end of the file –
                # so skip parsing it all
                return False

        return False

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        assert calc.output.filename is not None, "Must have a set output"

        if calc.output.filename.endswith(".hess"):
            logger.warning("Failed to set the potential energy")
            return PotentialEnergy(0.0)

        for line in reversed(calc.output.file_lines):
            if "FINAL SINGLE POINT ENERGY" in line:
                return PotentialEnergy(line.split()[4], units="Ha")

        raise CouldNotGetProperty(name="energy")

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        assert calc.output.filename is not None, "Must have a set output"

        fn_ext = ".hess" if calc.output.filename.endswith(".hess") else ".out"

        # First try the .xyz file generated
        xyz_file_name = calc.output.filename.replace(fn_ext, ".xyz")
        if os.path.exists(xyz_file_name):
            try:
                return xyz_file_to_atoms(xyz_file_name).coordinates

            except XYZfileWrongFormat:
                raise AtomsNotFound(f"Failed to parse {xyz_file_name}")

        # Then the Hessian file
        hess_file_name = calc.output.filename.replace(fn_ext, ".hess")
        if os.path.exists(hess_file_name):
            hess_file_lines = open(hess_file_name, "r").readlines()

            coords = []
            for i, line in enumerate(hess_file_lines):
                if "$atoms" not in line:
                    continue

                for aline in hess_file_lines[
                    i + 2 : i + 2 + calc.molecule.n_atoms
                ]:
                    _, _, x, y, z = aline.split()
                    coords.append([float(x), float(y), float(z)])

                return Coordinates(coords, units="a0").to("Å")

        # and finally the potentially long .out file
        if os.path.exists(calc.output.filename) and fn_ext == ".out":
            coords = []

            # There could be many sets in the file, so take the last
            for i, line in enumerate(calc.output.file_lines):
                if "CARTESIAN COORDINATES (ANGSTROEM)" not in line:
                    continue

                coords, n_atoms = [], calc.molecule.n_atoms
                for oline in calc.output.file_lines[i + 2 : i + 2 + n_atoms]:
                    _, x, y, z = oline.split()
                    coords.append([float(x), float(y), float(z)])

            return Coordinates(coords, units="Å")

        raise NoCalculationOutput("Failed to find any ORCA output files")

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        """
         e.g.

        .HIRSHFELD ANALYSIS
         ------------------

         Total integrated alpha density =     12.997461186
         Total integrated beta density  =     12.997461186

           ATOM     CHARGE      SPIN
            0 C   -0.006954    0.000000
            . .      .            .
        """
        charges: List[float] = []

        for i, line in enumerate(calc.output.file_lines):
            if "HIRSHFELD ANALYSIS" in line:
                charges = []
                first, last = i + 7, i + 7 + calc.molecule.n_atoms
                for charge_line in calc.output.file_lines[first:last]:
                    charges.append(float(charge_line.split()[-2]))

        return charges

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        """
        e.g.

        #------------------
        CARTESIAN GRADIENT                                            <- i
        #------------------

           1   C   :   -0.011390275   -0.000447412    0.000552736    <- j
        """
        gradients: List[List[float]] = []

        for i, line in enumerate(calc.output.file_lines):
            if (
                "CARTESIAN GRADIENT" in line
                or "The final MP2 gradient" in line
            ):
                gradients = []
                if "CARTESIAN GRADIENT" in line:
                    first, last = i + 3, i + 3 + calc.molecule.n_atoms
                if "The final MP2 gradient" in line:
                    first, last = i + 1, i + 1 + calc.molecule.n_atoms
                if "CARTESIAN GRADIENT (NUMERICAL)" in line:
                    first, last = i + 2, i + 2 + calc.molecule.n_atoms

                for grad_line in calc.output.file_lines[first:last]:
                    if len(grad_line.split()) <= 3:
                        continue

                    dadx, dady, dadz = grad_line.split()[-3:]
                    gradients.append([float(dadx), float(dady), float(dadz)])

        return Gradient(gradients, units="Ha a0^-1").to("Ha Å^-1")

    @staticmethod
    def _start_line_hessian(calc, file_lines):
        """
        Find the line where the Hessian starts in an ORCA Hessian file
        e.g. H2O.hess

        Arguments:
            calc (autode.calculation.Calculation):
            file_lines (list(str)):

        Returns:
            (int):

        Raises:
            (autode.exceptions.CouldNotGetProperty | AssertionError):
        """

        for i, line in enumerate(file_lines):
            if "$hessian" not in line:
                continue

            # Ensure the number of atoms is present, and is the number expected
            n_atoms = int(file_lines[i + 1].split()[0]) // 3
            assert n_atoms == calc.molecule.n_atoms
            return i + 3

        raise CouldNotGetProperty(f"No Hessian found in the Hessian file")

    def hessian_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> Hessian:
        """Grab the Hessian from the output .hess file

        e.g.::

            $hessian
            9
                        0         1
                               2          3            4
            0      6.48E-01   4.376E-03   2.411E-09  -3.266E-01  -2.5184E-01
            .         .          .           .           .           .
        """
        assert calc.input.keywords is not None, "Must have keywords"

        assert calc.output.filename is not None, "Output filename must be set"
        hess_filename = calc.output.filename

        if calc.output.filename.endswith(".out"):
            hess_filename = calc.output.filename.replace(".out", ".hess")

        if not os.path.exists(hess_filename):
            raise CouldNotGetProperty("Could not find Hessian file")

        file_lines = open(hess_filename, "r", encoding="utf-8").readlines()

        hessian_blocks = []
        start_line = self._start_line_hessian(calc, file_lines)

        for j, h_line in enumerate(file_lines[start_line:]):
            if len(h_line.split()) == 0:
                # Assume we're at the end of the Hessian
                break

            # Skip blank lines in the file, marked by one or more fewer items
            # than the previous
            if len(h_line.split()) < len(
                file_lines[start_line + j - 1].split()
            ):
                continue

            # First item is the coordinate number, thus append all others
            hessian_blocks.append([float(v) for v in h_line.split()[1:]])

        n_atoms = calc.molecule.n_atoms
        hessian = [block for block in hessian_blocks[: 3 * n_atoms]]

        for i, block in enumerate(hessian_blocks[3 * n_atoms :]):
            hessian[i % (3 * n_atoms)] += block

        return Hessian(
            np.array(hessian),
            atoms=calc.molecule.atoms,
            functional=calc.input.keywords.functional,
            units="Ha a0^-2",
        ).to("Ha Å^-2")

    @work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
    def _get_version_no_output(self) -> str:
        """
        Get the version of ORCA without an existing output file
        """

        try:
            run_external(
                params=[self.path, "-h"],
                output_filename="tmp",
                stderr_to_log=False,
            )
            line = next(l for l in open("tmp", "r") if "Program Version" in l)
            return line.split()[2]

        except (OSError, IOError, StopIteration):
            return "???"

    def get_keywords(self, calc_input, molecule):
        """Modify the keywords for this calculation with the solvent + fix for
        single atom optimisation calls"""
        kwds_cls = calc_input.keywords.__class__

        new_keywords = kwds_cls()

        for keyword in calc_input.keywords:
            if "scalfreq" in keyword.lower():
                raise UnsupportedCalculationInput(
                    "Frequency scaling within ORCA will not alter the "
                    "calculated frequencies. Use ade.Config.freq_scale_factor"
                )

            if "opt" in keyword.lower() and molecule.n_atoms == 1:
                logger.warning("Can't optimise a single atom")
                continue

            if isinstance(keyword, kws.ECP) and keyword.orca is None:
                # Use the default specification for applying ECPs
                continue

            if isinstance(keyword, kws.MaxOptCycles):
                continue  # Set in print_num_optimisation_steps

            if isinstance(keyword, kws.Keyword):
                new_keywords.append(keyword.orca)

            else:
                new_keywords.append(str(keyword))

        if molecule.solvent is not None:
            self.add_solvent_keyword(molecule, new_keywords)

        # Sort the keywords with all the items with newlines at the end, so
        # the first keyword line is a single contiguous line
        return kwds_cls(
            sorted(new_keywords, key=lambda kw: 1 if "\n" in kw else 0)
        )

    def use_vdw_gaussian_solvent(self, keywords) -> bool:
        """
        Determine if the calculation should use the gaussian charge scheme which
        generally affords better convergence for optimiations in implicit solvent

        Arguments:
            keywords (autode.wrappers.keywords.Keywords):

        Returns:
            (bool):
        """
        assert self.implicit_solvation_type is not None, "Must have a solvent"

        if self.implicit_solvation_type.lower() != "cpcm":
            return False

        if keywords.contain_any_of("freq", "optts") and not self.is_v5:
            logger.warning(
                "Cannot do analytical frequencies with gaussian "
                "charge scheme - switching off"
            )
            return False

        return True

    def add_solvent_keyword(self, molecule, keywords):
        """Add a keyword to the input file based on the solvent"""

        if self.implicit_solvation_type.lower() not in ["smd", "cpcm"]:
            raise UnsupportedCalculationInput(
                "Implicit solvent type must be " "either SMD or CPCM"
            )

        if (
            self.use_vdw_gaussian_solvent(keywords)
            and molecule.solvent.orca not in vdw_gaussian_solvent_dict
        ):
            err = (
                f"CPCM solvent with gaussian charge not available for "
                f"{molecule.solvent.name}. Available solvents are "
                f"{vdw_gaussian_solvent_dict.keys()}"
            )

            raise UnsupportedCalculationInput(message=err)

        solv_name = vdw_gaussian_solvent_dict[molecule.solvent.orca]
        keywords.append(f"CPCM({solv_name})")
        return

    def print_solvent(self, inp_file, molecule, keywords):
        """Add the solvent block to the input file"""

        if molecule.solvent is None:
            return

        if self.implicit_solvation_type.lower() == "smd":
            print(
                f"%cpcm\n"
                f"smd true\n"
                f'SMDsolvent "{molecule.solvent.orca}"\n'
                f"end",
                file=inp_file,
            )

        if self.use_vdw_gaussian_solvent(keywords):
            print("%cpcm\n" "surfacetype vdw_gaussian\n" "end", file=inp_file)
        return

    @property
    def is_v5(self):
        """Is this ORCA version at least 5.0.0?"""
        return self._get_version_no_output()[0] == "5"


class ORCAOptimiser(ExternalOptimiser):
    def __init__(self, output_lines: List[str]):
        self._lines = output_lines

    @property
    def converged(self) -> bool:
        """Has the optimisation converged?"""

        for line in reversed(self._lines):
            if "THE OPTIMIZATION HAS CONVERGED" in line:
                return True

        return False

    @property
    def last_energy_change(self) -> "PotentialEnergy":
        """Find the last energy change in the file"""

        energies = []
        for line in self._lines:
            if "FINAL SINGLE POINT ENERGY" in line:
                energies.append(PotentialEnergy(line.split()[4], units="Ha"))

        if len(energies) < 2:
            return PotentialEnergy(np.inf)

        return energies[-1] - energies[-2]


orca = ORCA()
