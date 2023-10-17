import numpy as np
import autode.wrappers.keywords as kws
import autode.wrappers.methods

from typing import List, TYPE_CHECKING
from copy import deepcopy
from autode.constants import Constants
from autode.utils import run_external
from autode.opt.optimisers.base import ExternalOptimiser
from autode.values import PotentialEnergy, Coordinates, Gradient
from autode.hessians import Hessian
from autode.geom import symm_matrix_from_ltril
from autode.config import Config
from autode.exceptions import AtomsNotFound, CouldNotGetProperty
from autode.log import logger
from autode.constraints import Constraints
from autode.utils import work_in_tmp_dir

if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor
    from autode.opt.optimisers.base import BaseOptimiser


def _add_opt_option(keywords, new_option):
    for keyword in keywords:
        if "opt" not in keyword.lower():
            continue

        opt_options = []
        if "=(" in keyword:
            # get the individual options
            unformated_options = keyword[5:-1].split(",")
            opt_options = [
                option.lower().strip() for option in unformated_options
            ]

        elif "=" in keyword:
            opt_options = [keyword[4:]]

        if not any(op.lower() == new_option.lower() for op in opt_options):
            opt_options.append(new_option)

        new_keyword = f'Opt=({", ".join(opt_options)})'
        keywords.remove(keyword)
        keywords.append(new_keyword)

    return None


def _modify_keywords_for_point_charges(keywords):
    """For a list of Gaussian keywords modify to include z-matrix if not
    already included. Required if point charges are included in the calc"""
    logger.warning("Modifying keywords as point charges are present")

    keywords.append("Charge")
    _add_opt_option(keywords, new_option="Z-Matrix")

    return None


def _n_ecp_elements(keywords, molecule):
    """Number of elements that require an ECP"""

    ecp_kwd = keywords.ecp

    if ecp_kwd is None:
        return 0

    ecp_elems = set(
        atom.label
        for atom in molecule.atoms
        if atom.atomic_number >= ecp_kwd.min_atomic_number
    )

    return len(ecp_elems)


def _get_keywords(calc_input, molecule):
    """Modify the input keywords to try and fix some Gaussian's quirks"""

    new_keywords = []  # List of keywords as strings for this calculation

    for keyword in calc_input.keywords.copy():
        # Replace the basis set file specification with genecp
        if str(keyword).endswith(".gbs"):
            logger.info("Found a custom basis set file adding genecp")
            new_keywords.append("genecp")
            continue

        if (
            isinstance(keyword, kws.BasisSet)
            and _n_ecp_elements(calc_input.keywords, molecule) > 0
        ):
            logger.info("Required and ECP so will print a custom basis set")
            new_keywords.append("genecp")
            continue

        elif isinstance(keyword, kws.ECP):
            # ECPs are dealt with in a custom file
            continue

        if isinstance(keyword, kws.MaxOptCycles):
            continue  # Handled after the full set of keywords is set

        elif isinstance(keyword, kws.Keyword):
            kwd_str = keyword.g09 if getattr(keyword, "g09") else keyword.g16

            # Add any empirical dispersion
            if isinstance(keyword, kws.DispersionCorrection):
                new_keywords.append(f"EmpiricalDispersion={kwd_str}")

            # and any other keywords, that may be a Keyword with a g09/g16
            # attribute or just a name
            else:
                new_keywords.append(kwd_str)

        else:
            new_keywords.append(str(keyword))

    # Mod redundant keywords is required if there are any constraints or
    # modified internal coordinates
    if molecule.constraints.any:
        new_keywords.append("Geom=ModRedun")

    if calc_input.added_internals is not None:
        new_keywords.append("Geom=ModRedun")

    # Remove the optimisation keyword if there is only a single atom
    opt = False
    for keyword in new_keywords:
        if "opt" not in keyword.lower():
            continue

        opt = True

        if molecule.n_atoms == 1:
            logger.warning("Cannot do an optimisation for a single atom")
            new_keywords.remove(keyword)

    # Further modification is required if there are surrounding point charges
    if calc_input.point_charges is not None:
        _modify_keywords_for_point_charges(new_keywords)

    if isinstance(calc_input.keywords, kws.OptKeywords):
        max_cycles = calc_input.keywords.max_opt_cycles

        if max_cycles is not None:
            _add_opt_option(new_keywords, f"MaxCycles={int(max_cycles)}")

    # By default perform all optimisations without symmetry
    if opt and not any(kw.lower() == "nosymm" for kw in new_keywords):
        if hasattr(molecule, "is_linear") and molecule.is_linear():
            # Allow symmetry for linear molecules so the free energy
            # calculation doesn't fail
            pass
        else:
            new_keywords.append("NoSymm")

    return new_keywords


def _print_point_charges(inp_file, calc_input):
    """Add point charges to the input file"""

    if calc_input.point_charges is None:
        return

    print("\n", end="", file=inp_file)
    for point_charge in calc_input.point_charges:
        x, y, z = point_charge.coord
        print(
            f"{x:^12.8f} {y:^12.8f} {z:^12.8f} {point_charge.charge:^12.8f}",
            file=inp_file,
        )
    return


def _print_added_internals(inp_file, calc_input):
    """Add any internal coordinates to the input file"""

    if calc_input.added_internals is None:
        return

    for i, j in calc_input.added_internals:
        # Gaussian indexes atoms from 1
        print("B", i + 1, j + 1, file=inp_file)

    return


def _print_constraints(inp_file, molecule):
    """Add any distance or cartesian constraints to the input file"""

    if molecule.constraints.distance is not None:
        for (i, j), dist in molecule.constraints.distance.items():
            # Gaussian indexes atoms from 1
            print("B", i + 1, j + 1, dist, "B", file=inp_file)
            print("B", i + 1, j + 1, "F", file=inp_file)

    if molecule.constraints.cartesian is not None:
        for i in molecule.constraints.cartesian:
            # Gaussian indexes atoms from 1
            print("X", i + 1, "F", file=inp_file)
    return


def _print_custom_basis(inp_file, calc_input, molecule):
    """Print the definition of the custom basis set file"""
    keywords = calc_input.keywords

    for keyword in keywords:
        if isinstance(keyword, kws.Keyword) and getattr(keyword, "g09"):
            str_keyword = keyword.g09
        else:
            str_keyword = str(keyword)

        if str_keyword.endswith(".gbs"):
            print(f"@{keyword}", file=inp_file)
            return

    if _n_ecp_elements(keywords, molecule) == 0:
        return

    # Must need a custom basis set file because there are ECPs to print
    ecp_kwd, basis_kwd = keywords.ecp, keywords.basis_set

    if ecp_kwd is None or basis_kwd is None:
        raise RuntimeError(
            "Expecting to print a custom basis set file with "
            "both a basis set and an ECP"
        )

    ecp_elems = set(
        atom.label
        for atom in molecule.atoms
        if atom.atomic_number >= ecp_kwd.min_atomic_number
    )

    other_elems = set(
        atom.label for atom in molecule.atoms if atom.label not in ecp_elems
    )

    print("@basis.gbs", file=inp_file)

    # Keyword strings that could be defined as either G09 or G16
    ecp_str = ecp_kwd.g09 if getattr(ecp_kwd, "g09") else ecp_kwd.g16
    basis_str = basis_kwd.g09 if getattr(basis_kwd, "g09") else basis_kwd.g16

    with open("basis.gbs", "w") as basis_file:
        if len(other_elems) > 0:
            print(*other_elems, "0", file=basis_file)
            print(f"{basis_str}", "****", sep="\n", file=basis_file)

        print(*ecp_elems, "0", file=basis_file)
        print(
            f"{ecp_str}",
            "****",
            "",
            " ".join(ecp_elems) + " 0",
            f"{ecp_str}",
            sep="\n",
            file=basis_file,
        )

    calc_input.additional_filenames.append("basis.gbs")
    return None


def _rerun_angle_failure(calc):
    """
    Gaussian will sometimes encounter a 180 degree angle and crash. This
    function performs a few geometry optimisation cycles in cartesian
    coordinates then switches back to internals

    Arguments:
        calc (autode.calculation.Calculation):

    Returns:
        (autode.calculation.Calculation):
    """
    from autode.calculations import CalculationOutput

    cart_calc = deepcopy(calc)

    # Iterate through a copied set of keywords
    for keyword in cart_calc.input.keywords.copy():
        if keyword.lower().startswith("geom"):
            cart_calc.input.keywords.remove(keyword)

        elif keyword.lower().startswith("opt"):
            options = []
            if "=(" in keyword:
                # get the individual options
                options = [
                    option.lower().strip()
                    for option in keyword[5:-1].split(",")
                ]

                for option in options:
                    if option.startswith("maxcycles") or option.startswith(
                        "maxstep"
                    ):
                        options.remove(option)

            elif "=" in keyword:
                options = [keyword[4:]]
            options += ["maxcycles=3", "maxstep=1", "cartesian"]

            new_keyword = f'Opt=({", ".join(options)})'
            cart_calc.input.keywords.remove(keyword)
            cart_calc.input.keywords.append(new_keyword)

    # Generate the new calculation and run
    cart_calc.name += "_cartesian"
    cart_calc.molecule.constraints = Constraints(distance=None, cartesian=None)
    cart_calc.molecule.reset_graph()
    cart_calc.output = CalculationOutput()
    cart_calc.run()

    if not cart_calc.terminated_normally:
        logger.warning("Cartesian calculation did not converge")
        return None

    logger.info("Returning to internal coordinates")

    # Reset the required parameters for the new calculation
    fixed_calc = deepcopy(calc)
    fixed_calc.name += "_internal"
    fixed_calc.output = CalculationOutput()
    fixed_calc.run()

    return fixed_calc


def _run_hessian(calc):
    """
    Run a hessian calculation based on a corresponding optimisation or
    single point calculation. Used for when an external force driver is used
    and the dummy basis set does not carry over to the frequency calculation.
    See: https://github.com/duartegroup/autodE/pull/86

    Arguments:
        calc (autode.calculation.Calculation):

    Returns:
        (autode.calculation.Calculation):
    """
    from autode.calculations import CalculationOutput

    hess_calc = deepcopy(calc)  # Uses a copy so the current calc. is unchanged

    # Remove any optimisation keywords
    for keyword in filter(
        lambda kwd: "opt" in kwd.lower(), hess_calc.input.keywords
    ):
        hess_calc.input.keywords.remove(keyword)

    # Add Geom(Redundant) to be compatible with External
    hess_calc.input.keywords.append("Freq Geom(Redundant)")

    # Generate the new calculation and run
    hess_calc.name += "_hess"
    hess_calc.molecule.constraints = Constraints(distance=None, cartesian=None)
    hess_calc.output = CalculationOutput()
    hess_calc.run()

    return hess_calc


def _freq_in_keywords(calc):
    """Is 'Freq' in a a set of keywords used to run a calculation"""
    return any("freq" in keyword.lower() for keyword in calc.input.keywords)


def _calc_uses_external_method(calc):
    """Does this Gaussian calculation use an external force driver?"""
    return any("external" in kwd.lower() for kwd in calc.input.keywords)


class G09(autode.wrappers.methods.ExternalMethodOEGH):
    def __init__(
        self,
        executable_name="g09",
        path=None,
        keywords_set=None,
        implicit_solvation_type=None,
    ):
        """Gaussian 09"""

        if keywords_set is None:
            keywords_set = Config.G09.keywords

        if implicit_solvation_type is None:
            implicit_solvation_type = Config.G09.implicit_solvation_type

        super().__init__(
            executable_name=executable_name,
            path=Config.G09.path if path is None else path,
            keywords_set=keywords_set,
            implicit_solvation_type=implicit_solvation_type,
            doi_list=["http://gaussian.com/citation/"],
        )

    def __repr__(self):
        return f"Gaussian09(available = {self.is_available})"

    def generate_input_for(self, calc) -> None:
        """Print a Gaussian input file"""
        molecule = calc.molecule

        with open(calc.input.filename, "w") as inp_file:
            # Gaussian defines the total memory for the whole calculation, not
            # per core
            total_mem = int(Config.max_core.to("MB") * calc.n_cores)
            print(f"%mem={total_mem}MB", file=inp_file)

            if calc.n_cores > 1:
                print(f"%nprocshared={calc.n_cores}", file=inp_file)

            keywords = _get_keywords(calc.input, molecule)
            print("#", *keywords, file=inp_file, end=" ")

            if molecule.solvent is not None:
                print(
                    f"scrf=(smd,solvent={molecule.solvent.g09})", file=inp_file
                )
            else:
                print("", file=inp_file)

            print(f"\n {calc.name}\n", file=inp_file)
            print(molecule.charge, molecule.mult, file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(
                    f"{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}",
                    file=inp_file,
                )

            _print_point_charges(inp_file, calc.input)
            print("", file=inp_file)
            _print_added_internals(inp_file, calc.input)
            _print_constraints(inp_file, molecule)

            if molecule.constraints.any or calc.input.added_internals:
                print("", file=inp_file)  # needs an extra blank line
            _print_custom_basis(inp_file, calc.input, molecule)

            # Gaussian needs blank lines at the end of the file
            print("\n", file=inp_file)

        return None

    @staticmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.com"

    @staticmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.log"

    def version_in(self, calc) -> str:
        """Get the version of Gaussian used in this calculation"""

        for line in calc.output.file_lines:
            if line.startswith("Gaussian ") and "Revision" in line:
                return line.lstrip("Gaussian ")

        logger.warning("Could not find the Gaussian version number")
        return "???"

    def execute(self, calc) -> None:
        @work_in_tmp_dir(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=(".log", ".com", ".gbs"),
        )
        def execute_g09():
            run_external(
                params=[calc.method.path, calc.input.filename],
                output_filename=calc.output.filename,
            )

        execute_g09()
        return None

    def terminated_normally_in(self, calc, rerun_if_failed=True):
        termination_strings = [
            "Normal termination of Gaussian",
            "Number of steps exceeded",
        ]

        bend_ok = True  # Gaussian can fail when 180º bends are encountered
        for line in reversed(calc.output.file_lines):
            if any(string in line for string in termination_strings):
                logger.info("Gaussian terminated normally")
                return True

            if "Bend failed for angle" in line:
                logger.warning("Gaussian encountered a 180° angle and crashed")
                bend_ok = False
                break

        if bend_ok or not rerun_if_failed:
            return False

        # Set a limit on the amount of times we do this
        if calc.name.endswith("internal_internal_internal_internal"):
            return False

        try:
            # To fix the calculation requires the atoms to be in the output
            fixed_calc = _rerun_angle_failure(calc)

        except AtomsNotFound:
            return False

        if fixed_calc is not None and fixed_calc.terminated_normally:
            logger.info("The 180° angle issue has been fixed")
            calc.output = fixed_calc.output
            calc.name = fixed_calc.name
            return True

        return False

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        for line in reversed(calc.output.file_lines):
            if "SCF Done" in line or "E(CIS)" in line:
                return PotentialEnergy((line.split()[4]), units="Ha")

            if "E(CORR)" in line or "E(CI)" in line:
                return PotentialEnergy(line.split()[3], units="Ha")

            if "E(CIS(D))" in line:
                return PotentialEnergy(line.split()[5], units="Ha")

            if line.startswith(" Energy=") and "NIter=" in line:
                return PotentialEnergy(line.split()[1], units="Ha")

        raise CouldNotGetProperty(name="energy")

    def optimiser_from(self, calc: "CalculationExecutor") -> "BaseOptimiser":
        return G09Optimiser(output_lines=calc.output.file_lines)

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        """Get the final set of coordinates from a G09 output"""
        coords: List[List[float]] = []

        for i, line in enumerate(calc.output.file_lines):
            if "Input orientation" in line:
                coords.clear()
                xyz_lines = calc.output.file_lines[
                    i + 5 : i + 5 + calc.molecule.n_atoms
                ]

                for xyz_line in xyz_lines:
                    _, _, _, x, y, z = xyz_line.split()
                    coords.append([float(x), float(y), float(z)])

        return Coordinates(coords, units="Å")

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        charges_section = False
        charges: List[float] = []
        for line in reversed(calc.output.file_lines):
            if "sum of mulliken charges" in line.lower():
                charges_section = True

            if len(charges) == calc.molecule.n_atoms:
                return list(reversed(charges))

            if charges_section and len(line.split()) == 3:
                charges.append(float(line.split()[2]))

        logger.error("Something went wrong finding the atomic charges")
        return charges

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        """
        Get gradients from a Gaussian output file in the format


        -------------------------------------------------------------------
         Center     Atomic                   Forces (Hartrees/Bohr)
         Number     Number              X              Y              Z
         -------------------------------------------------------------------
          1        6          -0.000205102    0.000074692    0.000073625
          .        .                .              .               .
        """
        n_atoms = calc.molecule.n_atoms
        raw_gradient: List[np.ndarray] = []

        for i, line in enumerate(calc.output.file_lines):
            if "Forces (Hartrees/Bohr)" not in line:
                continue

            raw_gradient = []  # NOTE: possibly multiple gradients in a file

            for force_line in calc.output.file_lines[i + 3 : i + 3 + n_atoms]:
                try:
                    _, _, fx, fy, fz = force_line.split()
                    force = np.array([float(fx), float(fy), float(fz)])

                    grad = -force / Constants.a0_to_ang
                    raw_gradient.append(grad)

                except ValueError:
                    logger.warning("Failed to set gradient line")

        return Gradient(raw_gradient, units="Ha a0^-1").to("Ha Å^-1")

    def hessian_from(
        self, calc: "autode.calculations.executors.CalculationExecutor"
    ) -> Hessian:
        r"""
        Extract the Hessian from a Gaussian09 calculation, which is printed as
        just the lower triangular portion but is symmetric so the full 3Nx3N
        matrix can be re-constructed. Read it from the final output block
        sandwiched between 1\1\ ..... \\@

        Arguments:
            calc (autode.calculation.Calculation):

        Returns:
            (autode.hessians.Hessian):

        Raises:
            (IndexError | ValueError):
        """
        assert calc.input.keywords is not None, "Must have keywords"

        if _calc_uses_external_method(calc) and not _freq_in_keywords(calc):
            # Using external force drivers can lead to failed Hessian calcs.
            calc = _run_hessian(calc)

        hess_lines = []
        append_line = False

        for line in reversed(calc.output.file_lines):
            if (
                r"\\@" in line
                or line.startswith(" @")
                or line.startswith(r" \@")
            ):
                append_line = True

            if append_line:
                #                 Strip off new-lines and spaces
                hess_lines.append(line.strip("\n").strip(" "))

            if "NImag" in line:
                break

        r"""
        For a block with the format:

        ...[C*(O1C1O1)]\NImag=0\\H_x1x1, H_y1x1, ...\\
        F_x1, F_y1, ...\\\@

        get the elements of the Hessian, noting that the lines have been
        parsed backwards, hence the [::-1]
        """

        hess_str = "".join(hess_lines[::-1]).split(r"\\")[-3]
        hess_values = [float(val) for val in hess_str.split(",")]

        n = 3 * calc.molecule.n_atoms

        if len(hess_values) != n * (n + 1) // 2:
            raise CouldNotGetProperty(
                "Not enough elements of the Hessian " "matrix found"
            )

        return Hessian(
            symm_matrix_from_ltril(hess_values),
            atoms=self.atoms_from(calc),
            functional=calc.input.keywords.functional,
            units="Ha a0^-2",
        ).to("Ha Å^-2")


class G09Optimiser(ExternalOptimiser):
    def __init__(self, output_lines: List[str]):
        self._lines = output_lines

    @property
    def converged(self) -> bool:
        """Has the optimisation converged?"""

        for line in reversed(self._lines):
            if "Optimization completed" in line:
                return True

        return False

    @property
    def last_energy_change(self) -> "PotentialEnergy":
        """Find the last energy change in the file"""

        energies = []
        for line in self._lines:
            if "SCF Done" in line or "E(CIS)" in line:
                energy_str = line.split()[4]
                energies.append(PotentialEnergy(energy_str, units="Ha"))

        if len(energies) < 2:
            return PotentialEnergy(np.inf)

        return energies[-1] - energies[-2]


g09 = G09()
