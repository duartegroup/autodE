import numpy as np
import autode.wrappers.keywords as kws
import autode.wrappers.methods

from typing import TYPE_CHECKING, List

from autode.utils import run_external_monitored
from autode.values import PotentialEnergy, Gradient, Coordinates
from autode.hessians import Hessian
from autode.geom import symm_matrix_from_ltril
from autode.config import Config
from autode.exceptions import UnsupportedCalculationInput, CouldNotGetProperty
from autode.log import logger
from autode.utils import work_in_tmp_dir


if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor


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
        return ""  # No ECP is defined in these keywords

    # Set of unique atomic symbols that require an ECP
    ecp_elems = set(
        atom.label
        for atom in molecule.atoms
        if atom.atomic_number >= ecp_kwd.min_atomic_number
    )

    if len(ecp_elems) == 0:
        return ""  # No atoms require an ECP

    ecp_str = "\necp\n"
    ecp_str += "\n".join(
        f"  {label}   library  {ecp_kwd.nwchem}" for label in ecp_elems
    )
    ecp_str += "\nend"

    return ecp_str


def get_keywords(calc_input, molecule):
    """Generate a keywords list and adding solvent"""

    new_keywords = []

    for keyword in calc_input.keywords:
        if "scf" in keyword.lower():
            if molecule.solvent is not None:
                raise UnsupportedCalculationInput(
                    "NWChem only supports " "solvent for DFT calcs"
                )

        if isinstance(keyword, kws.Functional):
            keyword = f"dft\n  maxiter 100\n  xc {keyword.nwchem}\nend"

        elif isinstance(keyword, kws.BasisSet):
            keyword = f"basis\n  *   library {keyword.nwchem}\nend"
            keyword += ecp_block(molecule, keywords=calc_input.keywords)

        elif isinstance(keyword, kws.ECP):
            # ECPs are added to the basis block
            continue

        elif isinstance(keyword, kws.MaxOptCycles):
            continue  # Maximum number of optimisation cycles in driver block

        elif isinstance(keyword, kws.Keyword):
            keyword = keyword.nwchem

        if "opt" in keyword.lower() and molecule.n_atoms == 1:
            logger.warning("Cannot do an optimisation for a single atom")

            # Replace any 'opt' containing word in this keyword with energy
            words = []
            for word in keyword.split():
                if "opt" in word:
                    words.append("energy")
                else:
                    words.append(word)

            new_keywords.append(" ".join(words))

        elif keyword.lower().startswith("dft"):
            lines = keyword.split("\n")
            lines.insert(1, f"  mult {molecule.mult}")
            new_keyword = "\n".join(lines)
            new_keywords.append(new_keyword)

        elif keyword.lower().startswith("scf"):
            if not any("nopen" in kw for kw in new_keywords):
                lines = keyword.split("\n")
                lines.insert(1, f"  nopen {molecule.mult - 1}")
                new_keywords.append("\n".join(lines))

        elif "driver" in keyword.lower() and isinstance(
            calc_input.keywords, kws.OptKeywords
        ):
            raise UnsupportedCalculationInput(
                f"NWChem uses autodE implemented optimisers. {keyword} will "
                f"be unused"
            )

        else:
            new_keywords.append(keyword)

    if any("task scf" in kw.lower() for kw in new_keywords) and not any(
        "nopen" in kw.lower() for kw in new_keywords
    ):
        # Need to set the spin state
        new_keywords.insert(1, f"scf\n    nopen {molecule.mult - 1}\nend")

    return new_keywords


class NWChem(autode.wrappers.methods.ExternalMethodEGH):
    def __init__(self):
        super().__init__(
            executable_name="nwchem",
            path=Config.NWChem.path,
            keywords_set=Config.NWChem.keywords,
            implicit_solvation_type=Config.NWChem.implicit_solvation_type,
            doi_list=["10.1063/5.0004997"],
        )

    def __repr__(self):
        return f"NWChem(available = {self.is_available})"

    def generate_input_for(self, calc: "CalculationExecutor") -> None:
        assert calc.input.filename is not None, "Must have an input filename"
        molecule = calc.molecule
        keywords = get_keywords(calc.input, molecule)

        with open(calc.input.filename, "w") as inp_file:
            print(f"start {calc.name}\necho", file=inp_file)

            if calc.molecule.solvent is not None:
                print(
                    f"cosmo\n "
                    f"do_cosmo_smd true\n "
                    f"solvent {calc.molecule.solvent.nwchem}\n"
                    f"end",
                    file=inp_file,
                )

            print("geometry noautosym", end=" ", file=inp_file)
            if molecule.constraints.distance or molecule.constraints.cartesian:
                print("noautoz", file=inp_file)
            else:
                print("", file=inp_file)

            for atom in molecule.atoms:
                x, y, z = atom.coord
                print(
                    f"{atom.label:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}",
                    file=inp_file,
                )

            print("end", file=inp_file)

            print(f"charge {calc.molecule.charge}", file=inp_file)

            if calc.input.point_charges is not None:
                print("bq", file=inp_file)
                for pc in calc.input.point_charges:
                    x, y, z = pc.coord
                    print(
                        f"{x:^12.8f} {y:^12.8f} {z:^12.8f} {pc.charge:^12.8f}",
                        file=inp_file,
                    )
                print("end", file=inp_file)

            print(f'memory {int(Config.max_core.to("MB"))} mb', file=inp_file)

            print(*keywords, sep="\n", file=inp_file)

            # Will used partial an ESP initialisation to generate partial
            # atomic charges - more accurate than the standard Mulliken
            # analysis (or at least less sensitive to the method)
            print("task esp", file=inp_file)

        return None

    @staticmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.nw"

    @staticmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.out"

    def version_in(self, calc: "CalculationExecutor") -> str:
        """Get the NWChem version from the output file"""
        for line in calc.output.file_lines:
            if "(NWChem)" in line:
                # e.g. Northwest Computational Chemistry Package (NWChem) 6.6
                return line.split()[-1]

        logger.warning("Could not find the NWChem version")
        return "???"

    def execute(self, calc: "CalculationExecutor"):
        @work_in_tmp_dir(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=(".nw", ".out"),
        )
        def execute_nwchem():
            params = [
                "mpirun",
                "-np",
                str(calc.n_cores),
                calc.method.path,
                calc.input.filename,
            ]

            run_external_monitored(
                params,
                calc.output.filename,
                break_words=["Received an Error", "MPI_ABORT"],
            )

        execute_nwchem()
        return None

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        for n_line, line in enumerate(reversed(calc.output.file_lines)):
            if any(
                substring in line
                for substring in [
                    "CITATION",
                    "Failed to converge in maximum number of steps or available time",
                ]
            ):
                logger.info("nwchem terminated normally")
                return True
            if "MPI_ABORT" in line:
                return False

            if n_line > 500:
                return False

        return False

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        wf_strings = [
            "Total CCSD energy",
            "Total CCSD(T) energy",
            "Total SCS-MP2 energy",
            "Total MP2 energy",
            "Total RI-MP2 energy",
        ]

        for line in reversed(calc.output.file_lines):
            if any(
                string in line
                for string in ["Total DFT energy", "Total SCF energy"]
            ):
                return PotentialEnergy(line.split()[4], units="Ha")

            if any(string in line for string in wf_strings):
                return PotentialEnergy(line.split()[3], units="Ha")

        raise CouldNotGetProperty(name="energy")

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        xyzs_section = False
        coords: List[List[float]] = []

        for line in calc.output.file_lines:
            if "Output coordinates in angstroms" in line:
                xyzs_section = True
                coords.clear()

            if "Atomic Mass" in line:
                xyzs_section = False

            if xyzs_section and len(line.split()) == 6:
                if line.split()[0].isdigit():
                    _, _, _, x, y, z = line.split()
                    coords.append([float(x), float(y), float(z)])

        return Coordinates(coords, units="Å")

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        """
        e.g.
         Atom              Coordinates                           Charge

                                                  ESP


        1 C    -0.000814    0.000010    0.001095   -0.266058
        . .       .            .            .          .
        """
        charges_section = False
        charges: List[float] = []

        for line in calc.output.file_lines:
            if (
                len(line.split()) == 3
                and "Atom" in line
                and "Coordinates" in line
                and "Charge" in line
            ):
                charges_section = True
                charges.clear()

            if charges_section and len(line.split()) == 6:
                charge = line.split()[-1]
                charges.append(float(charge))

            if charges_section and "------------" in line:
                charges_section = False

        return charges

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        gradients: List[np.ndarray] = []
        n_atoms = calc.molecule.n_atoms

        for i, line in enumerate(calc.output.file_lines):
            if "DFT ENERGY GRADIENTS" not in line:
                continue

            gradients = []

            for grad_line in calc.output.file_lines[i + 4 : i + 4 + n_atoms]:
                x, y, z = grad_line.split()[5:]
                gradients.append(np.array([float(x), float(y), float(z)]))

        return Gradient(gradients, units="Ha a0^-1").to("Ha Å^-1")

    @staticmethod
    def _atom_masses_from_hessian(calc: "CalculationExecutor") -> List[float]:
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
            if "Atom information" not in line:
                continue

            atom_lines = file_lines[-i + 2 : -i + 2 + n_atoms]
            break

        if atom_lines is None:
            raise CouldNotGetProperty("No masses found in output file")

        # Replace double notation for standard 'E' and float all the final
        # entries, which should be the masses in amu
        return [
            float(line.split()[-1].replace("D", "E")) for line in atom_lines
        ]

    def hessian_from(self, calc: "CalculationExecutor") -> Hessian:
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
        logger.info(f"Attempting to set the Hessian from {calc.name}")

        try:
            line_idx = next(
                i
                for i, line in enumerate(calc.output.file_lines)
                if "MASS-WEIGHTED NUCLEAR HESSIAN" in line
            )
        except StopIteration:
            raise CouldNotGetProperty("Hessian not found in the output file")

        hess_lines: List[List[float]] = [
            [] for _ in range(calc.molecule.n_atoms * 3)
        ]

        for hess_line in calc.output.file_lines[line_idx + 6 :]:
            if "NORMAL MODE EIGENVECTORS" in hess_line:
                break  # Finished the Hessian block

            if "D" in hess_line:
                # e.g.     1    4.50945D-01  ...
                idx = hess_line.split()[0]
                try:
                    _ = hess_lines[int(idx) - 1]
                except (ValueError, IndexError):
                    raise CouldNotGetProperty(
                        "Unexpected hessian formating: " f"{hess_line}"
                    )

                values = [
                    float(x) for x in hess_line.replace("D", "E").split()[1:]
                ]
                hess_lines[int(idx) - 1] += values

        atom_masses = self._atom_masses_from_hessian(calc)
        hess = symm_matrix_from_ltril(array=hess_lines)

        # Un-mass weight from Kamu^-1 to 1
        mass_arr = np.repeat(atom_masses, repeats=3, axis=np.newaxis) * 1e-3
        hess *= np.sqrt(np.outer(mass_arr, mass_arr))

        return Hessian(
            hess,
            atoms=self.atoms_from(calc),
            functional=calc.input.keywords.functional,
            units="Ha a0^-2",
        ).to("Ha Å^-2")


nwchem = NWChem()
