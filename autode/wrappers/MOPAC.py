import os
import numpy as np
import autode.wrappers.keywords as kwds
import autode.wrappers.methods

from typing import List, TYPE_CHECKING

from autode.opt.optimisers.base import ExternalOptimiser
from autode.values import PotentialEnergy, Gradient, Coordinates
from autode.utils import run_external
from autode.exceptions import NotImplementedInMethod
from autode.config import Config
from autode.constants import Constants
from autode.exceptions import UnsupportedCalculationInput
from autode.log import logger
from autode.utils import work_in_tmp_dir
from autode.exceptions import CouldNotGetProperty


if TYPE_CHECKING:
    from autode.calculations.executors import CalculationExecutor
    from autode.opt.optimisers.base import BaseOptimiser


def get_keywords(calc_input, molecule):
    """Get the keywords to use for a MOPAC calculation"""
    # To determine if there is an optimisation or single point the keywords
    # needs to be a subclass of Keywords
    assert isinstance(calc_input.keywords, kwds.Keywords)

    keywords = [
        kwd
        for kwd in calc_input.keywords.copy()
        if not isinstance(kwd, kwds.MaxOptCycles)
    ]

    if isinstance(calc_input.keywords, kwds.SinglePointKeywords):
        # Single point calculation add the 1SCF keyword to prevent opt
        if not any("1scf" in kw.lower() for kw in keywords):
            keywords.append("1SCF")

    if isinstance(calc_input.keywords, kwds.GradientKeywords):
        # Gradient calculation needs GRAD
        if not any("grad" in kw.lower() for kw in keywords):
            keywords.append("GRAD")

        # Gradient calculation add the 1SCF keyword to prevent opt
        if not any("1scf" in kw.lower() for kw in keywords):
            keywords.append("1SCF")

    if calc_input.point_charges is not None:
        keywords.append("QMMM")

    if molecule.solvent is not None:
        if molecule.solvent.dielectric is None:
            err_str = (
                f"Could not use solvent {molecule.solvent} for MOPAC "
                f"calculation, a dielectric constant was not defined"
            )
            raise UnsupportedCalculationInput(message=err_str)

        keywords.append(f"EPS={molecule.solvent.dielectric}")

    # Add the charge and multiplicity
    keywords.append(f"CHARGE={molecule.charge}")

    if "ENPART" not in keywords:
        keywords.append("ENPART")  # Print an energy partition, and also E_tot

    if molecule.mult != 1:
        if molecule.mult == 2:
            keywords.append("DOUBLET")
        elif molecule.mult == 3:
            keywords.append("OPEN(2,2)")
        else:
            logger.critical("Unsupported spin multiplicity")
            raise UnsupportedCalculationInput

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
    atoms = _get_atoms_linear_interp(
        atoms=molecule.atoms, bonds=bonds, final_distances=distances
    )

    # Populate a flat list of atom ids to fix
    fixed_atoms = [i for bond in bonds for i in bond]

    return atoms, fixed_atoms


def print_atoms(inp_file, atoms, fixed_atom_idxs):
    """Print the atoms to the input file depending on whether they are fixed"""

    for i, atom in enumerate(atoms):
        x, y, z = atom.coord

        if i in fixed_atom_idxs:
            line = f"{atom.label:<3}{x:^10.5f} 0 {y:^10.5f} 0 {z:^10.5f} 0"
        else:
            line = f"{atom.label:<3}{x:^10.5f} 1 {y:^10.5f} 1 {z:^10.5f} 1"

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
        for point_charge in calc.input.point_charges:
            # V = q/r_ij
            potential += point_charge.charge / np.linalg.norm(
                coord - point_charge.coord
            )

        # Distance in Å need to be converted to a0 and then the energy
        # Ha e^-1 to kcal mol-1 e^-1
        potentials.append(
            Constants.ha_to_kcalmol * Constants.a0_to_ang * potential
        )

    with open(f"{calc.name}_mol.in", "w") as pc_file:
        print(f"\n{len(atoms)} 0", file=pc_file)

        for potential in potentials:
            print(f"0 0 0 0 {potential}", file=pc_file)

    calc.input.additional_filenames.append(f"{calc.name}_mol.in")
    return


def _get_atoms_linear_interp(
    atoms, bonds, final_distances
) -> "autode.atoms.Atoms":
    """For a geometry defined by a set of xyzs, set the constrained bonds to
    the correct lengths

    ---------------------------------------------------------------------------
    Arguments:
        atoms (list(autode.atoms.Atom)): list of atoms

        bonds (list(tuple)): List of bond ids on for which the final_distances
                             apply
        final_distances (list(float)): List of final bond distances for the
                                       bonds

    Returns:
        (list(autode.atoms.Atom)): Shifted atoms
    """

    coords = np.array([atom.coord for atom in atoms])
    atoms_and_shift_vecs = {}

    for n, bond in enumerate(bonds):
        atom_a, atom_b = bond
        ab_vec = coords[atom_b] - coords[atom_a]
        d_crr = np.linalg.norm(ab_vec)
        d_final = final_distances[n]

        ab_norm_vec = ab_vec / d_crr

        atoms_and_shift_vecs[atom_b] = 0.5 * (d_final - d_crr) * ab_norm_vec
        atoms_and_shift_vecs[atom_a] = -0.5 * (d_final - d_crr) * ab_norm_vec

    for n, coord in enumerate(coords):
        if n in atoms_and_shift_vecs.keys():
            coord += atoms_and_shift_vecs[n]

        atoms[n].coord = coord

    return atoms


class MOPAC(autode.wrappers.methods.ExternalMethodOEG):
    def __init__(self):
        super().__init__(
            executable_name="mopac",
            path=Config.MOPAC.path,
            keywords_set=Config.MOPAC.keywords,
            implicit_solvation_type=Config.MOPAC.implicit_solvation_type,
            doi_list=["10.1007/BF00128336"],
        )

    def __repr__(self):
        return f"MOPAC(available = {self.is_available})"

    def generate_input_for(self, calc: "CalculationExecutor") -> None:
        molecule = calc.molecule
        assert calc.input.filename, "Filename must be defined"

        with open(calc.input.filename, "w") as input_file:
            keywords = get_keywords(calc.input, molecule)
            print(*keywords, "\n\n", file=input_file)

            atoms, fixed_atom_idxs = get_atoms_and_fixed_atom_indexes(molecule)

            if molecule.constraints.cartesian is not None:
                fixed_atom_idxs += molecule.constraints.cartesian

            print_atoms(input_file, atoms, fixed_atom_idxs)
            print_point_charges(calc, atoms)

        return None

    @staticmethod
    def input_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.mop"

    @staticmethod
    def output_filename_for(calc: "CalculationExecutor") -> str:
        return f"{calc.name}.out"

    def version_in(self, calc: "CalculationExecutor") -> str:
        """Get the version of MOPAC used to execute this calculation"""

        for line in calc.output.file_lines:
            if "(Version:" in line and len(line.split()) >= 3:
                # e.g.        MOPAC2016 (Version: 19.144L)

                try:
                    name = line.split()[0]
                    # Item between the brackets with only the version number
                    version = line.split("(")[1].split(")")[0].split()[1]

                    return f"{name} {version}"

                except IndexError:
                    pass

        logger.warning("Could not get the version number from the output file")
        return "???"

    def execute(self, calc):
        @work_in_tmp_dir(
            filenames_to_copy=calc.input.filenames,
            kept_file_exts=(".mop", ".out"),
            use_ll_tmp=True,
        )
        def execute_mopac():
            logger.info(f"Setting the number of OMP threads to {calc.n_cores}")
            os.environ["OMP_NUM_THREADS"] = str(calc.n_cores)
            run_external(
                params=[calc.method.path, calc.input.filename],
                output_filename=calc.output.filename,
            )

        execute_mopac()
        return None

    def terminated_normally_in(self, calc: "CalculationExecutor") -> bool:
        n_errors = 0

        for i, line in enumerate(reversed(calc.output.file_lines)):
            if "Error" in line:
                n_errors += 1

            if i == 100:
                break

        return n_errors == 0

    def _energy_from(self, calc: "CalculationExecutor") -> PotentialEnergy:
        def _energy(x):
            return PotentialEnergy(x, units="eV").to("Ha")

        for line in calc.output.file_lines:
            if "ETOT (EONE + ETWO)" in line:
                return _energy(line.split()[-2])

            if "TOTAL ENERGY" in line:
                return _energy(line.split()[3])

        raise CouldNotGetProperty(name="energy")

    def optimiser_from(self, calc: "CalculationExecutor") -> "BaseOptimiser":
        is_converged = any(
            "GRADIENT" in l and "IS LESS THAN CUTOFF" in l
            for l in reversed(calc.output.file_lines)
        )
        return MOPACOptimiser(converged=is_converged)

    def coordinates_from(self, calc: "CalculationExecutor") -> Coordinates:
        coords: List[List[float]] = []
        n_atoms = calc.molecule.n_atoms

        for i, line in enumerate(calc.output.file_lines):
            if i == len(calc.output.file_lines) - 3:
                # At the end of the file
                break

            line_length = len(calc.output.file_lines[i + 3].split())

            if "CARTESIAN COORDINATES" in line and line_length == 5:
                #                              CARTESIAN COORDINATES
                #
                #    1    C        1.255660629     0.020580974    -0.276235553

                coords = []
                xyz_lines = calc.output.file_lines[i + 2 : i + 2 + n_atoms]
                for xyz_line in xyz_lines:
                    x, y, z = xyz_line.split()[2:]
                    coords.append([float(x), float(y), float(z)])

        return Coordinates(coords, units="Å")

    def partial_charges_from(self, calc: "CalculationExecutor") -> List[float]:
        raise NotImplementedInMethod

    def gradient_from(self, calc: "CalculationExecutor") -> Gradient:
        gradients_section = False
        raw = []
        for line in calc.output.file_lines:
            if "FINAL  POINT  AND  DERIVATIVES" in line:
                gradients_section = True

            if gradients_section and "ATOM   CHEMICAL" in line:
                gradients_section = False

            if gradients_section and len(line.split()) == 8:
                _, _, _, _, _, _, value, _ = line.split()
                try:
                    raw.append(float(value))
                except ValueError:
                    raise CouldNotGetProperty(name="gradients")

        return Gradient(raw, units="kcal mol^-1 Å^-1").to("Ha Å^-1")


class MOPACOptimiser(ExternalOptimiser):
    def __init__(self, converged: bool):
        self._converged = converged

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def last_energy_change(self) -> "PotentialEnergy":
        raise NotImplementedError


mopac = MOPAC()
