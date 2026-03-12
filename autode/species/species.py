import numpy as np
import autode.values as val
from copy import deepcopy
from datetime import date
from typing import Optional, Union, List, Sequence, Any, TypeVar, TYPE_CHECKING

from scipy.spatial import distance_matrix
from autode.log import logger
from autode import methods
from autode.atoms import Atom, Atoms, AtomCollection
from autode.exceptions import SolventUnavailable
from autode.geom import calc_rmsd, get_rot_mat_euler
from autode.constraints import Constraints
from autode.log.methods import methods as method_log
from autode.calculations.types import CalculationType
from autode.conformers.conformers import Conformers
from autode.solvent import get_solvent, Solvent, ExplicitSolvent
from autode.calculations import Calculation
from autode.config import Config
from autode.input_output import atoms_to_xyz_file
from autode.mol_graphs import (
    MolecularGraph,
    make_graph,
    reorder_nodes,
    is_isomorphic,
)
from autode.hessians import Hessian, NumericalHessianCalculator
from autode.units import ha_per_ang_sq, ha_per_ang
from autode.thermochemistry.symmetry import symmetry_number
from autode.thermochemistry.igm import calculate_thermo_cont, LFMethod
from autode.utils import requires_atoms, work_in, requires_conformers
from autode.wrappers.keywords import (
    OptKeywords,
    HessianKeywords,
    GradientKeywords,
    SinglePointKeywords,
)

if TYPE_CHECKING:
    from autode.solvent.solvents import Solvent
    from autode.conformers import Conformer, Conformers
    from autode.wrappers.methods import Method
    from autode.wrappers.keywords import Keywords
    from autode.values import Coordinates

TypeSpecies = TypeVar("TypeSpecies", bound="Species")


class Species(AtomCollection):
    def __init__(
        self,
        name: str,
        atoms: Union[List[Atom], Atoms, None],
        charge: Union[float, int],
        mult: Union[float, int],
        solvent_name: Optional[str] = None,
    ):
        """
        A molecular species. A collection of atoms with a charge and spin
        multiplicity in a solvent (None is gas phase)

        ----------------------------------------------------------------------
        Arguments:
            name (str): Name of the species

            atoms (list(autode.atoms.Atom) | None): List of atoms in the
                                                    species, or None

            charge (int): Charge on the species

            mult (int): Spin multiplicity of the species. 2S+1, where S is the
                        number of unpaired electrons

        Keyword Arguments:
            solvent_name (str | None): Name of the solvent, or None for a
                                       species  in the gas phase
        """
        super().__init__(atoms=atoms)

        self.name = name

        self._charge = int(charge)
        self._mult = int(mult)
        self._solvent = get_solvent(solvent_name, kind="implicit")
        self._graph: Optional[MolecularGraph] = None

        #: All energies calculated at a geometry (autode.values.Energies)
        self.energies = val.Energies()
        self._grad: Optional[val.Gradient] = None
        self._hess: Optional[Hessian] = None

        self._conformers = Conformers()

        self.constraints = Constraints()

    def __str__(self):
        """Unique species identifier"""

        if self.atoms is None:
            atoms_str = ""

        else:
            # Only use the first 100 atoms
            atoms_str = "".join([atom.label for atom in self.atoms[:100]])

        solv_str = self.solvent.name if self.solvent is not None else "none"

        return f"{self.name}_{self.charge}_{self.mult}_{atoms_str}_{solv_str}"

    def _repr(self, prefix: str):
        """Base representation of a Species/Molecule/Complex etc."""

        string = (
            f"{prefix}("
            f"{self.name}, "
            f"n_atoms={self.n_atoms}, "
            f"charge={self.charge}, "
            f"mult={self.mult})"
        )

        return string

    def __repr__(self):
        """Brief representation of this species"""
        return self._repr(prefix="Species")

    def __eq__(self, other) -> bool:
        """
        Equality of this species to another. Only checks based on
        the equality of the strings, which should be semi unique, without
        checks for positional equality
        """
        return str(self) == str(other)

    def copy(self: TypeSpecies) -> TypeSpecies:
        """Copy this whole species"""
        return deepcopy(self)

    def new_species(
        self, name="species", with_constraints: bool = False
    ) -> "Species":
        """
        A new version of this species, identical properties without any
        energies, gradients, hessian, conformers or constraints.

        -----------------------------------------------------------------------
        Arguments:
            name (str): Name of the new species

            with_constraints (bool): Should the constraints from this species be copied
                                    into the new one

        Returns:
            (autode.species.Species):
        """
        species = Species(name, self.atoms.copy(), self.charge, self.mult)
        species.graph = None if self.graph is None else self.graph.copy()
        species.solvent = None if self.solvent is None else self.solvent.copy()

        if with_constraints:
            species.constraints = self.constraints.copy()

        return species

    @property
    def charge(self) -> int:
        """Total charge on this species"""
        return self._charge

    @charge.setter
    def charge(self, value: Any) -> None:
        self._charge = int(value)

    @property
    def mult(self) -> int:
        """Total spin multiplicity on this species (2S + 1)"""
        return self._mult

    @mult.setter
    def mult(self, value: Any) -> None:
        try:
            assert int(value) > 0
        except (ValueError, AssertionError, TypeError):
            raise ValueError(
                f"Failed to set the spin multiplicity to {value}. "
                f"Must be a non-zero positive integer"
            )

        self._mult = int(value)

    @property
    def solvent(self) -> Optional["Solvent"]:
        """
        Solvent which this species is immersed in

        -----------------------------------------------------------------------
        Returns:
            (autode.solvent.Solvent | None): Solvent or None if the species is
                                             in the gas phase
        """
        return self._solvent

    @solvent.setter
    def solvent(self, value: Union["Solvent", str, None]):
        """
        Set the solvent for this species. For a species in the gas phase
        set mol.solvent = None

        -----------------------------------------------------------------------
        Arguments;
            value (autode.solvent.Solvent | str | None):
        """
        if value is None:
            self._solvent = None

        elif type(value) is str:
            self._solvent = get_solvent(solvent_name=value, kind="implicit")

        elif isinstance(value, Solvent):
            self._solvent = value

        else:
            raise SolventUnavailable(
                "Expecting either a string or Solvent, " f"had: {value}"
            )

    @AtomCollection.atoms.setter  # type: ignore[attr-defined]
    def atoms(self, value: Union[List[Atom], Atoms, None]):
        """
        Set the atoms for this species, and reset the energies

        -----------------------------------------------------------------------
        Arguments:
            value (list(autode.atoms.Atom) | None):
        """

        if value is None:
            self._atoms = None
            return

        # If the geometry is identical up to rotations/translations then
        # energies do not need to be changed
        if self.n_atoms == len(value) and all(
            a.label == v.label for a, v in zip(self.atoms, value)
        ):
            self.coordinates = np.array([v.coord for v in value])

        else:
            self._atoms = Atoms(value)
            self._clear_energies_gradient_hessian()

        return

    @AtomCollection.coordinates.setter  # type: ignore[attr-defined]
    def coordinates(self, value: Union["Coordinates", np.ndarray, list]):
        """
        Set the coordinates of this species. If the geometry has changed then
        the energies, gradient and Hessian will be set to None.

        -----------------------------------------------------------------------
        Arguments:
            value: numpy array or nested list of coordinate values
                  (str or float).
        """
        assert self._atoms is not None, "Can't set coordinates without atoms"

        rmsd = calc_rmsd(
            coords1=np.asarray(value).reshape((-1, 3)),  # N x 3
            coords2=self.coordinates,
        )
        if rmsd > 1e-8:
            self._clear_energies_gradient_hessian()

        self._atoms.coordinates = val.Coordinates(value)
        return

    def _clear_energies_gradient_hessian(self) -> None:
        logger.info(f"Geometry changed- resetting energies of {self.name}")
        self.energies.clear()
        self.gradient = None
        self.hessian = None
        return None

    @property
    def graph(self) -> Optional[MolecularGraph]:
        """
        Molecular graph with atoms(V) and bonds(E)

        Note: Graphs are lazily evaluated, i.e. if one has not been generated
        for this species before and it does have atoms then a graph will be
        generated. Subsequent accesses of this property will use the cached
        internal/private self._graph attribute
        """
        if self.atoms is None:
            logger.warning("Had no atoms, so no molecular graph")
            return None

        if self._graph is None:
            make_graph(self)

        return self._graph

    @graph.setter
    def graph(self, value: Optional[MolecularGraph]):
        """Setter for the molecular graph"""
        self._graph = value

    @property
    def formula(self) -> str:
        """
        Molecular formula of this species. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> blank_mol = ade.Molecule()
            >>> blank_mol.formula
            ''
            >>> h2 = ade.Molecule(smiles='[H][H]')
            >>> h2.formula
            'H2'

        -----------------------------------------------------------------------
        Returns:
            (str): Formula
        """

        if self.atoms is None:
            return ""

        symbols = [atom.label for atom in self.atoms]

        formula_str = ""
        for symbol in sorted(set(symbols)):
            num = symbols.count(symbol)
            formula_str += f'{symbol}{num if num > 1 else ""}'

        return formula_str

    @property
    def hessian(self) -> Optional[Hessian]:
        """
        Hessian (d^2E/dx^2) at this geometry (autode.values.Hessian | None)
        shape = (3*n_atoms, 3*n_atoms)
        """
        return self._hess

    @hessian.setter
    def hessian(self, value: Union[Hessian, np.ndarray, None]):
        """Set the Hessian matrix as a Hessian value"""
        logger.info("Setting hessian")

        if value is None:
            self._hess = None
            return

        required_shape = (3 * self.n_atoms, 3 * self.n_atoms)

        if hasattr(value, "shape") and value.shape != required_shape:
            raise ValueError(
                "Could not set the Hessian. Incorrect shape: "
                f"{value.shape} != {required_shape}"
            )

        elif isinstance(value, Hessian):
            self._hess = value

            if self._hess.atoms is None:
                self._hess.atoms = self.atoms

        elif isinstance(value, np.ndarray):
            logger.warning(
                "Setting the Hessian from a numpy array - assuming "
                "units of Ha Å^-2"
            )
            self._hess = Hessian(value, atoms=self.atoms, units=ha_per_ang_sq)

        else:
            raise ValueError(
                f"Could not set Hessian with {value}, Must be "
                f"a numpy array or a Hessian."
            )

    @property
    def gradient(self) -> Optional[val.Gradient]:
        """
        Gradient (dE/dx) at this geometry.

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Gradients | None): Gradient with shape = (n_atoms, 3)
        """
        return self._grad

    @gradient.setter
    def gradient(self, value: Union[val.Gradient, np.ndarray, None]):
        """Set the gradient matrix"""

        if value is None:
            self._grad = None
            return

        if hasattr(value, "shape") and value.shape != (self.n_atoms, 3):
            try:
                value = value.reshape((self.n_atoms, 3))
            except (ValueError, AttributeError):
                raise ValueError(
                    "Could not set the gradient. Incorrect shape: "
                    f"{value.shape}. Must be either {(self.n_atoms, 3)}, "
                    f"or {(self.n_atoms * 3,)}"
                )

        if isinstance(value, val.Gradient):
            self._grad = value

        elif isinstance(value, np.ndarray):
            logger.warning(
                "Setting the gradients from a numpy array - "
                "assuming Ha / Å units"
            )
            self._grad = val.Gradient(value, units=ha_per_ang)

        else:
            raise ValueError(
                f"Could not set the gradient with {value}, Must "
                f"be a numpy array or a Gradient."
            )

    @property
    def frequencies(self) -> Optional[List[val.Frequency]]:
        """
        Frequencies from Hessian diagonalisation, in cm-1 by default and
        are projected from rotation and translation

        -----------------------------------------------------------------------
        Returns:
            (list(autode.values.Frequency) | None):
        """
        if self.hessian is None:
            logger.warning("No Hessian has been calculated - no frequencies")
            return None

        return self.hessian.frequencies_proj

    @property
    def vib_frequencies(self) -> Optional[List[val.Frequency]]:
        """
        Vibrational frequencies, which are all but the lowest 6 for a
        non-linear molecule and all but the lowest 5 for a linear one

        -----------------------------------------------------------------------
        Returns:
            (list(autode.values.Frequency) | None): Vibrational frequencies
        """
        n = 6 if not self.is_linear() else 5

        return self.frequencies[n:] if self.frequencies is not None else None

    @property
    def imaginary_frequencies(self) -> Optional[List[val.Frequency]]:
        """
        Imaginary frequencies of a molecule

        -----------------------------------------------------------------------
        Returns:
            (list(autode.values.Frequency) | None): Imaginary frequencies, or
                                                    None if there are none
        """
        if self.frequencies is None:
            logger.warning("Had no frequencies - could not find any imaginary")
            return None

        imag_freqs = [freq for freq in self.frequencies if freq.is_imaginary]

        if len(imag_freqs) == 0:
            logger.warning("No imaginary frequencies")
            return None

        return imag_freqs

    def normal_mode(self, mode_number: int) -> Optional[val.Coordinates]:
        """
        Vibrational normal mode indexed from 0, the first 6 are translation
        and rotation and have zero displacements. The first vibrational mode
        has mode_number = 6.

        -----------------------------------------------------------------------
        Arguments:
            mode_number (int):

        Returns:
            (autode.values.Coordinates):
        """
        if self.hessian is None:
            logger.warning("Could not calculate a normal mode displacement")
            return None

        return self.hessian.normal_modes_proj[mode_number]

    @property
    @requires_atoms
    def bond_matrix(self) -> np.ndarray:
        """
        Numpy boolean array containing which atoms are bonded, also known as
        an adjacency matrix.

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray): Adjacency matrix. shape = (n_atoms, n_atoms)
        """
        assert self.graph is not None, "Must have a molecular graph"

        matrix = np.zeros(shape=(self.n_atoms, self.n_atoms), dtype=bool)

        for bond in self.graph.edges:
            matrix[tuple(bond)] = matrix[tuple(reversed(bond))] = True

        return matrix

    @property
    def partial_charges(self) -> List[float]:
        """Partial charges on all the atoms present in this species"""
        return [atom.partial_charge for atom in self.atoms]

    @partial_charges.setter
    def partial_charges(self, value: List[float]):
        """Partial charges on all the atoms present in this species"""

        try:
            _ = list(value)
            assert len(value) == self.n_atoms
        except (TypeError, ValueError, AssertionError):
            raise ValueError(
                f"Failed to assign partial charges from {value} "
                f"must be a list with length n_atoms"
            )

        for atom, charge in zip(self.atoms, value):
            atom.partial_charge = charge

    @property
    def radius(self) -> val.Distance:
        """
        Calculate an approximate radius of this species. Does not consider any
        VdW radii of the outer most atoms i.e. purely determined on nuclear
        positions

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Distance): Radius
        """
        if self.n_atoms == 0:
            return val.Distance(0.0)

        coords = self.coordinates
        return val.Distance(np.max(distance_matrix(coords, coords)) / 2.0)

    @property
    def sn(self) -> int:
        """
        Calculate the symmetry number (σ_R) of the atoms. Only implemented for
        'small' molecules <50 atoms

        References:
        [1] Theor Chem Account (2007) 118:813
        [2] . Phys. Chem. B (2010) 114:16304

        -----------------------------------------------------------------------
        Returns:
            (int): σ_R
        """
        if self.n_atoms == 0:
            return 1

        if self.n_atoms > 50:
            logger.warning(
                "Symmetry number calculations are not implemented "
                "for large molecules. Assuming C1 -> σ_R=1"
            )
            return 1

        return symmetry_number(self)

    @property
    def is_explicitly_solvated(self) -> bool:
        return self.solvent is not None and self.solvent.is_explicit

    @property
    def is_implicitly_solvated(self) -> bool:
        return self.solvent is not None and self.solvent.is_implicit

    @property
    def atomic_symbols(self) -> List[str]:
        """Atomic symbols of all atoms in this species"""
        return [atom.label for atom in self.atoms] if self.atoms else []

    @property
    def sorted_atomic_symbols(self) -> List[str]:
        """Atomic symbols of all atoms sorted alphabetically"""
        return list(sorted(self.atomic_symbols))

    @property
    def atomic_masses(self) -> List[float]:
        """Atom masses of all the atoms in this species"""
        return [atom.mass for atom in self.atoms] if self.atoms else []

    @property
    def energy(self) -> Optional[val.PotentialEnergy]:
        """
        Last computed potential energy. Setting with a float assumes electornic
        Hartree units. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> species = ade.Species(name='H', atoms=[ade.Atom('H')], charge=0, mult=1)
            >>> species.energy is None
            True
            >>> species.energy = -0.5
            >>> species.energy
            Energy(-0.5 Ha)
            >>> species.single_point(method=ade.methods.ORCA())
            >>> species.energy
            Energy(-0.50104 Ha)

        Energies are instances of autode.values.Energy so can be converted
        to different units simply:

        .. code-block:: Python

            >>> species.energy.to('kcal mol-1')
            Energy(-314.40567 kcal mol-1)
            >>> species.energy.to('eV')
            Energy(-13.63394 eV)

        All previsouly calculated energies of a species are availble with the
        energies attribute:

        .. code-block:: Python

            >>> species.energies
            [Energy(-0.5 Ha), Energy(-0.50104 Ha)]

        -----------------------------------------------------------------------
        Returns:
            (autode.values.PotentialEnergy): Energy
        """
        return self.energies.last(val.PotentialEnergy)

    @energy.setter
    def energy(self, value: Union[val.Energy, str, float, None]):
        """
        Add an energy to the list of energies at this geometry

        -----------------------------------------------------------------------
        Arguments:
            value (float | autode.values.Energy | None):
        """

        if value is None:
            # No change required
            pass

        elif isinstance(value, val.PotentialEnergy):
            self.energies.append(value)

        else:
            # Attempt to cast the value to Potential energy
            self.energies.append(val.PotentialEnergy(float(value)))

    @property
    def h_cont(self) -> Optional[val.EnthalpyCont]:
        """
        Return the enthalpic contribution to the energy

        -----------------------------------------------------------------------
        Returns:
             (autode.values.Energy | None): H - E_elec
        """
        return self.energies.last(val.EnthalpyCont)

    @property
    def g_cont(self) -> Optional[val.FreeEnergyCont]:
        """
        Return the Gibbs (free) contribution to the energy

        -----------------------------------------------------------------------
        Returns:
             (autode.values.Energy | None): G - E_elec
        """
        return self.energies.last(val.FreeEnergyCont)

    @property
    def free_energy(self) -> Optional[val.FreeEnergy]:
        """
        Free energy (G or A) of this species, calculated using the last energy
        and free energy contribution

        -----------------------------------------------------------------------
        Returns:
            (autode.values.FreeEnergy | None): 'Gibbs' free energy
        """
        if self.energy is None or self.g_cont is None:
            logger.warning("Could not calculate G - an energy was None")
            return None

        return val.FreeEnergy(self.energy + self.g_cont)

    @property
    def enthalpy(self) -> Optional[val.Enthalpy]:
        """
        Enthalpy (H) of this species, calculated using the last energy and
        enthalpy contribution. Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> h2 = ade.Molecule(smiles='[H][H]')
            >>> orca = ade.methods.ORCA()
            >>>
            >>> h2.optimise(method=orca)
            >>> h2.calc_h_cont(method=orca)
            >>> h2.enthalpy
            Enthalpy(-1.15069 Ha)

        The enthalpy contribution is seperated, so performing a single point
        provides a new enthalpy using the electronic energy at the single-point
        level of theory:

        .. code-block:: Python

            >>> h2.single_point(method=orca)
            >>> h2.enthalpy
            Enthalpy(-1.15497 Ha)

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Enthalpy | None): Enthalpy
        """
        if self.energy is None or self.h_cont is None:
            logger.warning("Could not calculate H - an energy was None")
            return None

        return val.Enthalpy(self.energy + self.h_cont)

    @property
    def zpe(self) -> Optional[val.Energy]:
        """
        Zero point vibrational energy of this species. Any imaginary
        vibrational frequencies present are converted to their real analogues.

        -----------------------------------------------------------------------
        Returns:
            (autode.values.Energy | None): ZPE if frequencies are defined
        """
        if self.n_atoms < 2:
            # A single atom (or empty set) does not have any vibrational energy
            return val.Energy(0.0)

        if self.vib_frequencies is None:
            logger.warning(
                "Vibrational frequencies not available, cannot "
                "determine zero point energy"
            )
            return None

        h = 6.62607004e-34  # Planks constant / J s
        zpe = 0.5 * h * sum(nu.real.to("hz") for nu in self.vib_frequencies)

        return val.Energy(float(zpe), units="J").to("Ha")

    @property
    def has_reasonable_coordinates(self) -> bool:
        """
        Does this species have a 'reasonable' set of coordinates? I.e. No
        atom-atom distances that are particularly short or long. Also checks
        that all the atoms don't lie in a single plane, which is possible for
        a failed 3D embedding of a structure.

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        if self.n_atoms < 2:
            return True

        dist_matrix = distance_matrix(self.coordinates, self.coordinates)
        dist_matrix[np.diag_indices(self.n_atoms)] = 1.0

        if np.min(dist_matrix) < 0.7 or np.max(dist_matrix) > 1e6:
            logger.warning(
                f"Species({self.name}) did not have a set of "
                f"reasonable coordinates. Small or large distances"
            )
            return False

        assert self.graph is not None, "Need graph to check"
        if self.atoms.are_planar() and not self.graph.expected_planar_geometry:
            logger.warning(
                "Atoms lie in a plane but the molecular graph ⇒ "
                f"a non-planar structure. Species({self.name}) did "
                f"not have a reasonable set of coordinates"
            )
            return False

        return True

    @property
    def has_valid_spin_state(self) -> bool:
        """
        Does this species have a valid spin state given the atomic composition and
        charge state?

        .. code-block:: Python

            >>> import autode as ade
            >>> h = ade.Molecule(atoms=[ade.Atom('H')], charge=0, mult=1)
            >>> h.has_valid_spin_state
            False
            >>> hydride = ade.Molecule(atoms=[ade.Atom('H')], charge=-1, mult=1)
            >>> hydride.has_valid_spin_state
            True
        """
        num_electrons = (
            sum(atom.atomic_number for atom in self.atoms) - self.charge
        )
        num_unpaired_electrons = self.mult - 1
        return (
            num_unpaired_electrons <= num_electrons
            and num_electrons % 2 == num_unpaired_electrons % 2
        )

    @property
    def n_conformers(self) -> int:
        """
        Number of conformers of this species

        -----------------------------------------------------------------------
        Returns:
            (int):
        """
        return 0 if self.conformers is None else len(self.conformers)

    @property
    def conformers(self) -> "Conformers":
        """Conformers of this species"""
        return self._conformers

    @conformers.setter
    def conformers(
        self,
        value: Union[List["Conformer"], "Conformers", None],
    ) -> None:
        """
        Set conformers of this species

        -----------------------------------------------------------------------
        Arguments:
            value (list(autode.conformers.Conformer) | None):
        """
        if value is None:
            self._conformers.clear()
            return

        self._conformers = Conformers([conf for conf in value])

    def _generate_conformers(self, *args, **kwargs):
        raise NotImplementedError(
            "Could not generate conformers. "
            "generate_conformers() not implemented"
        )

    def _default_hessian_calculation(
        self, method=None, keywords=None, n_cores=None
    ):
        """Construct a default Hessian calculation"""

        method = methods.method_or_default_hmethod(method)
        keywords = keywords if keywords is not None else method.keywords.hess

        calc = Calculation(
            name=f"{self.name}_hess",
            molecule=self,
            method=method,
            keywords=HessianKeywords(keywords),
            n_cores=Config.n_cores if n_cores is None else n_cores,
        )
        return calc

    def _default_opt_calculation(
        self, method=None, keywords=None, n_cores=None
    ):
        """Construct a default optimisation calculation"""

        method = methods.method_or_default_hmethod(method)
        keywords = keywords if keywords is not None else method.keywords.opt
        logger.info(f"Using keywords: {keywords} to optimise with {method}")

        calc = Calculation(
            name=f"{self.name}_opt",
            molecule=self,
            method=method,
            keywords=OptKeywords(keywords),
            n_cores=Config.n_cores if n_cores is None else n_cores,
        )

        return calc

    def _run_hess_calculation(self, **kwargs):
        """Run a Hessian calculation on this species

        ----------------------------------------------------------------------
        Keyword Arguments:
            calc: Calculation, if undefined then use a default calculation

            method: Method to use for the calculation, if it's undefined.
                    Defaults to methods.get_hmethod()

            keywords: Keywords to use in a calculation, if it's undefined.
                      Defaults to method.keywords.hess
        """

        if self.n_atoms < 2:
            logger.warning(
                f"Not running a Hessian calculation on only "
                f"{self.n_atoms} atom(s). Cannot have frequencies"
            )
            return None

        calc = kwargs.pop("calc", None)

        if calc is None:
            calc = self._default_hessian_calculation(**kwargs)

        calc.run()
        return None

    @requires_conformers
    def _set_lowest_energy_conformer(self):
        """Set the species energy and atoms as those of the lowest energy
        conformer"""
        conformer = self.conformers.lowest_energy

        if conformer is None:
            raise RuntimeError(
                "Failed to set the lowest energy conformer as "
                "no suitable conformers were present"
            )

        self.atoms = conformer.atoms.copy()
        self.energy = conformer.energy
        return None

    def populate_conformers(self, *args, **kwargs):
        """Populate self.conformers"""
        return self._generate_conformers(*args, **kwargs)

    @requires_atoms
    def reorder_atoms(self, mapping: dict) -> None:
        """
        Reorder the atoms in this species (in place) using a mapping. For
        example, to reorder the atoms in a HF molecule:

        .. code-block:: Python

            >>> import autode as ade
            >>> hf = ade.Species(name='HF', charge=0, mult=1,
            ...                   atoms=[ade.Atom('H'), ade.Atom('F', x=1)])
            >>> hf.atoms
            Atoms([Atom(H, 0.0 0.0 0.0), Atom(F, 1.0, 0.0, 0.0)])
            >>> hf.reorder_atoms(mapping={0: 1, 1: 0})

        -----------------------------------------------------------------------
        Arguments:
            mapping (dict): Dictionary keyed with current atom indexes with
                            the values as the required indexing

        Raises:
            (ValueError): If the mapping is invalid
        """
        if not (
            set(mapping.keys())
            == set(mapping.values())
            == set(list(range(self.n_atoms)))
        ):
            raise ValueError("Invalid mapping. Must be 1-1 for all atoms")

        self._atoms = Atoms(
            [self.atoms[i] for i in sorted(mapping, key=lambda k: mapping[k])]
        )

        if self.graph is None:
            return  # No need to re-order a graph that is not set

        self.graph = reorder_nodes(
            graph=self.graph, mapping={u: v for v, u in mapping.items()}
        )
        return

    @requires_atoms
    def is_linear(
        self,
        tol: Optional[float] = None,
        angle_tol: val.Angle = val.Angle(1.0, "degrees"),
    ) -> bool:
        """
        Determine if a species is linear i.e all atoms are colinear

        -----------------------------------------------------------------------
        Keyword Arguments:
            tol (float | None): Tolerance on |cos(θ)| - 1 where θ is the angle
                                between the vector from atom 0 to 1 and from
                                0 to n (n > 1). Present for compatibility and
                                overrides angle_tol if not None

            angle_tol (autode.values.Angle): Tolerance on the angle considered
                                             to be linear
        """
        if tol is not None:
            angle_tol = val.Angle(np.arccos(1.0 - tol), units="rad")

        return self.atoms.are_linear(angle_tol=angle_tol)

    @requires_atoms
    def is_planar(
        self, tol: Union[float, val.Distance] = val.Distance(1e-4)
    ) -> bool:
        """
        Determine if a species is planar i.e all atoms are coplanar

        -----------------------------------------------------------------------
        Keyword Arguments:
            tol (float | None): Tolerance on the dot product between normal
                                vectors.
        """
        return self.atoms.are_planar(distance_tol=tol)

    @requires_atoms
    def translate(self, vec: Sequence[float]) -> None:
        """
        Translate the molecule by vector

        -----------------------------------------------------------------------
        Arguments:
            vec (np.ndarray | list(float)): Vector to translate by shape = (3,)
        """
        for atom in self.atoms:
            atom.translate(vec)

        return None

    @requires_atoms
    def rotate(
        self,
        axis: Union[np.ndarray, Sequence],
        theta: Union[val.Angle, float],
        origin: Union[np.ndarray, Sequence, None] = None,
    ) -> None:
        """
        Rotate the molecule by around an axis

        -----------------------------------------------------------------------
        Arguments:
            axis (np.ndarray | list(float)): Axis to rotate around. len(axis)=3

            theta (Angle | float): Angle to rotate anticlockwise by if float
                                   then assume radian units

            origin (np.ndarray | list(float) | None): Origin of the rotation
        """

        # NOTE: Requires copy as the origin may be one of the coordinates
        origin = np.zeros(3) if origin is None else np.array(origin, copy=True)

        coords = self.coordinates
        coords -= origin
        coords = np.dot(coords, get_rot_mat_euler(axis=axis, theta=theta).T)
        coords += origin

        # Set the new coordinates of each atom
        for atom, new_coord in zip(self.atoms, coords):
            atom.coord = new_coord

        return None

    @requires_atoms
    def centre(self) -> None:
        """Translate this molecule so the centroid (~COM) is at the origin"""
        self.translate(vec=-np.average(self.coordinates, axis=0))
        return None

    @requires_atoms
    def reset_graph(self) -> None:
        """
        Reset the molecular graph of this species by its connectivity
        """
        return make_graph(self)

    def has_same_connectivity_as(self, other: "Species") -> bool:
        """
        Determine if this species have the same connectivity as another

        -----------------------------------------------------------------------
        Arguments:
            other: A species which to check connectivity against

        Returns:
            (bool): Does another species have the same connectivity?
        """

        if not (hasattr(other, "n_atoms") and hasattr(other, "graph")):
            raise ValueError(
                f"Could not check if {other} had the same "
                f"connectivity as {self}, it had no n_atoms or "
                "graph attribute"
            )

        if self.n_atoms != other.n_atoms:
            return False  # Must have an identical number of atoms

        if self.n_atoms <= 1:
            return True  # 1 or 0 atom molecules have the same connectivity

        if self.graph is None or other.graph is None:
            raise ValueError(
                "Cannot check connectivity, a graph was undefined"
            )

        return is_isomorphic(self.graph, other.graph)

    @requires_atoms
    def print_xyz_file(
        self,
        title_line: Optional[str] = None,
        filename: Optional[str] = None,
        additional_title_line: Optional[str] = None,
        with_solvent: bool = True,
        append: bool = False,
    ) -> None:
        """
        Print a standard xyz file from this molecule's atoms

        -----------------------------------------------------------------------
        Keyword Arguments:
            title_line: String to add as the second line of the .xyz file

            filename: Filename ending with .xyz. If None then will use the
                      name of this molecule

            additional_title_line: Additional elements to add to the title line

            with_solvent: If the solvent is explicit then include the solvent
                          atoms in the .xyz file

            append: Should the structure be appended to the existing file
        """

        if filename is None:
            filename = f"{self.name}.xyz"

        # Default generated title line
        if title_line is None:
            title_line = (
                f"Generated by autodE on: {date.today()}. "
                f"charge = {self.charge} "
                f"mult = {self.mult} "
            )
            if self.solvent is not None and self.solvent.is_implicit:
                title_line += f"solvent_name = {self.solvent.name} "
            if self.energy is not None:
                title_line += f"E = {self.energy:.6f} Ha"

        if additional_title_line is not None:
            title_line += additional_title_line

        atoms = self.atoms
        # Add the explicit solvent molecules if present and requested
        if (
            self.solvent is not None
            and self.solvent.is_explicit
            and with_solvent
        ):
            atoms += self.solvent.atoms

        atoms_to_xyz_file(
            atoms=atoms,
            filename=filename,
            title_line=title_line,
            append=append,
        )
        return None

    @requires_atoms
    def print_mol_file(self, filename: Optional[str] = None) -> None:
        """
        Print a mol file for this molecule, with the connectivity defined in
        the graph. All bonds are defined as single bonds in molfile.

        -----------------------------------------------------------------------
        Arguments:
            filename (str): Name of the file to print
        """
        if filename is None:
            filename = f"{self.name}.mol"
        assert filename.endswith(".mol"), "Must end with *.mol"
        assert self.graph is not None

        with open(filename, "w") as mol_file:
            print(f"{self.name}", file=mol_file)
            print("autodE", file=mol_file)
            print("", file=mol_file)
            print(
                f"{self.n_atoms:3d}{self.graph.number_of_edges():3d}"
                f"  0  0  0  0  0  0  0  0999 V2000",
                file=mol_file,
            )
            for atom in self.atoms:
                x, y, z = atom.coord  # (Å)
                print(
                    f"{x:10.5f}{y:10.5f}{z:10.5f} {atom.label:3s}",
                    f"  0  0  0  0  0  0  0  0  0  0  0  0",
                    file=mol_file,
                )
            for edge in self.graph.edges:
                print(
                    f"{edge[0]+1:3d}{edge[1]+1:3d}  1" f"  0  0  0  0",
                    file=mol_file,
                )
            print("M  END", file=mol_file)

    @requires_atoms
    def optimise(
        self,
        method: Optional["Method"] = None,
        reset_graph: bool = False,
        calc: Optional[Calculation] = None,
        keywords: Union[Sequence[str], str, None] = None,
        n_cores: Optional[int] = None,
    ) -> None:
        """
        Optimise the geometry using a method

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            reset_graph (bool): Reset the molecular graph

            calc (autode.calculation.Calculation): Different e.g. constrained
                                                   optimisation calculation

            keywords (list(str) | None): Calculation keywords to use, if None
                                         then use the default for the method.
                                         Does not include solvent-specific ones

            n_cores (int | None): Number of cores to use for the calculation,
                                  if None then will default to
                                  autode.Config.n_cores
        Raises:
            (autode.exceptions.CalculationException):
        """
        logger.info(f"Running optimisation of {self.name}")

        if calc is None and method is None:
            raise ValueError(
                "Optimisation cannot be performed without "
                "a specified method or calculation."
            )

        if calc is None:
            calc = self._default_opt_calculation(method, keywords, n_cores)

        calc.run()

        method_name = "" if method is None else method.name
        self.print_xyz_file(
            filename=f"{self.name}_optimised_{method_name}.xyz"
        )

        if reset_graph:
            self.reset_graph()

        return None

    @requires_atoms
    def calc_thermo(
        self,
        method: Optional["Method"] = None,
        calc: Optional[Calculation] = None,
        temp: Union[val.Temperature, float] = val.Temperature(298.15),
        keywords: Union[Sequence[str], str, None] = None,
        **kwargs,
    ) -> None:
        """
        Calculate the free energy and enthalpy contributions using the
        ideal gas approximation

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            calc (autode.calculation.Calculation):

            keywords (autode.wrappers.keywords.Keywords):

            temp (float | autode.values.Temperature): Temperature in K

        Keyword Arguments:

            lfm_method (LFMethod | str): Method to treat low frequency
                                         modes. {'igm', 'truhlar', 'grimme'}.
                                         Defaults to Config.lfm_method

            ss (str): Standard state to use.  Defaults to Config.standard_state

        Raises:
            (autode.exceptions.CalculationException | ValueError):

        See Also:
            :meth:`autode.thermochemistry.igm.calculate_thermo_cont` for
            additional kwargs
        """
        logger.info(
            f"Calculating thermochemical contributions for {self.name}"
        )

        if isinstance(temp, float):
            logger.warning(
                "Temperature defined as a float. Assuming units of K"
            )
            temp = val.Temperature(temp)

        if "lfm_method" in kwargs:
            try:
                kwargs["lfm_method"] = LFMethod[kwargs["lfm_method"].lower()]
            except KeyError:
                raise ValueError(
                    f'{kwargs["lfm_method"]} is not valid. Must '
                    f"be one of: {[m for m in LFMethod]}"
                )

        if calc is not None and calc.output.exists:
            logger.info(
                "Setting the atoms, energy and Hessian from an "
                "existing calculation"
            )
            if calc.molecule.hessian is None:
                raise ValueError(
                    f"Failed to set the Hessian from {calc.name}."
                    f" Maybe run() hasn't been called?"
                )

            self.atoms = calc.molecule.atoms.copy()
            self.energy = calc.molecule.energy
            self.hessian = calc.molecule.hessian

        elif self.hessian is None or (
            calc is not None and not calc.output.exists
        ):
            logger.info(
                "Calculation did not exist or Hessian was None - "
                "calculating the Hessian"
            )
            self._run_hess_calculation(
                method=method, calc=calc, keywords=keywords
            )

        calculate_thermo_cont(self, temp=temp, **kwargs)
        return None

    @requires_atoms
    def calc_g_cont(self, *args, **kwargs) -> None:
        """Calculate the Gibbs free (G) contribution for this species  using
        Species.calc_thermo()"""
        return self.calc_thermo(*args, **kwargs)

    @requires_atoms
    def calc_h_cont(self, *args, **kwargs) -> None:
        """Calculate the enthalpic (H) contribution for this species using
        Species.calc_thermo()"""
        return self.calc_thermo(*args, **kwargs)

    @requires_atoms
    def single_point(
        self,
        method: "Method",
        keywords: Union["Keywords", Sequence[str], str, None] = None,
        n_cores: Optional[int] = None,
    ) -> None:
        """
        Calculate the single point energy of the species using a method

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (list(str) | None): Calculation keywords to use, if None
                                         then use the default for the method

            n_cores (int | None): Number of cores to use for the calculation,
                                  if None then use autode.Config.n_cores

        Raises:
            (autode.exceptions.CalculationException):
        """
        logger.info(f"Running single point energy evaluation of {self.name}")

        if keywords is None:
            keywords = method.keywords.sp
            logger.info(f"Using default single point keywords: {keywords}")

        else:
            keywords = SinglePointKeywords(keywords)

        assert keywords is not None, "Must have keywords"
        sp = Calculation(
            name=f"{self.name}_sp",
            molecule=self,
            method=method,
            keywords=keywords,
            n_cores=Config.n_cores if n_cores is None else n_cores,
        )
        sp.run()
        return None

    @work_in("conformers")
    def find_lowest_energy_conformer(
        self,
        lmethod: Optional["Method"] = None,
        hmethod: Optional["Method"] = None,
        allow_connectivity_changes: bool = False,
    ) -> None:
        """
        Find the lowest energy conformer of this species. Populates
        species.conformers and sets species.atoms and species.energy. By
        default will only optimise at a low-level method

        -----------------------------------------------------------------------
        Keyword Arguments:
            lmethod (autode.wrappers.ElectronicStructureMethod): Low-level
                     method to use.

            hmethod (autode.wrappers.ElectronicStructureMethod): High-level
                     method to use.

            allow_connectivity_changes (bool): Allow changes in connectivity,
                                               although not (by definition) a
                                               conformer it is useful to allow

        Raises:
            (RuntimeError): If no conformers (with energies) can be generated
        """
        logger.info("Finding lowest energy conformer")

        if self.n_atoms <= 2:
            logger.warning(
                "Cannot have conformers of a species with 2 atoms " "or fewer"
            )
            return None

        lmethod = methods.method_or_default_lmethod(lmethod)

        method_log.add("Low energy conformers located with the")
        self._generate_conformers()

        # For all generated conformers optimise with the low level of theory
        method_string = f"and optimised using {lmethod.name}"
        if hmethod is not None:
            method_string += f" then with {hmethod.name}"
        method_log.add(f"{method_string}.")

        self.conformers.optimise(method=lmethod)
        self.conformers.prune(remove_no_energy=True)

        if hmethod is not None:
            if Config.hmethod_sp_conformers:
                # Use only single point energies on lmethod geometries
                assert hmethod.keywords.low_sp is not None
                assert hmethod is not None
                self.conformers.single_point(
                    method=hmethod, keywords=hmethod.keywords.low_sp
                )
            else:
                # Otherwise run a full optimisation
                self.conformers.optimise(hmethod)

        if not allow_connectivity_changes:
            assert self.graph is not None, "Must have a graph"
            self.conformers.prune_diff_graph(self.graph)

        self._set_lowest_energy_conformer()
        logger.info(f"Lowest energy conformer found. E = {self.energy}")
        return None

    def explicitly_solvate(
        self, num: int = 10, solvent: Union[str, "Species", None] = None
    ) -> None:
        """
        Explicitly solvate this Molecule

        ----------------------------------------------------------------------
        Keyword Arguments:

            num (int): Number of solvent molecules to add around this molecule.
                       Default = 10

            solvent (str | autode.species.Species | None):

        Raises:
            (ValueError): If the solvent is not defined as a string or a
                          Species and the solvent of this species is not defined
        """
        if solvent is None and self.solvent is None:
            raise ValueError(
                f"{self.name} must be solvated with a solvent "
                "specified, as it is currently in the gas phase"
            )

        if isinstance(solvent, Species):
            self.solvent = ExplicitSolvent(solvent=solvent, num=num)

        elif isinstance(solvent, str):
            self.solvent = get_solvent(solvent, kind="explicit", num=num)

        elif (
            solvent is None
            and self.solvent is not None
            and self.solvent.is_implicit
        ):
            assert self.solvent is not None
            self.solvent = self.solvent.to_explicit(num=num)

        else:
            raise ValueError(
                f"Unsupported solvent *{solvent}*. Must be "
                f"either a string or a Species."
            )

        assert self.solvent is not None
        print(
            "WARNING: Explicit solvation is experimental is not implemented "
            "beyond generating a single reasonable initial structure "
        )
        self.solvent.randomise_around(self)
        return None

    def calc_hessian(
        self,
        method: "Method",
        keywords: Optional["Keywords"] = None,
        numerical: bool = False,
        use_central_differences: bool = False,
        coordinate_shift: Union[float, val.Distance] = val.Distance(
            2e-3, units="Å"
        ),
        n_cores: Optional[int] = None,
    ) -> None:
        """
        Calculate the Hessian

        -----------------------------------------------------------------------
        Arguments:
            method: Method to use to calculate the Hessian

            keywords: Keywords to use to calculate the Hessian, or gradient if
                      numerical = True

            numerical: Whether to do a numerical frequency calculation using
                       analytic gradients

            use_central_differences: Use central differences to calculate the
                                 numerical Hessian. If True then use
                                df/dx = [f(x+h) - f(x-h)] / 2h
                                otherwise use single sided differences (faster
                                but less accurate)
                                df/dx = [f(x+h) - f(x)] / h

            coordinate_shift: Shift applied to each Cartesian coordinate (h)
                              in the calculation of the numerical Hessian

            n_cores: Number of cores to use for the calculation. If None
                     then default to Config.n_cores
        """

        if not method.implements(CalculationType.hessian):
            logger.warning(
                f"{method} does not implement a Hessian - using a "
                f"numerical Hessian and overriding the keywords"
            )
            numerical = True

            if not isinstance(keywords, GradientKeywords):
                logger.warning(f"Using default gradient keywords for {method}")
                keywords = method.keywords.grad

        if numerical:
            if not isinstance(coordinate_shift, val.Distance):
                logger.warning(
                    f"Calculating numerical Hessian with "
                    f"h = {coordinate_shift}. Assuming units of Å"
                )
                coordinate_shift = val.Distance(coordinate_shift, units="Å")

            if keywords is None:
                logger.info(
                    "Using default gradient keywords to evaluate "
                    "numerical Hessian"
                )
                assert method.keywords.grad is not None
                keywords = method.keywords.grad

            nhc = NumericalHessianCalculator(
                self,
                method=method,
                keywords=GradientKeywords(keywords),
                do_c_diff=use_central_differences,
                shift=coordinate_shift,
                n_cores=n_cores,
            )
            nhc.calculate()
            self.hessian = nhc.hessian

        if not numerical:
            self._run_hess_calculation(
                method=method, calc=None, keywords=keywords, n_cores=n_cores
            )
        return None

    def has_identical_composition_as(self, species: "Species") -> bool:
        """Does this species have the same chemical identity as another?"""
        return self.sorted_atomic_symbols == species.sorted_atomic_symbols

    @property
    def solvent_name(self) -> Optional[str]:
        """
        Name of the solvent or None

        Returns:
            (str | None):
        """
        return None if self._solvent is None else self._solvent.name

    @solvent_name.setter
    def solvent_name(self, value: Optional[str]) -> None:
        """
        Set the solvent of this species given an optional name. Setting this to None
        removes the solvent

        Arguments:
            value (str | None): Name of the solvent to use
        """
        if value is None:
            self._solvent = None
        else:
            self._solvent = get_solvent(value, kind="implicit")

    # --- Method aliases ---
    symmetry_number = sn
