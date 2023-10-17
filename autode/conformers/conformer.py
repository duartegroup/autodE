import numpy as np

from typing import Optional, TYPE_CHECKING

from autode.atoms import Atoms
from autode.values import Coordinates
from autode.exceptions import AtomsNotFound
from autode.log import logger
from autode.species.species import Species

if TYPE_CHECKING:
    from autode.calculations.calculation import Calculation
    from autode.wrappers.methods import Method
    from autode.wrappers.keywords import Keywords


class Conformer(Species):
    def __init__(
        self,
        name: str = "conf",
        atoms: Optional["Atoms"] = None,
        solvent_name: Optional[str] = None,
        charge: int = 0,
        mult: int = 1,
        dist_consts: Optional[dict] = None,
        species: Optional[Species] = None,
    ):
        """
        Construct a conformer either using the standard species constructor,
        or from a species directly.

        -----------------------------------------------------------------------
        See Also:
            (autode.species.species.Species):
        """
        super().__init__(name, atoms, charge, mult, solvent_name=solvent_name)
        self._parent_atoms = None
        self._coordinates = None

        if species is not None:
            self._parent_atoms = species.atoms
            self._coordinates = species.coordinates.copy()
            self.charge = species.charge  # Require identical charge/mult/solv
            self.mult = species.mult
            self.solvent = species.solvent

        if atoms is not None:  # Specified atoms overrides species
            self.atoms = Atoms(atoms)

        self.constraints.update(distance=dist_consts)

    def __repr__(self):
        """Representation of a conformer"""
        return self._repr(prefix="Conformer")

    def __eq__(self, other):
        return super().__eq__(other)

    def single_point(
        self,
        method: "Method",
        keywords: Optional["Keywords"] = None,
        n_cores: Optional[int] = None,
    ):
        """
        Calculate a single point and default to a low level single point method

        ----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            keywords (autode.wrappers.keywords.Keywords):

            n_cores (int | None): If None then defaults to Config.n_cores
        """
        keywords = method.keywords.low_sp if keywords is None else keywords

        return super().single_point(method, keywords, n_cores=n_cores)

    def optimise(
        self,
        method: Optional["Method"] = None,
        reset_graph: bool = False,
        calc: Optional["Calculation"] = None,
        keywords: Optional["Keywords"] = None,
        n_cores: Optional[int] = None,
    ):
        """
        Optimise the geometry of this conformer using a method. Will use
        low_opt keywords if no keywords are given.

        -----------------------------------------------------------------------
        Arguments:
            method (autode.wrappers.base.ElectronicStructureMethod):

            reset_graph (bool):

            calc (autode.calculation.Calculation):

            keywords (autode.wrappers.keywords.Keywords):

            n_cores (int | None): If None then defaults to Config.n_cores
        """
        try:
            if keywords is None and method is not None:
                keywords = method.keywords.low_opt

            super().optimise(
                method, keywords=keywords, calc=calc, n_cores=n_cores
            )

        except AtomsNotFound:
            logger.error(f"Atoms not found for {self.name} but not critical")
            self.atoms = None

        return None

    @property
    def coordinates(self) -> Optional[Coordinates]:
        """Coordinates of this conformer"""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: np.ndarray):
        """Set the coordinates of this conformer"""
        if self._parent_atoms is None:
            raise ValueError(
                "Conformer has no parent atoms. Setting the "
                "coordinates will leave the atoms undefined"
            )

        self._coordinates = Coordinates(value)

    @property
    def atoms(self) -> Optional[Atoms]:
        """
        Atoms of this conformer are built from the parent atoms and the
        coordinates that are unique to this conformer.
        """

        if self._parent_atoms is None or self._coordinates is None:
            return None

        atoms = Atoms()
        for parent_atom, coord in zip(self._parent_atoms, self._coordinates):
            atom = parent_atom.copy()
            atom.coord = coord

            atoms.append(atom)

        return atoms

    @atoms.setter
    def atoms(self, value: Optional[Atoms]):
        """
        Set the atoms of this conformer.

        If None then set the corresponding coordinates of this conformer to
        None (such that self.atoms is None). If this conformer has coordinates
        then set those from the individual atomic coordinates otherwise
        set the coordinates as a batch
        """

        if value is None:  # Clear the coordinates
            self._coordinates = None
            return

        if self._parent_atoms is None:
            self._parent_atoms = value

        if self._coordinates is None:
            self._coordinates = value.coordinates
            return

        for i, atom in enumerate(value):
            parent_atom = self._parent_atoms[i]
            if atom.label != parent_atom.label:
                raise ValueError(
                    "Cannot alter the atomic symbols of a "
                    "conformer. Parent molecule was different: "
                    f"{atom.label} != {parent_atom.label}"
                )

            self._coordinates[i] = atom.coord.copy()
