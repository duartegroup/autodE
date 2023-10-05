from abc import ABC
import numpy as np
import autode.exceptions as ex

from typing import Optional, TYPE_CHECKING

from autode.atoms import metals
from autode.config import Config
from autode.geom import calc_rmsd
from autode.constraints import DistanceConstraints
from autode.log import logger
from autode.methods import get_hmethod, get_lmethod
from autode.mol_graphs import make_graph, species_are_isomorphic
from autode.species.species import Species
from autode.exceptions import AutodeException


if TYPE_CHECKING:
    from autode.species import Species
    from autode.bond_rearrangement import BondRearrangement
    from autode.atoms import Atoms


class TSbase(Species, ABC):
    r"""
    Base transition state class. e.g.::

                 H   H
                 \ /
        F ------- C --------Cl
           r1    |   r2              r1 = 2.0 Å
                 H                   r2 = 2.2 Å
    """

    def __init__(
        self,
        atoms: "Atoms",
        reactant: Optional["Species"] = None,
        product: Optional["Species"] = None,
        name: str = "ts_guess",
        charge: int = 0,
        mult: int = 1,
        bond_rearr: Optional["BondRearrangement"] = None,
        solvent_name: Optional[str] = None,
    ):
        """
        Parent transition state class

        -----------------------------------------------------------------------
        Arguments:
            atoms: Atoms with positions and symbols

            reactant: If None then mode checking will not be available

            product: If None then mode checking will not be available

            name: Name of this TS guess

            charge: Total charge on this TS guess, in units of e

            mult: Spin multiplicity (2S+1) for S unpaired electrons

            bond_rearr: Bond rearrangement associated with the transformation
                        reactant -> product

            solvent_name: Name of the solvent
        """
        super().__init__(
            name=name,
            atoms=atoms,
            charge=charge if reactant is None else reactant.charge,
            mult=mult if reactant is None else reactant.mult,
            solvent_name=solvent_name,
        )

        self.reactant = reactant
        self.product = product
        self.bond_rearrangement = bond_rearr

        self._init_graph()
        self._init_solvent()

    def __eq__(self, other):
        """Equality of this TS to another"""
        return (
            isinstance(other, TSbase)
            and calc_rmsd(self.coordinates, other.coordinates) < 1e-6,
            super().__eq__(other),
        )

    def _init_graph(self) -> None:
        """Set the molecular graph for this TS object from the reactant"""

        if self.reactant is not None:
            logger.warning(f"Setting the graph of {self.name} from reactants")
            assert self.reactant.graph is not None
            self._graph = self.reactant.graph.copy()

        return None

    def _init_solvent(self) -> None:
        """Initialise the solvent on this TS guesss"""

        if (
            self.reactant is not None and self.product is not None
        ) and self.reactant.solvent != self.product.solvent:
            raise ValueError(
                "Cannot initialise a TS guess with reactants "
                "and products immersed in different solvents"
            )

        if self.reactant is not None:
            if (
                self.solvent is not None
                and self.reactant.solvent != self.solvent
            ):
                raise ValueError(
                    "Reactant does not have the same solvent as "
                    "the TS guess"
                )

            if (
                self.solvent is None  # No solvent has been given explicitly
                and self.reactant.solvent is not None
            ):
                logger.info(
                    "Setting TS guess solvent from reactant state to "
                    f"{self.reactant.solvent}"
                )
                self.solvent = self.reactant.solvent

        return None

    @property
    def has_imaginary_frequencies(self) -> bool:
        """Does this possible transition state have any imaginary modes?"""
        return self.imaginary_frequencies is not None

    @property
    def could_have_correct_imag_mode(self) -> bool:
        """
        Determine if a point on the PES could have the correct imaginary mode.
        This must have

        (0) An imaginary frequency      (quoted as negative in most EST codes)
        (1) The most negative(/imaginary) is more negative that a threshold,
            which is defined as autode.config.Config.min_imag_freq

        -----------------------------------------------------------------------
        Returns:
            (bool):

        Raises:
            (ValueError): If the bond-rearrangement is not set, so that there
                          is no chance of determining the right mode
        """
        if self.bond_rearrangement is None:
            raise ValueError(
                "Do not have a bond rearrangement - cannot "
                "check the imaginary mode"
            )

        if self.hessian is None:
            logger.info("Calculating the hessian..")
            self._run_hess_calculation(method=get_hmethod())

        imag_freqs = self.imaginary_frequencies

        if imag_freqs is None:
            logger.warning(
                "Hessian had no imaginary modes. Do not have the "
                "correct mode"
            )
            return False

        logger.info(f"Hessian had {len(imag_freqs)} imaginary modes")
        if imag_freqs[0] > Config.min_imag_freq:
            logger.warning("Imaginary modes were too small to be significant")
            return False

        # Check very conservatively for the correct displacement
        if not self.imag_mode_has_correct_displacement(
            delta_threshold=0.05, req_all=False
        ):
            logger.warning("Species does not have the correct imaginary mode")
            return False

        logger.info("Species could have the correct imaginary mode")
        return True

    @property
    def has_correct_imag_mode(self) -> bool:
        """Check that the imaginary mode is 'correct' set the calculation
        (hessian or optts)

        -----------------------------------------------------------------------
        Returns:
            (bool):

        Raises:
            (ValueError): If reactants and products aren't set, thus cannot
                        run a quick reaction profile
        """

        # Run a fast check on  whether it's likely the mode is correct
        if not self.could_have_correct_imag_mode:
            return False

        if self.imag_mode_has_correct_displacement(req_all=True):
            logger.info(
                "Displacement of the active atoms in the imaginary "
                "mode bond forms and breaks the correct bonds"
            )
            return True

        # Perform displacements over the imaginary mode to ensure the mode
        # connects reactants and products
        if self.imag_mode_links_reactant_products(disp_mag=1.0):
            logger.info("Imaginary mode does link reactants and products")
            return True

        logger.warning("Species does *not* have the correct imaginary mode")
        return False

    def imag_mode_has_correct_displacement(
        self,
        disp_mag: float = 1.0,
        delta_threshold: float = 0.3,
        req_all: bool = True,
    ) -> bool:
        """
        Check whether the imaginary mode in a calculation with a hessian forms
        and breaks the correct bonds

        -----------------------------------------------------------------------
        Keyword Arguments:
            disp_mag (float):

            delta_threshold (float): Required ∆r on a bond for the bond to be
                                     considered as forming
            req_all (bool): Require all the bonds to have the correct displacements

        Returns:
            (bool):
        """
        assert self.bond_rearrangement, "Must have a bond rearrangement"

        logger.info(
            "Checking displacement on imaginary mode forms the correct"
            " bonds"
        )

        f_species = displaced_species_along_mode(
            self, mode_number=6, max_atom_disp=0.5, disp_factor=disp_mag
        )

        b_species = displaced_species_along_mode(
            self, mode_number=6, max_atom_disp=0.5, disp_factor=-disp_mag
        )

        # Be conservative with metal complexes - what even is a bond..
        if imag_mode_generates_other_bonds(
            self, f_species, b_species, allow_mx=True
        ):
            logger.warning(
                "Imaginary mode generates bonds that are not active"
            )
            return False

        # Product could be either the forward displaced molecule or the
        # backwards equivalent
        for product in (f_species, b_species):
            fbond_bbond_correct_disps = []

            for fbond in self.bond_rearrangement.fbonds:
                ts_dist = self.distance(*fbond)
                p_dist = product.distance(*fbond)

                # Displaced distance towards products should be shorter than
                # the distance at the TS if the bond is forming
                if ts_dist - p_dist > delta_threshold:
                    fbond_bbond_correct_disps.append(True)

                else:
                    fbond_bbond_correct_disps.append(False)

            for bbond in self.bond_rearrangement.bbonds:
                ts_dist = self.distance(*bbond)
                p_dist = product.distance(*bbond)

                # Displaced distance towards products should be longer than the
                # distance at the TS if the bond is breaking
                if p_dist - ts_dist > delta_threshold:
                    fbond_bbond_correct_disps.append(True)

                else:
                    fbond_bbond_correct_disps.append(False)

            logger.info(
                f"List of forming and breaking bonds that have the "
                f"correct properties {fbond_bbond_correct_disps}"
            )

            if all(fbond_bbond_correct_disps) and req_all:
                logger.info(
                    f"{product.name} afforded the correct bond "
                    f"forming/breaking reactants -> products"
                )
                return True

            if not req_all and any(fbond_bbond_correct_disps):
                logger.info("At least one bond had the correct displacement")
                return True

        logger.warning(
            "Displacement along the imaginary mode did not form "
            "and break the correct bonds"
        )
        return False

    def imag_mode_links_reactant_products(self, disp_mag: float = 1.0) -> bool:
        """Displaces atoms along the imaginary mode forwards (f) and backwards (b)
        to see if products and reactants are made

        -----------------------------------------------------------------------
        Keyword Arguments:
            disp_mag (float): Distance to be displaced along the imag mode
                             (default: 1.0 Å)

        Returns:
            (bool): if the imag mode is correct or not
        """
        logger.info(
            "Displacing along imag modes to check that the TS links "
            "reactants and products"
        )
        if self.reactant is None or self.product is None:
            raise ValueError(
                "Could not check imaginary mode – reactants "
                " and/or products not set "
            )

        # Generate and optimise conformers with the low level of theory
        try:
            self.reactant.populate_conformers()
            self.product.populate_conformers()
        except NotImplementedError:
            logger.error(
                "Could not generate conformers of reactant/product(s)"
                " QRC will run without conformers"
            )

        # Get the species by displacing forwards along the mode
        f_mol = displaced_species_along_mode(
            self, mode_number=6, disp_factor=disp_mag, max_atom_disp=0.2
        )
        f_mol.name = f"{self.name}_forwards"

        # and the same backwards
        b_mol = displaced_species_along_mode(
            self, mode_number=6, disp_factor=-disp_mag, max_atom_disp=0.2
        )
        b_mol.name = f"{self.name}_backwards"
        assert f_mol.graph and b_mol.graph, "Must have graphs"

        # The high and low level methods may not have the same minima, so
        # optimise and recheck isomorphisms
        for method in (get_hmethod(), get_lmethod()):
            for mol in (f_mol, b_mol):
                try:
                    mol.optimise(
                        method=method,
                        keywords=method.keywords.low_opt,
                        reset_graph=True,
                    )

                except ex.CalculationException:
                    logger.error(
                        f"Failed to optimise {mol.name} with "
                        f"{method}. Assuming no link"
                    )
                    return False

            if forward_backward_isomorphic_to_reactant_product(
                f_mol, b_mol, self.reactant, self.product
            ):
                return True

        logger.info(f"Forwards displaced edges {f_mol.graph.edges}")
        logger.info(f"Backwards displaced edges {b_mol.graph.edges}")
        return False

    @property
    def active_bond_constraints(self) -> DistanceConstraints:
        """
        Set all the distance constraints required in an optimisation as the
        active bonds

        -----------------------------------------------------------------------
        Returns:
            (dict): Keyed with atom indexes for the active atoms (tuple) and
                    equal to the constrained value
        """
        assert self.graph is not None, "Must have a molecular graph"
        constraints = DistanceConstraints()

        for edge in self.graph.edges:
            if self.graph.edges[edge]["active"]:
                constraints[edge] = self.distance(*edge)

        return constraints


def displaced_species_along_mode(
    species: Species,
    mode_number: int,
    disp_factor: float = 1.0,
    max_atom_disp: float = 99.9,
) -> Species:
    """
    Displace the geometry along a normal mode with mode number indexed from 0,
    where 0-2 are translational normal modes, 3-5 are rotational modes and 6
    is the largest magnitude imaginary mode (if present). To displace along
    the second imaginary mode we have mode_number=7

    ---------------------------------------------------------------------------
    Arguments:
        species (autode.species.Species):

        mode_number (int): Mode number to displace along

        disp_factor (float): Distance to displace (default: {1.0})

        max_atom_disp (float): Maximum displacement of any atom (Å)

    Returns:
        (autode.species.Species):

    Raises:
        (autode.exceptions.CouldNotGetProperty | autode.exceptions.AutodeException):
    """
    logger.info(f"Displacing along mode {mode_number} in {species.name}")

    mode_disp_coords = species.normal_mode(mode_number)
    if mode_disp_coords is None:
        raise AutodeException(
            "Could not get a displaced species. No normal mode "
            "could be found"
        )

    coords = species.coordinates
    disp_coords = coords.copy() + disp_factor * mode_disp_coords

    # Ensure the maximum displacement distance any single atom is below the
    # threshold (max_atom_disp), by incrementing backwards in steps of 0.05 Å,
    # for disp_factor = 1.0 Å
    for _ in range(20):
        if (
            np.max(np.linalg.norm(coords - disp_coords, axis=1))
            < max_atom_disp
        ):
            break

        disp_coords -= (disp_factor / 20) * mode_disp_coords

    # Create a new species from the initial
    disp_species = Species(
        name=f"{species.name}_disp",
        atoms=species.atoms.copy(),
        charge=species.charge,
        mult=species.mult,
    )
    disp_species.coordinates = disp_coords

    return disp_species


def imag_mode_generates_other_bonds(
    ts: TSbase, f_species: Species, b_species: Species, allow_mx: bool = False
) -> bool:
    """
    Determine if the forward or backwards displaced molecule break or make
    bonds that aren't in all the active bonds bond_rearrangement.all. Will be
    fairly conservative here

    ---------------------------------------------------------------------------
    Arguments:
        ts (autode.transition_states.base.TSbase):

        f_species (autode.species.Species): Forward displaced species

        b_species (autode.species.Species): Backward displaced species

        allow_mx (bool): Allow any metal-X bonds where X is another element

    Returns:
        (bool):
    """

    _ts: TSbase = ts.copy()
    assert _ts.graph is not None, "TS must have a molecular graph"

    for species in (_ts, f_species, b_species):
        make_graph(species, rel_tolerance=0.3)

    for product in (f_species, b_species):
        assert product.graph is not None, "Must have a graph for product"

        new_bonds_in_product = set(
            [
                bond
                for bond in product.graph.edges
                if bond not in _ts.graph.edges
            ]
        )

        if allow_mx:
            new_bonds_in_product = set(
                [
                    (i, j)
                    for i, j in new_bonds_in_product
                    if _ts.atoms[i].label not in metals
                    and _ts.atoms[j].label not in metals
                ]
            )

        br = _ts.bond_rearrangement
        assert br is not None, "Must have a bond rearrangement"

        if not set(a for b in new_bonds_in_product for a in b).issubset(
            set(br.active_atoms)
        ):
            logger.warning(f"New bonds in product: {new_bonds_in_product}")
            logger.warning(
                f"Active bonds: {br.all}. Active atoms {br.active_atoms}"
            )
            return True

    logger.info("Imaginary mode does not generate any other unwanted bonds")
    return False


def forward_backward_isomorphic_to_reactant_product(
    forwards: Species,
    backwards: Species,
    reactant: "Species",
    product: "Species",
) -> bool:
    """
    Are the forward/backward displaced species isomorphic to
    reactants/products?

    ---------------------------------------------------------------------------
    Arguments:
        forwards (autode.species.Species):

        backwards (autode.species.Species):

        reactant (autode.species.ReactantComplex):

        product (autode.species.ProductComplex):

    Returns:
        (bool):
    """

    if any(mol.atoms is None for mol in (forwards, backwards)):
        logger.warning(
            "Atoms not set in the output. " "Cannot calculate isomorphisms"
        )
        return False

    if species_are_isomorphic(backwards, reactant) and species_are_isomorphic(
        forwards, product
    ):
        logger.info(
            "Forwards displacement lead to products " "and backwards reactants"
        )
        return True

    if species_are_isomorphic(forwards, reactant) and species_are_isomorphic(
        backwards, product
    ):
        logger.info(
            "Backwards displacement lead to products "
            "and forwards to reactants"
        )
        return True

    return False
