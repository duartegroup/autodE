import autode as ade
import numpy as np

from autode.log import logger
from autode.path.path import Path
from autode.transition_states.ts_guess import TSguess
from autode.utils import work_in
from autode.constraints import DistanceConstraints
from autode.bonds import ScannedBond

from typing import TYPE_CHECKING, List, Optional


if TYPE_CHECKING:
    from autode.species import ReactantComplex, ProductComplex, Species
    from autode.transition_states import TSguess
    from autode.wrappers.methods import Method
    from autode.bond_rearrangement import BondRearrangement
    from autode.wrappers.keywords.keywords import OptKeywords


def get_ts_adaptive_path(
    reactant: "ReactantComplex",
    product: "ProductComplex",
    method: "Method",
    bond_rearr: "BondRearrangement",
    name: str = "adaptive",
) -> Optional[TSguess]:
    """
    Generate a TS guess geometry based on an adaptive path along multiple
    breaking and/or forming bonds

    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.species.ReactantComplex):

        product (autode.species.ProductComplex):

        method (autode.wrappers.base.ElectronicStructureMethod):

        bond_rearr (autode.bond_rearrangement.BondRearrangement):

        name (str):

    Returns:
        (autode.transition_states.ts_guess.TSguess | None):
    """
    fbonds, bbonds = bond_rearr.fbonds, bond_rearr.bbonds

    ts_path = AdaptivePath(
        init_species=reactant,
        bonds=pruned_active_bonds(reactant, fbonds, bbonds),
        method=method,
        final_species=product,
    )
    ts_path.generate(name=name)

    if ts_path.peak_idx is None:
        logger.warning("Adaptive path had no peak")
        return None

    ts_guess = TSguess(
        atoms=ts_path[ts_path.peak_idx].atoms,
        reactant=reactant,
        product=product,
        bond_rearr=bond_rearr,
        name=name,
    )
    return ts_guess


def pruned_active_bonds(
    reactant: "ReactantComplex", fbonds: list, bbonds: list
) -> List[ScannedBond]:
    """
    Prune the set of forming and breaking bonds for special cases

    (1) Three bonds form a ring, in which case the adaptive path may fail to
    traverse the MEP. If so then delete the breaking bond with the largest
    overlap to the forming bond e.g.::

           H
         /  \
        M --- C

    where all the bonds drawn are active and the C-H bond is forming


    ---------------------------------------------------------------------------
    Arguments:
        reactant (autode.species.Species):

        fbonds (list(autode.pes.pes.FormingBond)):

        bbonds (list(autode.pes.pes.BreakingBond)):

    Returns:
        (list(autode.pes.pes.ScannedBond)):
    """
    logger.info("Pruning active bonds for special cases")

    # Flat list of all the atom indexes in the breaking/forming bonds
    a_atoms = [bond.atom_indexes for bond in fbonds + bbonds]

    coords = reactant.coordinates

    if len(fbonds) == 1 and len(bbonds) == 2 and len(set(a_atoms)) == 3:
        logger.info("Found 3-membered ring with 2 breaking & 1 forming bonds")

        f_i, f_j = fbonds[0].atom_indexes
        f_vec = coords[f_i] - coords[f_j]
        f_vec /= np.linalg.norm(f_vec)

        b0_i, b0_j = bbonds[0].atom_indexes
        b0_projection = np.dot((coords[b0_i] - coords[b0_j]), f_vec)

        b1_i, b1_j = bbonds[1].atom_indexes
        b1_projection = np.dot((coords[b1_i] - coords[b1_j]), f_vec)

        if b0_projection > b1_projection:
            logger.info(f"Excluding {bbonds[0]}")
            bbonds.pop(0)
        else:
            logger.info(f"Excluding {bbonds[1]}")
            bbonds.pop(1)

    if any(bond.dr < 0 for bond in bbonds):
        logger.info(
            "Found at least one breaking bond where the final distance"
            " is shorter than the initial - removing"
        )
        """
        Counterintuitively, this is possible e.g. metallocyclobutate formation
        from a metalocyclopropane and a alkylidene (due to the way bonds are
        defined)
        """
        bbonds = [bond for bond in bbonds if bond.dr > 0]

    return fbonds + bbonds


class AdaptivePath(Path):
    def __init__(
        self,
        bonds: List[ScannedBond],
        method: "Method",
        init_species: Optional["Species"] = None,
        final_species: Optional["Species"] = None,
    ):
        """
        PES Path

        -----------------------------------------------------------------------
        Arguments:
            init_species (autode.species.Species):

            bonds (list(autode.pes.ScannedBond)):

            method (autode.wrappers.base.ElectronicStructureMethod):

            final_species (autode.species.Species):
        """
        super().__init__()

        self.method = method
        self.bonds = bonds
        self.final_species = final_species

        # Add the first point - will run a constrained minimisation if possible
        if init_species is not None:
            point = init_species.new_species()
            point.constraints.distance = DistanceConstraints(
                {b.atom_indexes: b.curr_dist for b in bonds}
            )
            self.append(point)

        self._check_bonds_have_initial_and_final_distances()

    def __eq__(self, other):
        """Equality of two adaptive paths"""
        if not isinstance(other, AdaptivePath):
            return False

        return super().__eq__(other)

    def _check_bonds_have_initial_and_final_distances(self) -> None:
        for bond in self.bonds:
            assert bond.curr_dist is not None and bond.final_dist is not None

    @work_in("initial_path")
    def append(self, point) -> None:
        """
        Append a point to the path and optimise it

        -----------------------------------------------------------------------
        Arguments:
            point (Species): Point on a path

        Raises:
            (autode.exceptions.CalculationException):
        """

        idx = len(self) - 1
        keywords: "OptKeywords" = self.method.keywords.low_opt.copy()
        keywords.max_opt_cycles = 50

        calc = ade.Calculation(
            name=f"path_opt{idx}",
            molecule=point,
            method=self.method,
            keywords=keywords,
            n_cores=ade.Config.n_cores,
        )
        calc.run()
        point.reset_graph()

        if self.method.name == "xtb" or self.method.name == "mopac":
            # XTB prints gradients including the constraints, which are ~0
            # the gradient here is just the derivative of the electronic energy
            # so rerun a gradient calculation, which should be very fast
            # while MOPAC doesn't print gradients for a constrained opt
            tmp_point_for_grad = point.new_species()
            assert self.method.keywords.grad is not None

            calc = ade.Calculation(
                name=f"path_grad{idx}",
                molecule=tmp_point_for_grad,
                method=self.method,
                keywords=self.method.keywords.grad,
                n_cores=ade.Config.n_cores,
            )
            calc.run()
            calc.clean_up(force=True, everything=True)
            assert tmp_point_for_grad.gradient is not None
            point.gradient = tmp_point_for_grad.gradient

        return super().append(point)

    def plot_energies(
        self, save=True, name="init_path", color="k", xlabel="ζ"
    ) -> None:
        return super().plot_energies(save, name, color, xlabel)

    def contains_suitable_peak(self) -> bool:
        """Does this path contain a peak suitable for a TS guess?"""
        if not self.contains_peak:
            return False

        assert self.peak_idx, "Must have a peak_idx if contains_peak"

        if self.final_species is None:
            logger.warning(
                "No final species set. Can't check peak suitability"
            )
            return False

        idx = self.product_idx(product=self.final_species)
        if idx is not None and self[idx].energy < self[self.peak_idx].energy:
            logger.info("Products made and have a peak. Assuming suitable!")
            return True

        # Products aren't made by isomorphism, but we may still have a suitable peak
        if any(
            self[-1].constraints.distance[b.atom_indexes] == b.final_dist
            for b in self.bonds
        ):
            logger.warning(
                "Have a peak, products not made on isomorphism, but"
                " at least one of the distances is final. Assuming "
                "the peak is suitable    "
            )
            return True

        return False

    def _adjust_constraints(self, point):
        """
        Adjust the geometry constraints based on the final point

        -----------------------------------------------------------------------
        Arguments:
            point (autode.neb.PathPoint):
        """
        logger.info(f"Adjusting constraints on point {len(self)}")

        # Flat list of all the atom indexes involved in the bonds
        atom_idxs = [i for bond in self.bonds for i in bond]

        max_step, min_step = ade.Config.max_step_size, ade.Config.min_step_size

        for bond in self.bonds:
            (i, j), coords = bond.atom_indexes, self[-1].coordinates

            # Normalised r_ij vector
            vec = coords[j] - coords[i]
            vec /= np.linalg.norm(vec)

            # Calculate |∇E_i·r| i.e. the gradient along the bond. Positive
            # values are downhill in energy to form the bond and negative
            # downhill to break it
            gradi = np.dot(self[-1].gradient[i], vec)  # |∇E_i·r| bond midpoint
            gradj = np.dot(self[-1].gradient[j], -vec)

            # Exclude gradients from atoms that are being substituted
            if atom_idxs.count(i) > 1:
                grad = gradj
            elif atom_idxs.count(j) > 1:
                grad = gradi
            else:
                grad = np.average((gradi, gradj))

            logger.info(f"|∇E_i·r| = {grad:.4f} on {bond}")

            # Downhill in energy to break/form this breaking/forming bond
            if grad * np.sign(bond.dr) > 0:
                dr = np.sign(bond.dr) * ade.Config.max_step_size

            # otherwise use a scaled value, depending on the gradient
            # large values will have small step sizes, down to min_step Å
            else:
                dr = (max_step - min_step) * np.exp(
                    -((grad / 0.05) ** 2)
                ) + min_step
                dr *= np.sign(bond.dr)

            new_dist = point.distance(*bond.atom_indexes) + dr

            # No need to go exceed final distances on forming/breaking bonds
            if bond.forming and new_dist < bond.final_dist:
                new_dist = bond.final_dist

            elif bond.breaking and new_dist > bond.final_dist:
                new_dist = bond.final_dist

            else:
                logger.info(f"Using step {dr:.3f} Å on bond: {bond}")

            point.constraints.distance[bond.atom_indexes] = new_dist

        return None

    def generate(self, init_step_size=0.2, name="initial") -> None:
        """
        Generate the path from the starting point; can be called only once!

        -----------------------------------------------------------------------
        Keyword arguments:
            init_step_size (float): Initial step size in all bonds to calculate
                           the gradient

            name (str): Prefix to use for saved plot and geometries
        """
        logger.info("Generating path from the initial species")
        assert len(self) == 1

        # Always perform an initial step linear in all bonds
        logger.info("Performing a linear step and calculating gradients")
        point = self[0].new_species(with_constraints=True)

        for bond in self.bonds:
            # Shift will be -min_step_size if ∆r is negative and larger than
            # the minimum step size
            dr = np.sign(bond.dr) * min(init_step_size, np.abs(bond.dr))
            point.constraints.distance[bond.atom_indexes] += dr

        self.append(point)
        logger.info("First point found")

        def reached_final_point():
            """Are there any more points to add?"""
            return all(
                point.constraints.distance[b.atom_indexes] == b.final_dist
                for b in self.bonds
            )

        logger.info("Adaptively adding points to the path")
        while not (reached_final_point() or self.contains_suitable_peak()):
            point = self[-1].new_species(with_constraints=True)
            self._adjust_constraints(point=point)
            self.append(point)

        self.plot_energies(name=f"{name}_path")
        self.print_geometries(name=f"{name}_path")

        return None
