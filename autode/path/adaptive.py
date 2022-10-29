import autode as ade
import numpy as np
from copy import deepcopy
from autode.log import logger
from autode.mol_graphs import make_graph
from autode.path.path import Path
from autode.transition_states.ts_guess import TSguess
from autode.utils import work_in


def get_ts_adaptive_path(
    reactant: "autode.species.ReactantComplex",
    product: "autode.species.ProductComplex",
    method: "autode.wrappers.methods.Method",
    bond_rearr: "autode.bond_rearrangement.BondRearrangement",
    name: str = "adaptive",
) -> "autode.transition_states.TSguess ":
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

    if not ts_path.contains_peak:
        logger.warning("Adaptive path had no peak")
        return None

    ts_guess = TSguess(
        atoms=ts_path[ts_path.peak_idx].species.atoms,
        reactant=reactant,
        product=product,
        bond_rearr=bond_rearr,
        name=name,
    )
    return ts_guess


def pruned_active_bonds(reactant, fbonds, bbonds):
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


class PathPoint:
    def __init__(self, species, constraints):
        """
        Point on a PES path

        -----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species):

            constraints (dict): Distance constraints keyed with atom indexes as
                        a tuple with distances (floats) as values
        """
        self.species = species
        self.species.constraints.distance = constraints

        self.energy = None  # Ha
        self.grad = None  # Ha Å^-1

    @property
    def constraints(self) -> dict:
        return self.species.constraints.distance

    def copy(self):
        """Return a copy of this point"""
        return PathPoint(
            species=self.species.new_species(),
            constraints=deepcopy(self.species.constraints.distance),
        )


class AdaptivePath(Path):
    def __init__(self, init_species, bonds, method, final_species=None):
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

        # Bonds need to have the initial and final dists to drive along them
        for bond in bonds:
            assert bond.curr_dist is not None and bond.final_dist is not None

        # Add the first point - will run a constrained minimisation if possible
        init_point = PathPoint(
            species=init_species,
            constraints={bond.atom_indexes: bond.curr_dist for bond in bonds},
        )

        if init_point.species.n_atoms > 0:
            self.append(init_point)

    def __eq__(self, other):
        """Equality of two adaptive paths"""
        if not isinstance(other, AdaptivePath):
            return False

        return super().__eq__(other)

    @work_in("initial_path")
    def append(self, point) -> None:
        """
        Append a point to the path and  optimise it

        -----------------------------------------------------------------------
        Arguments:
            point (PathPoint): Point on a path

        Raises:
            (autode.exceptions.CalculationException):
        """
        super().append(point)

        idx = len(self) - 1
        keywords = self.method.keywords.low_opt.copy()
        keywords.max_opt_cycles = 50

        calc = ade.Calculation(
            name=f"path_opt{idx}",
            molecule=self[idx].species,
            method=self.method,
            keywords=keywords,
            n_cores=ade.Config.n_cores,
        )
        calc.run()

        # Set the required properties from the calculation
        self[idx].species.atoms = calc.get_final_atoms()
        make_graph(self[idx].species)
        self[idx].energy = calc.get_energy()

        if self.method.name == "xtb" or self.method.name == "mopac":
            # XTB prints gradients including the constraints, which are ~0
            # the gradient here is just the derivative of the electronic energy
            # so rerun a gradient calculation, which should be very fast
            # while MOPAC doesn't print gradients for a constrained opt
            calc = ade.Calculation(
                name=f"path_grad{idx}",
                molecule=self[idx].species.new_species(),
                method=self.method,
                keywords=self.method.keywords.grad,
                n_cores=ade.Config.n_cores,
            )
            calc.run()
            self[idx].grad = calc.get_gradients()
            calc.clean_up(force=True, everything=True)

        else:
            self[idx].grad = calc.get_gradients()

        return None

    def plot_energies(
        self, save=True, name="init_path", color="k", xlabel="ζ"
    ) -> None:
        return super().plot_energies(save, name, color, xlabel)

    def contains_suitable_peak(self) -> bool:
        """Does this path contain a peak suitable for a TS guess?"""
        if not self.contains_peak:
            return False

        idx = self.product_idx(product=self.final_species)
        if idx is not None and self[idx].energy < self[self.peak_idx].energy:
            logger.info("Products made and have a peak. Assuming suitable!")
            return True

        # Products aren't made by isomorphism but we may still have a suitable
        # peak..
        if any(
            self[-1].constraints[b.atom_indexes] == b.final_dist
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
            (i, j), coords = bond.atom_indexes, self[-1].species.coordinates

            # Normalised r_ij vector
            vec = coords[j] - coords[i]
            vec /= np.linalg.norm(vec)

            # Calculate |∇E_i·r| i.e. the gradient along the bond. Positive
            # values are downhill in energy to form the bond and negative
            # downhill to break it
            gradi = np.dot(self[-1].grad[i], vec)  # |∇E_i·r| bond midpoint
            gradj = np.dot(self[-1].grad[j], -vec)

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

            new_dist = point.species.distance(*bond.atom_indexes) + dr

            # No need to go exceed final distances on forming/breaking bonds
            if bond.forming and new_dist < bond.final_dist:
                new_dist = bond.final_dist

            elif bond.breaking and new_dist > bond.final_dist:
                new_dist = bond.final_dist

            else:
                logger.info(f"Using step {dr:.3f} Å on bond: {bond}")

            point.constraints[bond.atom_indexes] = new_dist

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
        point = self[0].copy()

        for bond in self.bonds:
            # Shift will be -min_step_size if ∆r is negative and larger than
            # the minimum step size
            dr = np.sign(bond.dr) * min(init_step_size, np.abs(bond.dr))
            point.constraints[bond.atom_indexes] += dr

        self.append(point)
        logger.info("First point found")

        def reached_final_point():
            """Are there any more points to add?"""
            return all(
                point.constraints[b.atom_indexes] == b.final_dist
                for b in self.bonds
            )

        logger.info("Adaptively adding points to the path")
        while not (reached_final_point() or self.contains_suitable_peak()):

            point = self[-1].copy()
            self._adjust_constraints(point=point)
            self.append(point)

        self.plot_energies(name=f"{name}_path")
        self.print_geometries(name=f"{name}_path")

        return None
