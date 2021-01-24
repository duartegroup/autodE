import autode as ade
import numpy as np
from copy import deepcopy
from autode.log import logger
from autode.neb.path import Path
from autode.utils import work_in


class PathPoint:

    def copy(self):
        """Return a copy of this point"""
        return PathPoint(self.species.copy(), deepcopy(self.constraints))

    def __init__(self, species, constraints):
        """
        Point on a PES path

        Arguments:
            species (autode.species.Species):
            constraints (dict): Distance constraints keyed with atom indexes as
                        a tuple with distances (floats) as values
        """
        self.species = species
        self.constraints = constraints

        self.energy = None    # Ha
        self.grad = None      # Ha Å^-1


class InitialPath(Path):

    @work_in('initial_path')
    def append(self, point) -> None:
        """
        Append a point to the path and  optimise it

        Arguments:
            point (PathPoint): Point on a path

        Raises:
            (autode.exceptions.CalculationException):
        """
        super().append(point)

        idx = len(self) - 1
        calc = ade.Calculation(name=f'path_opt{idx}',
                               molecule=self[idx].species,
                               method=self.method,
                               keywords=self.method.keywords.low_opt,
                               n_cores=ade.Config.n_cores,
                               distance_constraints=self[idx].constraints)
        calc.run()

        # Set the required properties from the calculation
        self[idx].species.atoms = calc.get_final_atoms()
        self[idx].energy = calc.get_energy()

        if self.method.name == 'xtb':
            # XTB prints gradients including the constraints, which are ~0
            # the gradient here is just the derivative of the electronic energy
            # so rerun a gradient calculation, which should be very fast
            calc = ade.Calculation(name=f'path_grad{idx}',
                                   molecule=self[idx].species,
                                   method=self.method,
                                   keywords=self.method.keywords.grad,
                                   n_cores=ade.Config.n_cores)
            calc.run()
            self[idx].grad = calc.get_gradients()
            calc.clean_up(force=True, everything=True)

        else:
            self[idx].grad = calc.get_gradients()

        return None

    def plot_energies(self, save=True, name='init_path', color='k', xlabel='ζ'):
        return super().plot_energies(save, name, color, xlabel)

    def truncate_final_points(self):
        """Remove any points higher than the previous one from the lend of the
        path"""
        while len(self) >= 2 and self[-2].energy < self[-1].energy:
            self.pop(-1)

        return None

    def contains_suitable_peak(self):
        """Does this path contain a peak suitable for a TS guess?"""
        if not self.contains_peak:
            return False

        if self.products_made(product=self.final_species):
            logger.info('Products made and have a peak. Assuming suitable!')
            return True

        # Products aren't made by isomorphism but we may still have a suitable
        # peak..
        if any(self[-1].constraints[b.atom_indexes] == b.final_dist
               for b in self.bonds):
            logger.warning('Have a peak, products not made on isomorphism, but'
                           ' at least one of the distances is final. Assuming '
                           'the peak is suitable    ')
            return True

        return False

    def _adjust_constraints(self, point):
        """
        Adjust the geometry constraints based on the final point

        Arguments:
            point (autode.neb.PathPoint):
        """
        logger.info(f'Adjusting constraints on')
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

            logger.info(f'|∇E_i·r| = {grad:.4f} on {bond}')

            # Downhill in energy to break/form this breaking/forming bond
            if grad * np.sign(bond.dr) > 0:
                dr = np.sign(bond.dr) * ade.Config.max_step_size

            # otherwise use a scaled value, depending on the gradient
            # large values will have small step sizes, down to min_step Å
            else:
                dr = (max_step - min_step) * np.exp(-(grad/0.05)**2) + min_step
                dr *= np.sign(bond.dr)

            new_dist = point.species.distance(*bond.atom_indexes) + dr

            # No need to go exceed final distances on forming/breaking bonds
            if bond.forming and new_dist < bond.final_dist:
                new_dist = bond.final_dist

            elif bond.breaking and new_dist > bond.final_dist:
                new_dist = bond.final_dist

            else:
                logger.info(f'Using step {dr:.3f} Å on bond: {bond}')

            point.constraints[bond.atom_indexes] = new_dist

        return None

    def generate(self, init_step_size=0.2):
        """
        Generate the path from the starting point; can be called only once!

        Keyword arguments:
            init_step_size (float): Initial step size in all bonds to calculate
                           the gradient
        """
        logger.info('Generating path from the initial species')
        assert len(self) == 1

        # Always perform an initial step linear in all bonds
        logger.info('Performing a linear step and calculating gradients')
        point = self[0].copy()

        for bond in self.bonds:
            # Shift will be -min_step_size if ∆r is negative and larger than
            # the minimum step size
            dr = np.sign(bond.dr) * min(init_step_size, np.abs(bond.dr))
            point.constraints[bond.atom_indexes] += dr

        self.append(point)
        logger.info('First point found')

        def reached_final_point():
            """Are there any more points to add?"""
            return all(point.constraints[b.atom_indexes] == b.final_dist
                       for b in self.bonds)

        logger.info('Adaptively adding points to the path')
        while not (reached_final_point() or self.contains_suitable_peak()):

            point = self[-1].copy()
            self._adjust_constraints(point=point)
            self.append(point)

        self.plot_energies()
        self.print_geometries(name='initial_path')

        return None

    def __init__(self, init_species, bonds, method, final_species=None):
        """
        PES Path

        Arguments:
            init_species (autode.species.Species):
            bonds (list(autode.pes.ScannedBond)):
            method (autode.wrappers.base.ElectronicStructureMethod):

        Keyword Arguments:
            final_species (autode.species.Species):
        """
        super().__init__()

        self.method = method
        self.bonds = bonds
        self.final_species = final_species

        # Bonds need to have the initial and final dists to drive along them
        for bond in bonds:
            assert bond.curr_dist is not None and bond.final_dist is not None

        # Add the first point - will run a constrained minimisation
        init_point = PathPoint(species=init_species,
                               constraints={bond.atom_indexes: bond.curr_dist
                                            for bond in bonds})
        self.append(init_point)
