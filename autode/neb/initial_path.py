import autode as ade
import numpy as np
from copy import deepcopy
from autode.log import logger
from autode.neb.path import Path


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

    def _adjust_constraints(self, point):
        """
        Adjust the geometry constraints based on the final point

        Arguments:
            point (autode.neb.PathPoint):
        """

        for bond in self.bonds:
            (i, j), coords = bond.atom_indexes, self[-1].species.coordinates

            # Normalised r_ij vector
            vec = coords[j] - coords[i]
            vec /= np.linalg.norm(vec)

            # Calculate |∇E_i·r| i.e. the gradient along the bond

            # don't consider subst centers
            # max dr if grad in direction of change, min dr otherwise
            gradi = np.dot(self[-1].grad[i], vec)  # |∇E_i·r| bond midpoint
            gradj = np.dot(self[-1].grad[j], -vec)

            print(bond, gradi, gradj)

            dr_sign = np.sign(bond.dr)
            if dr_sign * gradi > 0 and dr_sign * gradj > 0:
                # Downhill move in direction of dr
                dr = ade.Config.max_step_size

            elif gradi * gradj < 0:
                # Downhill for one of the atoms
                dr = (ade.Config.max_step_size + ade.Config.min_step_size) / 2.0

            else:
                # Uphill to form/break this bond
                dr = ade.Config.min_step_size

            dr *= dr_sign
            new_dist = point.species.distance(*bond.atom_indexes) + dr

            if bond.forming and new_dist < bond.final_dist:
                new_dist = bond.final_dist

            if bond.breaking and new_dist > bond.final_dist:
                new_dist = bond.final_dist

            logger.info(f'Using step {dr_sign * dr:.3f} Å on bond: {bond}')
            point.constraints[bond.atom_indexes] = new_dist


        return None

    def generate(self, max_n=10, init_step_size=0.2):
        """
        Generate the path from the starting point; can be called only once!

        Keyword Arguments:
            max_n (int): Maximum number of constrained geometry optimisations
                         to do
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

        # Now sequentially add points to the path
        while True:

            point = self[-1].copy()
            # apply some new constraints
            self._adjust_constraints(point=point)

            if all(point.constraints[bond.atom_indexes] == bond.final_dist for
                   bond in self.bonds):
                logger.info('Done')
                break

            self.append(point)

        self.plot_energies()
        self.print_geometries(name='initial_path')
        print(self.rel_energies[self.peak_idx])

        return None

    def __init__(self, init_species, bonds, method):
        """
        PES Path

        Arguments:
            init_species (autode.species.Species):
            bonds (list(autode.pes.ScannedBond)):
            method (autode.wrappers.base.ElectronicStructureMethod):
        """
        super().__init__()

        self.method = method
        self.bonds = bonds

        # Bonds need to have the initial and final dists to drive along them
        for bond in bonds:
            assert bond.curr_dist is not None and bond.final_dist is not None

        # Add the first point - will run a constrained minimisation by default
        init_point = PathPoint(species=init_species,
                               constraints={bond.atom_indexes: bond.curr_dist
                                            for bond in bonds})
        self.append(init_point)
