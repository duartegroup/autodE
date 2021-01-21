import autode as ade
import numpy as np
from copy import deepcopy
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

    def optimise_point(self, idx):

        return None

    def truncate_final_points(self):
        """Remove any points higher than the previous one from the lend of the
        path"""
        while len(self) >= 2 and self[-2].energy < self[-1].energy:
            self.pop(-1)

        return None

    def generate(self, max_n=10):
        """
        Generate the path from the starting point; can be called only once!

        Keyword Arguments:
            max_n (int): Maximum number of constrained geometry optimisations
                         to do
        """
        assert len(self) == 1

        # Always perform an initial step linear in all bonds
        point = self[0].copy()

        for bond in self.bonds:

            dr = bond.final_dist - bond.curr_dist       # ∆r (init -> final)
            # Shift will be -min_step_size if ∆r is negative and larger than
            # 0.1 the minimum step size
            dr = np.sign(dr) * min(ade.Config.min_step_size, np.abs(dr))
            point.constraints[bond.atom_indexes] += dr

        self.append(point)

        point = self[-1].copy()
        for bond in self.bonds:
            (i, j), coords = bond.atom_indexes, self[-1].species.coordinates
            vec = coords[i] - coords[j]
            vec /= np.linalg.norm(vec)
            # Calculate |∇E·r| i.e. the gradient along the bond
            derivative = np.abs(np.dot(self[-1].grad[i], vec)) + np.abs(np.dot(self[-1].grad[j], vec))

            print(bond, derivative)
            print(np.dot(self[-1].grad[i], vec))
            print(np.dot(self[-1].grad[j], vec))
            print()

        self[-1].species.print_xyz_file()

        self.plot_energies()

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

        # Bonds need to have the correct attributes to drive along them
        for bond in bonds:
            assert bond.curr_dist is not None and bond.final_dist is not None

        self.bonds = bonds

        init_point = PathPoint(species=init_species,
                               constraints={bond.atom_indexes: bond.curr_dist
                                            for bond in bonds})
        self.append(init_point)
