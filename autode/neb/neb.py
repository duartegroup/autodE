from autode.species import Reactant, Product
from autode.input_output import xyz_file_to_atoms
from autode.input_output import atoms_to_xyz_file
from autode.calculation import Calculation
from autode.methods import XTB
from scipy.optimize import minimize
from copy import deepcopy
import numpy as np


def energy_gradient(image, method):
    """Calculate energies and gradients for an image using a EST method"""

    calc = Calculation(name=f'{image.name}_{image.iteration}',
                       molecule=image.species,
                       method=method,
                       keywords=method.keywords.grad)

    calc.run()
    image.grad = calc.get_gradients().flatten()
    image.energy = calc.get_energy()

    return None


def total_energy(flat_coords, images, method, n_cores):
    """Compute the total energy across all images"""

    images.set_coords(flat_coords)

    for i in range(1, len(images) - 1):
        energy_gradient(images[i], method)
        images[i].iteration += 1

    all_energies = [image.energy for image in images]
    rel_energies = [energy - min(all_energies) for energy in all_energies]
    return sum(rel_energies)


def get_force(im_l, im, im_r, k=0.005):
    """
    Compute F_i

    Arguments:
        im_l (autode.neb.Image): Left image (i-1)
        im (autode.neb.Image): (i)
        im_r (autode.neb.Image): Right image (i+1)
        k (float): Force constant of the spring
    """
    # ΔV_i^max
    dv_max = max(np.abs(im_r.energy - im.energy),
                 np.abs(im_l.energy - im.energy))

    # ΔV_i^min
    dv_min = min(np.abs(im_r.energy - im.energy),
                 np.abs(im_l.energy - im.energy))

    # x_i-1,   x_i,   x_i+1
    x_l, x, x_r = [image.species.get_coordinates().flatten()
                   for image in (im_l, im, im_r)]
    # τ_i+
    tau_plus = x_r - x
    # τ_i-
    tau_minus = x - x_l

    if im_l.energy < im.energy < im_r.energy:
        tau = tau_plus

    elif im_r.energy < im.energy < im_l.energy:
        tau = tau_minus

    elif im_l.energy < im_r.energy:
        tau = tau_plus * dv_max + tau_minus * dv_min

    elif im_r.energy < im_l.energy:
        tau = tau_plus * dv_min + tau_minus * dv_max

    else:
        raise RuntimeError

    # Normalised τ vector
    hat_tau = tau / np.linalg.norm(tau)

    # F_i^s||
    f_parallel = (np.linalg.norm(x_r - x) * k -
                  np.linalg.norm(x - x_l) * k) * hat_tau

    # ∇V(x)_i|_|_ = ∇V(x)_i - (∇V(x)_i•τ) τ
    grad_perp = im.grad - np.dot(im.grad, hat_tau) * hat_tau

    # F_i = F_i^s|| -  ∇V(x)_i|_|_
    return f_parallel - grad_perp


def derivative(flat_coords, images, method, n_cores):
    """Compute the derivative of the total energy with respect to all
    components"""

    # Forces for the first image are fixed at zero
    forces = np.array(images[0].grad)

    # No need to calculate gradient as should already be there from energy eval
    for i in range(1, len(images) - 1):
        force = get_force(im_l=images[i - 1],
                          im=images[i],
                          im_r=images[i + 1])

        forces = np.append(forces, force)

    # Final zero set of forces
    forces = np.append(forces, images[-1].grad)

    # dV/dx is negative of the force
    return -forces


class Image:

    def __init__(self, name):
        """
        Image in a NEB

        Arguments:
            name (str):
        """
        self.name = name

        # Current optimisation iteration of this image
        self.iteration = 0

        self.species = None         # autode.species.Species
        self.energy = None          # float
        self.grad = None            # np.ndarray shape (3xn_atoms,)


class Images:

    def __len__(self):
        return len(self._list)

    def __getitem__(self, item):
        return self._list[item]

    def coords(self):
        """Get a flat array of all components of every atom"""
        coords = np.array([])
        for image in self._list:

            coords = np.append(coords,
                               image.species.get_coordinates().flatten())
        return coords

    def set_coords(self, coords):
        """
        Set the flat array of coordinates to the species in the images

        Arguments:
            coords (np.ndarray): shape (num x n x 3,)
        """

        n_atoms = self._list[0].species.n_atoms
        coords = coords.reshape((len(self), n_atoms, 3))

        for i, image in enumerate(self._list):
            image.species.set_coordinates(coords[i])

        return None

    def __init__(self, num):

        self._list = [Image(name=str(i)) for i in range(num)]


class NEB:

    def print_geometries(self, name='neb'):
        """Print an xyz trajectory of the geometries in the NEB"""

        for i, image in enumerate(self.images):
            assert image.species is not None

            atoms_to_xyz_file(image.species.atoms,
                              f'{name}.xyz',
                              title_line=f'autodE NEB point {i}',
                              append=True)
        return None

    def interpolate_geometries(self):
        """Generate interpolated coordinates for these set of images"""
        n = len(self.images)

        # Interpolate images between the starting point i=0 and end point i=n-1
        for i in range(1, n - 1):

            # Use a copy of the starting point for atoms, charge etc.
            self.images[i].species = deepcopy(self.images[0].species)

            # For all the atoms in the species translate an amount so the
            # spacing is even between the initial and final points
            for j, atom in enumerate(self.images[i].species.atoms):

                # Shift vector is final minus current
                shift = self.images[-1].species.atoms[j].coord - atom.coord
                # then an equal spacing is the i-th point in the grid
                atom.translate(vec=shift * (i / n))

        self.print_geometries()
        return None

    def calculate(self, method, n_cores):
        """
        Optimise the NEB using forces calculated from electronic structure

        Arguments:
            method (autode.wrappers.ElectronicStructureMethod)
            n_cores (int)
        """
        # Calculate energy on the first and final points
        for idx in [0, -1]:
            energy_gradient(self.images[idx], method=method)
            # Zero the forces so the end points don't move
            self.images[idx].grad = np.zeros(shape=self.images[idx].grad.shape)

        # Minimise the total energy across the path initial -> final points
        # with respect to the coordinates of all the intermediate images
        result = minimize(total_energy,
                          x0=self.images.coords(),
                          method='BFGS',
                          jac=derivative,
                          args=(self.images, method, n_cores),
                          tol=0.01)
        print(result)

        # Set the optimised coordinates for all the images
        self.images.set_coords(result.x)

        return None

    def ts_guess(self):
        """Generate a TS guess for this NEB """
        max_energy_im = sorted(self.images, key=lambda im: im.energy)[-1]

        atoms_to_xyz_file(max_energy_im.species.atoms,
                          filename='ts_guess.xyz')

        # TODO return a TSguess
        raise NotImplementedError

    def __init__(self, initial_species, final_species, num):
        """
        Nudged elastic band class

        :param initial_species: (autode.species.Species)
        :param final_species: (autode.species.Species)
        :param num: (int) Number of images in the NEB
        """
        # Number of images must be even for successful decomposition
        assert num % 2 == 0

        self.images = Images(num=num)

        # Set the end points on the NEB
        self.images[0].species = initial_species
        self.images[-1].species = final_species


if __name__ == '__main__':

    reactant = Reactant(name='r', atoms=xyz_file_to_atoms('reactant.xyz'))
    product = Product(name='p', atoms=xyz_file_to_atoms('product.xyz'))

    neb = NEB(reactant, product, num=10)
    neb.interpolate_geometries()
    neb.calculate(method=XTB(), n_cores=4)
    neb.ts_guess()
