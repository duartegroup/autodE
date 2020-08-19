from autode.log import logger
from autode.input_output import atoms_to_xyz_file
from autode.calculation import Calculation
from autode.utils import work_in
from scipy.optimize import minimize
from multiprocessing import Pool
from copy import deepcopy
import numpy as np


def energy_gradient(image, method, n_cores):
    """Calculate energies and gradients for an image using a EST method"""

    calc = Calculation(name=f'{image.name}_{image.iteration}',
                       molecule=image.species,
                       method=method,
                       keywords=method.keywords.grad,
                       n_cores=n_cores)

    @work_in(image.name)
    def run():
        calc.run()
        image.grad = calc.get_gradients().flatten()
        image.energy = calc.get_energy()
        return None

    run()
    return image


def total_energy(flat_coords, images, method, n_cores):
    """Compute the total energy across all images"""
    images.set_coords(flat_coords)

    # Number of cores per process is the floored total divided by n images
    n_cores_pp = max(int(n_cores//len(images)), 1)

    logger.info(f'Calculating energy and forces for all images with '
                f'{n_cores} total cores and {n_cores_pp} per process')

    # Run an energy + gradient evaluation in parallel across all images
    with Pool(processes=n_cores) as pool:
        results = [pool.apply_async(func=energy_gradient,
                                    args=(images[i], method, n_cores_pp))
                   for i in range(1, len(images) - 1)]

        images[1:-1] = [result.get(timeout=None) for result in results]

    # Advance all the iteration numbers on the images to name correctly
    for i in range(1, len(images) - 1):
        images[i].iteration += 1

    all_energies = [image.energy for image in images]
    rel_energies = [energy - min(all_energies) for energy in all_energies]

    logger.info(f'Path energy = {sum(rel_energies):.5f}')
    return sum(rel_energies)


def get_force(im_l, im, im_r, k=0.005):
    """
    Compute F_i. Notation from:
    Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)

    also a copy in autode/common

    Arguments:
        im_l (autode.neb.Image): Left image (i-1)
        im (autode.neb.Image): (i)
        im_r (autode.neb.Image): Right image (i+1)
        k (float): Force constant of the spring in Ha / Å^2
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
    logger.info(f'|F| = {np.linalg.norm(forces):.4f} Ha Å-1')
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

    def __setitem__(self, key, value):
        self._list[key] = value

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

        # Empty the file
        open(f'{name}.xyz', 'w').close()

        for i, image in enumerate(self.images):
            assert image.species is not None
            energy = image.energy if image.energy is not None else 'none'

            atoms_to_xyz_file(image.species.atoms,
                              f'{name}.xyz',
                              title_line=f'autodE NEB point {i}. E = {energy}',
                              append=True)
        return None

    def interpolate_geometries(self):
        """Generate simple interpolated coordinates for these set of images"""
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

    @work_in('NEB')
    def calculate(self, method, n_cores):
        """
        Optimise the NEB using forces calculated from electronic structure

        Arguments:
            method (autode.wrappers.ElectronicStructureMethod)
            n_cores (int)
        """
        self.print_geometries(name='neb_init')

        # Calculate energy on the first and final points
        for idx in [0, -1]:
            energy_gradient(self.images[idx], method=method, n_cores=n_cores)
            # Zero the forces so the end points don't move
            self.images[idx].grad = np.zeros(shape=self.images[idx].grad.shape)

        # Minimise the total energy across the path initial -> final points
        # with respect to the coordinates of all the intermediate images
        init_coords = self.images.coords()

        # Gradient tolerance is dependent on the total number of forces e.g.
        # an ~RMS rather than simply the norm
        gtol = len(self.images) * len(init_coords)/100 / 2E3
        logger.info(f'Minimising to |F|<{gtol:.4f} Ha Å-1 on NEB coordinates')

        result = minimize(total_energy,
                          x0=init_coords,
                          method='BFGS',
                          jac=derivative,
                          args=(self.images, method, n_cores),
                          tol=0.001,
                          options={'gtol': gtol, 'maxiter': 30})

        logger.info(f'NEB path energy = {result.fun:.5f} Ha, {result.message}')

        # Set the optimised coordinates for all the images
        self.images.set_coords(result.x)
        self.print_geometries(name='neb_optimised')
        return None

    def get_species_saddle_point(self):
        """Generate a TS guess for this NEB """

        def is_saddle(j):
            """Is an image j amn approximate saddle point in the surface?"""
            e = self.images[j].energy
            return self.images[j-1].energy < e and self.images[j+1].energy < e

        # A saddle point cannot be either the start or the end point..
        peaks = [i for i in range(1, len(self.images) - 1) if is_saddle(i)]

        for peak_idx in sorted(peaks, key=lambda p: -self.images[p].energy):
            yield self.images[peak_idx].species

        return None

    def _init_from_species_list(self, s_list):
        """Initialise from a list of species rather than just end points"""

        self.images = Images(num=len(s_list))

        for i, image in enumerate(self.images):
            image.species = s_list[i]

        return None

    def _init_from_end_points(self, initial, final):
        """Initialise from the start and finish points of the NEB"""

        self.images[0].species = initial
        self.images[-1].species = final

        return None

    def __init__(self, initial_species=None, final_species=None, num=8,
                 species_list=None):
        """
        Nudged elastic band class

        Arguments:
            initial_species (autode.species.Species):
            final_species (autode.species.Species):
            num (int): Number of images in the NEB
            species_list (list(autode.species.Species)): Intermediate images
                         along the NEB
        """
        self.images = Images(num=num)

        if species_list is not None:
            self._init_from_species_list(species_list)

        else:
            self._init_from_end_points(initial_species, final_species)

        # Number of images must be even for successful decomposition
        assert len(self.images) % 2 == 0

        logger.info(f'Initialised a NEB with {num} images')
