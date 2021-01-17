"""
The theory behind this original NEB implementation is taken from
Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)
"""
from autode.log import logger
from autode.input_output import atoms_to_xyz_file
from autode.calculation import Calculation
from autode.constants import Constants
from autode.utils import work_in
from autode.config import Config
from scipy.optimize import minimize
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
blues = plt.get_cmap('Blues')


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

    images.increment()
    images.plot_energies()

    all_energies = [image.energy for image in images]
    rel_energies = [energy - min(all_energies) for energy in all_energies]

    logger.info(f'Path energy = {sum(rel_energies):.5f}')
    return sum(rel_energies)


def derivative(flat_coords, images, method, n_cores):
    """Compute the derivative of the total energy with respect to all
    components"""

    # Forces for the first image are fixed at zero
    forces = np.array(images[0].grad)

    # No need to calculate gradient as should already be there from energy eval
    for i in range(1, len(images) - 1):
        force = images[i].get_force(im_l=images[i - 1], im_r=images[i + 1])
        forces = np.append(forces, force)

    # Final zero set of forces
    forces = np.append(forces, images[-1].grad)

    # dV/dx is negative of the force
    logger.info(f'|F| = {np.linalg.norm(forces):.4f} Ha Å-1')
    return -forces


class Image:

    def _tau_xl_x_xr(self, im_l, im_r):
        """
        Calculate the normalised τ vector, along with the coordinates of the
        left, this and right images

        :param im_l: (autode.neb.Image)
        :param im_r: (autode.neb.Image)
        :return: (np.ndarray)
        """

        # ΔV_i^max
        dv_max = max(np.abs(im_r.energy - self.energy),
                     np.abs(im_l.energy - self.energy))

        # ΔV_i^min
        dv_min = min(np.abs(im_r.energy - self.energy),
                     np.abs(im_l.energy - self.energy))

        # x_i-1,   x_i,   x_i+1
        x_l, x, x_r = [image.species.coordinates.flatten()
                       for image in (im_l, self, im_r)]
        # τ_i+
        tau_plus = x_r - x
        # τ_i-
        tau_minus = x - x_l

        if im_l.energy < self.energy < im_r.energy:
            tau = tau_plus

        elif im_r.energy < self.energy < im_l.energy:
            tau = tau_minus

        elif im_l.energy < im_r.energy:
            tau = tau_plus * dv_max + tau_minus * dv_min

        elif im_r.energy < im_l.energy:
            tau = tau_plus * dv_min + tau_minus * dv_max

        else:
            raise RuntimeError('Something went very wrong in the NEB!')

        # Normalised τ vector and coordinates of the images
        return tau / np.linalg.norm(tau), x_l, x, x_r

    def get_force(self, im_l, im_r):
        """
        Compute F_i. Notation from:
        Henkelman and H. J ́onsson, J. Chem. Phys. 113, 9978 (2000)

        also a copy in autode/common

        Arguments:
            im_l (autode.neb.Image): Left image (i-1)
            im_r (autode.neb.Image): Right image (i+1)
        """
        # τ,  x_i-1,  x_i,   x_i+1
        hat_tau, x_l, x, x_r = self._tau_xl_x_xr(im_l, im_r)

        # F_i^s||
        f_parallel = (np.linalg.norm(x_r - x) * self.k -
                      np.linalg.norm(x - x_l) * self.k) * hat_tau

        # ∇V(x)_i|_|_ = ∇V(x)_i - (∇V(x)_i•τ) τ
        grad_perp = self.grad - np.dot(self.grad, hat_tau) * hat_tau

        # F_i = F_i^s|| -  ∇V(x)_i|_|_
        return f_parallel - grad_perp

    def __init__(self, name, k):
        """
        Image in a NEB

        Arguments:
            name (str):
        """
        self.name = name

        # Current optimisation iteration of this image
        self.iteration = 0

        # Force constant in Eh/Å^2
        self.k = k

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

    @property
    def peak_idx(self):
        """Get the index of the highest energy peak in this set of images"""
        if any(image.energy is None for image in self):
            logger.error('Optimisation of at least one image failed')
            return StopIteration

        def is_saddle(j):
            """Is an image j amn approximate saddle point in the surface?"""
            e = self[j].energy
            return self[j-1].energy < e and self[j+1].energy < e

        # A saddle point cannot be either the start or the end point..
        peaks = [i for i in range(1, len(self) - 1) if is_saddle(i)]

        for peak_idx in sorted(peaks, key=lambda p: -self[p].energy):
            return peak_idx

        return None

    def increment(self):
        """Advance all the iteration numbers on the images to name correctly
        also update force constants"""
        if self.peak_idx is None:
            logger.warning('Lost peak in NEB: not incrementing so calculations'
                           ' will be skipped')
            return

        for image in self:
            image.iteration += 1

        if Config.adaptive_neb_k and all(im.energy is not None for im in self):
            logger.info('Updating force constants')
            # Notation from https://doi.org/10.1063/1.1329672
            delta_k = self.max_k - self.min_k

            # E_ref is the maximum energy of the end points
            energies = [image.energy for image in self]
            e_ref = max(energies[0], energies[-1])
            e_max = max(energies)

            for image in self:
                if image.energy < e_ref:
                    image.k = self.min_k

                else:
                    image.k = self.max_k - delta_k * ((e_max - image.energy)
                                                      / (e_max - e_ref))
        return None

    def plot_energies(self):
        """Plot the NEB surface"""

        energies = [image.energy for image in self]
        if any(energy is None for energy in energies):
            logger.error('Could not plot NEB surface, an energy was None')
            return

        # Relative energies in kcal mol-1
        energies = Constants.ha2kcalmol * (np.array(energies) - min(energies))

        # Plot the relative energies each iteration as a color gradient
        plt.plot(np.arange(len(self)), energies, marker='o',
                 color=blues((self[0].iteration+1)/20),
                 label='NEB')
        plt.ylim(-0.1*np.max(energies), 1.1*np.max(energies))
        plt.xlabel('NEB coordinate')
        plt.ylabel('∆$E$ / kcal mol$^{-1}$')

        return None

    def coords(self):
        """Get a flat array of all components of every atom"""
        coords = np.array([])
        for image in self._list:

            coords = np.append(coords,
                               image.species.coordinates.flatten())
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
            image.species.coordinates = coords[i]

        return None

    def __init__(self, num, init_k, min_k=None, max_k=None):
        """
        Set of images joined by harmonic springs with force constant k

        Arguments:

            num (int): Number of images
            init_k (float): Initial force constant (Ha Å^-2)
            min_k (None | float): Minimum value of k
            max_k (None | float): Maximum value of k
        """
        self.min_k = init_k / 10 if min_k is None else float(min_k)
        self.max_k = 2 * init_k if max_k is None else float(max_k)

        self._list = [Image(name=str(i), k=init_k) for i in range(num)]


class NEB:

    def _minimise(self, method, n_cores, etol, max_n=30):
        """Minimise th energy of every image in the NEB"""
        logger.info(f'Minimising to ∆E < {etol:.4f} Ha on all NEB coordinates')

        result = minimize(total_energy,
                          x0=self.images.coords(),
                          method='L-BFGS-B',
                          jac=derivative,
                          args=(self.images, method, n_cores),
                          tol=etol,
                          options={'maxfun': max_n})

        logger.info(f'NEB path energy = {result.fun:.5f} Ha, {result.message}')
        return result

    def partition(self, n):
        """
        Partition this NEB into n steps between each image i.e. n=2 affords
        double the images

        Arguments:
            n: (int)
        """
        logger.info('Interpolating')
        species_list = [self.images[0].species]    # First unchanged

        for i, image in enumerate(self.images[1:]):
            for j in range(1, n):
                new_species = self.images[i].species.copy()
                right_coords = image.species.coordinates

                for k, atom in enumerate(new_species.atoms):
                    shift = right_coords[k] - atom.coord
                    atom.translate(vec=shift * (j / n))

                species_list.append(new_species)
            species_list.append(image.species)

        # Reset the list of images
        self.images = Images(num=len(species_list), init_k=self.images[0].k)

        for i, image in enumerate(self.images):
            image.species = species_list[i]

        return None

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
            self.images[i].species = self.images[0].species.copy()

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
        self.print_geometries(name='neb_init')

        # Calculate energy on the first and final points
        for idx in [0, -1]:
            energy_gradient(self.images[idx], method=method, n_cores=n_cores)
            # Zero the forces so the end points don't move
            self.images[idx].grad = np.zeros(shape=self.images[idx].grad.shape)

        # Energy tolerance is ~1 kcal mol-1 per image
        result = self._minimise(method, n_cores,
                                etol=0.003 * len(self.images))

        # Set the optimised coordinates for all the images
        self.images.set_coords(result.x)
        self.print_geometries(name='neb_optimised')

        # and save the plot
        plt.savefig('neb_optimised.png', dpi=300)
        plt.close()
        return None

    def get_species_saddle_point(self):
        """Find a TS guess species for this NEB: highest energy saddle point"""
        if self.images.peak_idx is None:
            logger.warning('Found no peaks in the NEB')
            return None

        return self.images[self.images.peak_idx].species

    def _init_from_end_points(self, initial, final):
        """Initialise from the start and finish points of the NEB"""

        self.images[0].species = initial
        self.images[-1].species = final

        return None

    def __init__(self, initial_species=None, final_species=None, num=8,
                 species_list=None, k=0.1):
        """
        Nudged elastic band

        Arguments:
            initial_species (autode.species.Species):
            final_species (autode.species.Species):
            num (int): Number of images in the NEB
            species_list (list(autode.species.Species)): Intermediate images
                         along the NEB
        """
        self.images = Images(num=num, init_k=k)

        if species_list is not None:
            # Initialise from a list of species rather than just end points
            self.images = Images(num=len(species_list), init_k=k)

            for i, image in enumerate(self.images):
                image.species = species_list[i]

        else:
            self._init_from_end_points(initial_species, final_species)

        logger.info(f'Initialised a NEB with {num} images')
