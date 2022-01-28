"""
Dimer method for finding transition states given two points on the PES.
Notation follows
1. https://aip.scitation.org/doi/10.1063/1.2815812
based on
2. https://aip.scitation.org/doi/10.1063/1.2104507
3. https://aip.scitation.org/doi/10.1063/1.480097

-------------------------------------------------------
x : Cartesian coordinates
g : gradient in cartesian coordinates
"""
import numpy as np
from typing import Optional
from autode.calculation import Calculation
from autode.log import logger
from autode.values import GradientNorm
from autode.opt.optimisers.base import Optimiser
from autode.opt.coordinates.dimer import DimerCoordinates


class Dimer(Optimiser):
    """Dimer spanning two points on the PES with a TS at the midpoint"""

    def __init__(self,
                 maxiter:         int,
                 coords:          DimerCoordinates,
                 ratio_rot_iters: int = 10,
                 gtol:            GradientNorm = GradientNorm(1E-3, units='Å')
                 ):
        """
        Dimer optimiser

        -----------------------------------------------------------------------
        Arguments:
            maxiter: Maximum number of combined translation and rotation
                     iterations to perform

            coords: Coordinates of the dimer, consisting of the end points and
                    the interpolated midpoint

            ratio_rot_iters: Number of rotations per translation in each
                             dimer step
        """
        super().__init__(maxiter=maxiter, coords=coords)

        # TODO: check the linear interpolation isn't too large. i.e delta < tol
        self._ratio_rot_iters = ratio_rot_iters
        self.gtol = gtol

        logger.info(f'Initialised a dimer with Δ = {self._coords.delta:.4f} Å')

    @classmethod
    def optimise(cls,
                 species: 'autode.species.Species',
                 method:  'autode.wrappers.base.Method',
                 n_cores:  Optional[int] = None,
                 coords:   DimerCoordinates = None,
                 **kwargs) -> None:
        """
        Optimise a dimer pair of coordinates such that the species coordinates
        are close to a transition state

        -----------------------------------------------------------------------
        Arguments:
            species: Species to optimise to a TS using dimer iterations

            method: Electronic structure method to use to optimise

            n_cores: Number of cores to use for the optimisation

            coords: Dimer coordinates
        """

        if not isinstance(coords, DimerCoordinates):
            raise ValueError('A dimer optimisation must be initialised from '
                             'a set of dimer coordinates')

        optimiser = cls(maxiter=100, coords=coords)
        optimiser.run(species=species, method=method, n_cores=n_cores)
        return None

    def _step(self) -> None:
        """Do a step """
        pass

    def _initialise_run(self) -> None:
        """Initialise running the dimer optimisation"""

        self._coords._g = np.zeros(shape=(3, 3*self._species.n_atoms))

        # TODO: Hessian

        for idx in (0, 1):
            self._update_gradient_at(idx)

        return None

    @property
    def converged(self) -> bool:
        """Has the dimer converged?"""

        rms_g0 = np.sqrt(np.mean(np.square(self.g0)))
        return self.iteration > 0 and rms_g0 < self.gtol

    def _update_gradient_and_energy(self) -> None:
        return self._update_gradient_at(idx=0)

    def _update_gradient_at(self, idx: int) -> None:
        """Update the gradient at one of the points in the dimer"""

        if idx not in (0, 1, 2):
            raise RuntimeError('Dimer gradient must be updated at either the '
                               'midpoint: 0, left: 1 or right: 2 side')

        self._species.coordinates = self._coords[idx, :]

        grad = Calculation(name=f'{self._species.name}_{idx}_{self.iteration}',
                           molecule=self._species,
                           method=self._method,
                           keywords=self._method.keywords.grad,
                           n_cores=self._n_cores)
        grad.run()

        self._coords.e = self._species.energy = grad.get_energy()
        self._species.gradient = grad.get_gradients()
        self._coords.g[idx, :] = grad.get_gradients().flatten()

        grad.clean_up(force=True, everything=True)
        return None

    def rotate_coords(self, phi, update_g1=True):
        """
        Rotate the dimer by an angle phi around the midpoint.
        eqn. 13 in ref. [2]

        Arguments:
            phi (float): Rotation angle in radians (ϕ)

        Keyword Arguments:
            update_g1 (bool): Update the gradient on point 1 after the rotation
        """
        d, t = self._coords.delta, self._coords.tau_hat
        theta = self._coords.theta
        x0 = self._coords.x0

        self._coords.x1 = x0 + d * (t * np.cos(phi) + theta * np.sin(phi))
        self._coords.x2 = x0 - d * (t * np.cos(phi) + theta * np.sin(phi))

        if update_g1:
            self._update_gradient_at(idx=1)

        logger.info(f'Rotated coordinates, now have '
                    f'|g1 - g0| = {np.linalg.norm(self.g1 - self.g0):.4f}')
        return None

    def rotate(self, phi_tol):
        """Do a single steepest descent rotation of the dimer"""
        logger.info('Doing a single dimer rotation to minimise the curvature')

        c_phi0 = self.c  # Curvature at ϕ=0 i.e. no rotation
        # and derivative with respect to the rotation evaluated at ϕ=0
        dc_dphi0 = self.dc_dphi

        # keep a copy of the coordinates to reset after the test rotation
        x1_phi0, x2_phi0 = self.x1, self.x2

        logger.info(f'Current: C = {c_phi0:.4f}  and   dC/dϕ = {dc_dphi0:.4f}')

        # test point for rotation. eqn. 5 in ref [1]
        phi_1 = -0.5 * np.arctan(dc_dphi0 / (2.0 * np.linalg.norm(c_phi0)))

        if np.abs(phi_1) < phi_tol:
            logger.info('Test point rotation was below the threshold, not '
                        'rotating')
            self.iterations.append(DimerIteration(phi=phi_1, d=0, dimer=self))
            return None

        logger.info(f'Rotating by ϕ = {phi_1:.4f} rad and estimating the '
                    f'curvature')
        self.rotate_coords(phi=phi_1, update_g1=True)

        b1 = 0.5 * dc_dphi0  # eqn. 8 from ref. [1]

        a1 = ((c_phi0 - self.c + b1 * np.sin(2 * phi_1)) # eqn. 9 from ref. [1]
              / (1 - 2.0 * np.cos(2.0 * phi_1)))

        a0 = 2.0 * (c_phi0 - a1)  # eqn. 10 from ref. [1]

        phi_min = 0.5 * np.arctan(b1 / a1)
        logger.info(f'ϕ_min = {phi_min:.4f} rad')

        if np.abs(phi_min) < phi_tol:
            logger.info('Min rotation was below the threshold, not rotating')
            self.iterations.append(DimerIteration(phi=phi_min, d=0, dimer=self))
            return None

        c_min = 0.5 * a0 + a1 * np.cos(2.0*phi_min) + b1 * np.sin(2.0*phi_min)

        if c_min > c_phi0:
            logger.info('Optimised curvature was larger than the initial, '
                        'adding π/2')
            phi_min += np.pi / 2.0

        # Rotate back from the test point, then to the minimum
        self.x1, self.x2 = x1_phi0, x2_phi0
        self.rotate_coords(phi=phi_min, update_g1=True)

        self.iterations.append(DimerIteration(phi=phi_min, d=0, dimer=self))
        return None

    def optimise_rotation(self, phi_tol=8E-2, max_iterations=10):
        """Rotate the dimer optimally

        Keyword Arguments:
            phi_tol (float): Tolerance below which rotation is not performed
            max_iterations (int): Maximum number of rotation steps to perform
        """
        logger.info(f'Minimising dimer rotation up to δϕ = {phi_tol:.4f} rad')
        iteration, phi = 0, np.inf

        while iteration < max_iterations and phi > phi_tol:
            self.rotate(phi_tol=phi_tol)
            phi = np.abs(self.iterations[-1].phi)

            logger.info(f'Iteration={iteration}  '
                        f'ϕ={phi:.4f} ϕtol={phi_tol:.4f}')

            iteration += 1

        return None

    def translate(self, init_step_size=0.1, update_g0=True):
        """Translate the dimer, with the goal of the midpoint being the TS """
        trns_iters = [iteration for iteration in self.iterations
                      if iteration.last_step_did_translation()]

        if len(trns_iters) < 2:
            logger.info(f'Did not have two previous translation step, guessing'
                        f' γ = {init_step_size} Å')
            step_size = init_step_size

        else:
            prev_trns_iter = trns_iters[-2]
            logger.info(f'Did {len(trns_iters)} previous translations, can '
                        f'calculate γ')
            # Barzilai–Borwein method for the step size
            step_size = (np.abs(np.dot((self.x0 - prev_trns_iter.x0),
                                       (self.f_t - prev_trns_iter.f_t)))
                         / np.linalg.norm(self.f_t - prev_trns_iter.f_t)**2)

        delta_x = step_size * self.f_t
        length = np.linalg.norm(delta_x) / len(self.x0)
        logger.info(f'Translating by {length:.4f} Å per coordinate')

        for coords in (self.x0, self.x1, self.x2):
            coords += delta_x

        # Update the gradient of the midpoint, required for the translation
        if update_g0:
            self.g0 = self._get_gradient(coordinates=self.x0)

        self.iterations.append(DimerIteration(phi=0, d=length, dimer=self))
        return None

    @property
    def last_step_did_rotation(self):
        """Rotated this iteration?"""
        return not np.isclose(self._coords.phi, 0.0)

    @property
    def last_step_did_translation(self):
        """Translated this iteration?"""
        return not np.isclose(self._coords.d, 0.0)
