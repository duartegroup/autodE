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
from typing import Union, Optional
from autode.values import GradientNorm, PotentialEnergy
from autode.log import logger
from autode.opt.optimisers import NDOptimiser
from autode.input_output import atoms_to_xyz_file


class Dimer(NDOptimiser):
    """Dimer spanning two points on the PES with a TS at the midpoint"""

    def __init__(self,
                 maxiter: int,
                 gtol:    Union[float, GradientNorm] = GradientNorm(1E-3, units='Ha Å-1'),
                 etol:    Union[float, PotentialEnergy] = PotentialEnergy(1E-4, units='Ha'),
                 coords:  Optional['autode.opt.coordinates.DimerCoordinates'] = None,

                 ):
        """
        Dimer optimiser

        -----------------------------------------------------------------------
        Arguments:
            maxiter: Maximum number of combined translation and rotation
                     iterations to perform

            gtol: Tolerance on

        """
        super().__init__(maxiter=maxiter, coords=None)

        # Note the notation follows [1] and is not necessarily the most clear..
        self.x1 = species_1.coordinates.flatten()
        self.x2 = species_2.coordinates.flatten()
        self.x0 = (self.x1 + self.x2) / 2.0

        # TODO: check the linear interpolation isn't too large. i.e delta < tol

        # Run two initial gradient evaluations
        self.g0 = self._get_gradient(coordinates=self.x0)
        self.g1 = self._get_gradient(coordinates=self.x1)

        logger.info(f'Initialised a dimer with Δ = {self.delta:.4f} Å')

        self.iterations = DimerIterations()
        self.iterations.append(DimerIteration(phi=0, d=0, dimer=self))

    @property
    def tau(self):
        """τ = (x1 - x2)/2"""
        return (self.x1 - self.x2) / 2.0

    @property
    def tau_hat(self):
        """^τ = τ / |τ|"""
        tau = self.tau
        return tau / np.linalg.norm(tau)

    @property
    def delta(self):
        """Distance between the dimer point, Δ"""
        _delta = np.linalg.norm(self.x1 - self.x2) / 2.0

        if np.isclose(_delta, 0.0):
            raise RuntimeError('Zero distance between the dimer points')

        return _delta

    @property
    def f_r(self):
        """Rotational force F_R. eqn. 3 in ref. [1]"""
        tau_hat = self.tau_hat
        return (-2.0 * (self.g1 - self.g0)
                + 2.0 * (np.dot((self.g1 - self.g0), tau_hat)) * tau_hat)

    @property
    def f_t(self):
        """Translational force F_T, eqn. 2 in ref. [1]"""
        return - self.g0 + 2.0*np.dot(self.g0, self.tau_hat) * self.tau_hat

    @property
    def theta(self):
        """Rotation direction Θ, calculated using steepest descent"""
        f_r = self.f_r

        # F_R / |F_R| with a small jitter to prevent division by zero
        return f_r / (np.linalg.norm(f_r) + 1E-8)

    @property
    def c(self):
        """Curvature of the PES, C_τ.  eqn. 4 in ref [1]"""
        return np.dot((self.g1 - self.g0), self.tau_hat) / self.delta

    @property
    def dc_dphi(self):
        """dC_τ/dϕ eqn. 6 in ref [1] """
        return 2.0 * np.dot((self.g1 - self.g0), self.theta) / self.delta

    def run(self, max_iterations=100):
        """Optimise to convergence"""
        # TODO update self.species once optimised
        raise NotImplementedError

    def rotate_coords(self, phi, update_g1=True):
        """
        Rotate the dimer by an angle phi around the midpoint.
        eqn. 13 in ref. [2]

        Arguments:
            phi (float): Rotation angle in radians (ϕ)

        Keyword Arguments:
            update_g1 (bool): Update the gradient on point 1 after the rotation
        """
        delta, tau_hat, theta = self.delta, self.tau_hat, self.theta

        self.x1 = self.x0 + delta * (tau_hat * np.cos(phi) + theta * np.sin(phi))
        self.x2 = self.x0 - delta * (tau_hat * np.cos(phi) + theta * np.sin(phi))

        if update_g1:
            self.g1 = self._get_gradient(coordinates=self.x1)

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
                      if iteration.did_translation()]

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




class DimerIterations(list):

    def print_xyz_file(self, species, point='1'):
        """Print the xyz file for one of the points in the dimer

        Arguments:
            species (autode.species.Species):

        Keyword Arguments:
            point (str | int): Point of the dimer to print. One of [1, 0, 2]
                               where 1 and 2 are the end points and 0 the
                               mid point of the dimer
        """
        _species = species.copy()

        open(f'dimer_{point}.xyz', 'w').close()   # empty the file

        for i, iteration in enumerate(self):
            coords = getattr(iteration, f'x{point}')
            _species.coordinates = coords

            atoms_to_xyz_file(_species.atoms,
                              filename=f'dimer_{point}.xyz',
                              title_line=f'Dimer iteration = {i}',
                              append=True)
        return None



class DimerIteration(BaseDimer):
    """Single iteration of a TS dimer"""

    def did_rotation(self):
        """Rotated this iteration?"""
        return True if self.phi != 0 else False

    def did_translation(self):
        """Translated this iteration?"""
        return True if self.d != 0 else False

    def __init__(self, phi, d, dimer):

        """
        Initialise from a rotation angle, a distance and the whole dimer

        Arguments:
            phi (float): Rotation with respect to the previous iteration
            d (float): Translation distance with respect to the previous
                       iteration
            dimer (autode.opt.dimer.Dimer):
        """
        super().__init__()
        self.phi = phi
        self.d = d

        self.x0 = dimer.x0.copy()
        self.x1 = dimer.x1.copy()
        self.x2 = dimer.x2.copy()

        self.g0 = dimer.g0.copy()
        self.g1 = dimer.g1.copy()
