import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional
from autode.log import logger
from autode.config import Config
from autode.calculation import Calculation
from autode.values import GradientNorm, PotentialEnergy
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.optimisers.hessian_update import NullUpdate


class Optimiser(ABC):
    """Abstract base class for an optimiser"""

    def __init__(self,
                 maxiter: int,
                 coords:  Optional['autode.opt.OptCoordinates'] = None):
        """
        Optimiser

        ----------------------------------------------------------------------
        Arguments:
            maxiter: Maximum number of iterations to perform

            coords: Coordinates to use in the optimisation
                    e.g. CartesianCoordinates. If None then will initialise the
                    coordinates from _species
        """
        if int(maxiter) <= 0:
            raise ValueError('An optimiser must be able to run at least one '
                             f'step, but tried to set maxiter = {maxiter}')

        self._maxiter = int(maxiter)
        self._n_cores:  int = Config.n_cores

        self._history = _OptimiserHistory()

        self._coords = coords
        self._species: Optional['autode.species.Species'] = None
        self._method:  Optional['autode.wrappers.base.Method'] = None

    @classmethod
    @abstractmethod
    def optimise(cls,
                 species: 'autode.species.Species',
                 method:  'autode.wrappers.base.Method',
                 n_cores: Optional[int] = None,
                 coords:  Optional[OptCoordinates] = None,
                 **kwargs):
        """
        Optimise a species using a method

        .. code-block:: Python

          >>> import autode as ade
          >>> mol = ade.Molecule(smiles='C')
          >>> Optimiser.optimise(mol, method=ade.methods.ORCA())
        """

    def run(self,
            species: 'autode.species.Species',
            method:  'autode.wrappers.base.Method',
            n_cores: Optional[int] = None
            ) -> None:
        """
        Run the optimiser. Updates species.atoms and species.energy

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species): Species to optimise, if None
                    then use the species this optimiser was initalised with

            method (autode.methods.Method): Method to use. Calculations will
                   use method.keywords.grad for gradient calculations

        Keyword Arguments:
            n_cores (int | None): Number of cores to use for the gradient
                        evaluations. If None then use autode.Config.n_cores
        """
        self._n_cores = n_cores if n_cores is not None else Config.n_cores

        self._initialise_species_and_method(species, method)
        self._initialise_run()

        logger.info(f'Using {self._method} to optimise {self._species.name} '
                    f'with {self._n_cores} cores using {self._maxiter} max '
                    f'iterations')
        logger.info('Iteration\t|∆E| / \\kcal mol-1 \t||∇E|| / Ha Å-1')

        while not self.converged:

            self._step()                                # Update self._coords
            self._update_gradient_and_energy()          # Update self._coords.g

            self._log_convergence()

            if self._exceeded_maximum_iteration:
                break

        logger.info(f'Converged: {self.converged}, in {self.iteration} cycles')
        return None

    @property
    def iteration(self) -> int:
        """
        Iteration of the optimiser, which is equal to the length of the history
        minus one, for zero indexing.

        -----------------------------------------------------------------------
        Returns:
            (int): Current iteration
        """
        return len(self._history) - 1

    def _initialise_species_and_method(self,
                                       species: 'autode.species.Species',
                                       method:  'autode.wrappers.base.Method'
                                       ) -> None:
        """
        Initialise the internal species and method. They be the correct types

        -----------------------------------------------------------------------
         Raises:
             (ValueError): For incorrect types
         """
        from autode.species.species import Species
        from autode.wrappers.base import Method

        if not isinstance(species, Species):
            raise ValueError(f'{species} must be a autoode.Species instance '
                             f'but had {type(species)}')

        if not isinstance(method, Method):
            raise ValueError(f'{method} must be a autoode.wrappers.base.Method '
                             f'instance but had {type(method)}')

        self._method, self._species = method, species
        return None

    def _update_gradient_and_energy(self) -> None:
        """
        Update the gradient of the energy with respect to the coordinates
        using the method. Will transform from the current coordinates type
        to Cartesian coordinates to perform the calculation, then back.

        -----------------------------------------------------------------------
        Raises:
            (autode.exceptions.CalculationException):
        """
        # Calculations need to be performed in cartesian coordinates
        self._species.coordinates = self._coords.to('cart')

        grad = Calculation(name=f'{self._species.name}_opt_{self.iteration}',
                           molecule=self._species,
                           method=self._method,
                           keywords=self._method.keywords.grad,
                           n_cores=self._n_cores)
        grad.run()

        # Set the energy, gradient and remove all the calculation files
        self._coords.e = self._species.energy = grad.get_energy()
        self._species.gradient = grad.get_gradients()
        grad.clean_up(force=True, everything=True)

        fixed_idxs = self._species.constraints.cartesian
        self._coords.update_g_from_cart_g(arr=self._species.gradient,
                                          fixed_atom_idxs=fixed_idxs)
        return None

    def _update_hessian_gradient_and_energy(self) -> None:
        """
        Update the energy, gradient and Hessian using the method. Will
        transform from the current coordinates type to Cartesian coordinates
        to perform the calculation, then back. Uses a numerical Hessian if
        analytic Hessians are not implemented for this method

        -----------------------------------------------------------------------
        Raises:
            (autode.exceptions.CalculationException):
        """
        self._update_gradient_and_energy()

        species = self._species.new_species(name=f'{self._species.name}'
                                                 f'_opt_{self.iteration}')
        species.coordinates = self._coords.to('cartesian')

        species.calc_hessian(method=self._method,
                             keywords=self._method.keywords.hess,
                             n_cores=self._n_cores)

        self._species.hessian = species.hessian.copy()

        fixed_idxs = self._species.constraints.cartesian
        self._coords.update_h_from_cart_h(self._species.hessian,
                                          fixed_atom_idxs=fixed_idxs)
        return None

    @property
    def _coords(self) -> Optional[OptCoordinates]:
        """
        Current set of coordinates this optimiser is using
        """
        if len(self._history) == 0:
            logger.warning('Optimiser had no history, thus no coordinates')
            return None

        return self._history[-1]

    @_coords.setter
    def _coords(self, value: Optional[OptCoordinates]) -> None:
        """
        Set a new set of coordinates of this optimiser, will append to the
        current history.

        -----------------------------------------------------------------------
        Arguments:
            value (OptCoordinates | None):

        Raises:
            (ValueError): For invalid input
        """
        if value is None:
            return

        elif isinstance(value, OptCoordinates):
            self._history.append(value.copy())

        else:
            raise ValueError('Cannot set the optimiser coordinates with '
                             f'{value}')

    @abstractmethod
    def _step(self) -> None:
        """
        Take a step with this optimiser. Should only act on self._coords
        using the gradient (self._coords.g) and hessians (self._coords.h)
        """

    @abstractmethod
    def _initialise_run(self) -> None:
        """
        Initialise all attributes required to call self._step()

        For example:

            self._coords     (from self._species)
            self._coords.g
            self._coords.h
        """

    @property
    @abstractmethod
    def converged(self) -> bool:
        """Has this optimisation converged"""

    def _log_convergence(self) -> None:
        """Log the iterations in the form:
        Iteration   |∆E| / kcal mol-1    ||∇E|| / Ha Å-1
        """

    @property
    def _has_coordinates_and_gradient(self) -> bool:
        """Does this optimiser have defined coordinates and a gradient?"""
        return self._coords is not None and self._coords.g is not None

    @property
    def _exceeded_maximum_iteration(self) -> bool:
        """
        Has this optimiser exceeded the maximum number of iterations
        allowed?

        -----------------------------------------------------------------------
        Returns:
            (bool):
        """
        if self.iteration >= self._maxiter:
            logger.warning(f'Reached the maximum number of iterations '
                           f'*{self._maxiter}*. Did not converge')
            return True

        else:
            return False


class NDOptimiser(Optimiser, ABC):
    """Abstract base class for an optimiser in N-dimensions"""

    def __init__(self,
                 maxiter: int,
                 gtol:    GradientNorm,
                 etol:    PotentialEnergy,
                 coords:  Optional[OptCoordinates] = None,
                 **kwargs):
        """
        Geometry optimiser. Signature follows that in scipy.minimize so
        species and method are keyword arguments. Converged when both energy
        and gradient criteria are met.

        ----------------------------------------------------------------------
        Arguments:
            maxiter (int): Maximum number of iterations to perform

            gtol (autode.values.GradientNorm): Tolerance on RMS(|∇E|)

            etol (autode.values.PotentialEnergy): Tolerance on |E_i+1 - E_i|

        See Also:

            :py:meth:`Optimiser <Optimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, coords=coords)

        self.etol = etol
        self.gtol = gtol

        self._hessian_update_types = [NullUpdate]

    @property
    def gtol(self) -> GradientNorm:
        """
        Gradient tolerance on |∇E| i.e. the root mean square of each component

        -----------------------------------------------------------------------
        Returns:
            (autode.values.GradientNorm):
        """
        return self._gtol

    @gtol.setter
    def gtol(self, value: Union[int, float, GradientNorm]):
        """Set the gradient tolerance"""

        if float(value) <= 0:
            raise ValueError('Tolerance on the gradient (||∇E||) must be '
                             f'positive. Had: gtol={value}')

        self._gtol = GradientNorm(value)

    @property
    def etol(self) -> PotentialEnergy:
        """
        Energy tolerance between two consecutive steps of the optimisation

        -----------------------------------------------------------------------
        Returns:
            (autode.values.PotentialEnergy): Energy tolerance
        """
        return self._etol

    @etol.setter
    def etol(self, value: Union[int, float, PotentialEnergy]):
        """Set the energy tolerance"""
        if float(value) <= 0:
            raise ValueError('Tolerance on the energy change is absolute so '
                             f'must be positive. Had etol = {value}')

        self._etol = PotentialEnergy(value)

    @classmethod
    def optimise(cls,
                 species: 'autode.species.Species',
                 method:  'autode.wrappers.base.Method',
                 maxiter: int = 100,
                 gtol:    Union[float, GradientNorm] = GradientNorm(1E-3, units='Ha Å-1'),
                 etol:    Union[float, PotentialEnergy] = PotentialEnergy(1E-4, units='Ha'),
                 coords:  Optional[OptCoordinates] = None,
                 n_cores: Optional[int] = None,
                 **kwargs
                 ) -> None:
        """
        Convenience function for constructing and running an optimiser

        ----------------------------------------------------------------------
        Arguments:
            species (autode.species.Species):

            method (autode.methods.Method):

        Keyword Arguments
            maxiter (int): Maximum number of iteration to perform

            gtol (float | autode.values.GradientNorm): Tolerance on RMS(|∇E|)
                 i.e. the root mean square of the gradient components. If
                 a float then assume units of Ha Å^-1

            etol (float | autode.values.PotentialEnergy): Tolerance on |∆E|
                 between two consecutive iterations of the optimiser

            kwargs (Any): Additional keyword arguments to pass to the
                          constructor

        Raises:
            (ValueError | RuntimeError):
        """

        optimiser = cls(maxiter=maxiter, gtol=gtol, etol=etol, coords=coords,
                        **kwargs)
        optimiser.run(species, method, n_cores=n_cores)

        return None

    @property
    def converged(self) -> bool:
        """
        Is this optimisation converged? Must be converged based on both energy
        and gradient tolerance.

        -----------------------------------------------------------------------
        Returns:
            (bool): Converged?
        """
        if self._abs_delta_e < self.etol / 10:
            logger.warning(f'Energy change is overachieved. '
                           f'{self.etol.to("kcal")/10:.3f} kcal mol-1. '
                           f'Signaling convergence')
            return True

        return self._abs_delta_e < self.etol and self._g_norm < self.gtol

    @property
    def _abs_delta_e(self) -> PotentialEnergy:
        """
        Calculate the absolute energy difference

        .. math::
            |∆E| = |E_i - E_{i-1}|   for a step i

        -----------------------------------------------------------------------
        Returns:
            (autode.values.PotentialEnergy): Energy difference. Infinity if
                                  an energy difference cannot be calculated
        """

        if len(self._history) < 2:
            logger.info('First iteration - returning |∆E| = ∞')
            return PotentialEnergy(np.inf)

        e1, e2 = self._coords.e, self._history.penultimate.e

        if e1 is None or e2 is None:
            raise RuntimeError('Cannot determine absolute energy difference')

        return PotentialEnergy(abs(e1 - e2))

    @property
    def _g_norm(self) -> GradientNorm:
        """
        Calculate RMS(∇E) based on the current Cartesian gradient.

        -----------------------------------------------------------------------
        Returns:
            (autode.values.GradientNorm): Gradient norm. Infinity if the
                                          gradient is not defined
        """
        if self._coords is None:
            logger.warning('Had no coordinates - cannot determine ||∇E||')
            return GradientNorm(np.inf)

        cartesian_gradient = self._coords.to('cart').g

        if cartesian_gradient is None:
            return GradientNorm(np.inf)

        return GradientNorm(np.sqrt(np.mean(np.square(cartesian_gradient))))

    def _log_convergence(self) -> None:
        """Log the convergence of the energy """
        logger.info(f'{self.iteration}\t'
                    f'{self._abs_delta_e.to("kcal mol-1"):.3f}\t'
                    f'{self._g_norm:.5f}')

        return None

    def _updated_h_inv(self) -> np.ndarray:
        r"""
        Update the inverse of the Hessian matrix :math:`H^{-1}` for the
        current set of coordinates. If the first iteration then use the true
        inverse of the (estimated) Hessian, otherwise update the inverse
        using a viable update strategy.


        .. math::

            H_{l - 1}^{-1} \rightarrow H_{l}^{-1}

        """

        if self.iteration == 0:
            logger.info('First iteration so using exact inverse, H^-1')
            return np.linalg.inv(self._coords.h)

        return self._best_hessian_updater.updated_h_inv

    def _updated_h(self) -> np.ndarray:
        r"""
        Update the the Hessian matrix :math:`H` for the current set of
        coordinates. If the first iteration then use the initial Hessian

        .. math::

            H_{l - 1} \rightarrow H_{l}

        """

        if self.iteration == 0:
            logger.info('First iteration so not updating the Hessian')
            return self._coords.h

        return self._best_hessian_updater.updated_h

    @property
    def _best_hessian_updater(self) -> 'HessianUpdater':
        """
        Find the best Hessian update strategy by enumerating all the possible
        Hessian update types implemented for this optimiser and returning the
        first that meets the criteria to be used.

        -----------------------------------------------------------------------
        Returns:
            (autode.opt.optimisers.hessian_update.HessianUpdater):

        Raises:
            (RuntimeError): If no suitable strategies are found
        """
        coords_l, coords_k = self._coords, self._history.penultimate

        for update_type in self._hessian_update_types:
            updater = update_type(h=coords_k.h,
                                  h_inv=coords_k.h_inv,
                                  s=coords_l.raw - coords_k.raw,
                                  y=coords_l.g - coords_k.g)

            if not updater.conditions_met:
                continue

            return updater

        raise RuntimeError('Could not update the inverse Hessian - no '
                           'suitable update strategies')


class _OptimiserHistory(list):
    """Sequential history of coordinates"""

    @property
    def penultimate(self) -> OptCoordinates:
        """
        Last but one set of coordinates (the penultimate set)

        -----------------------------------------------------------------------
        Returns:
            (autode.opt.OptCoordinates):
        """
        if len(self) < 2:
            raise IndexError('Cannot obtain the penultimate set of '
                             f'coordinates, only had {len(self)}')

        return self[-2]

    @property
    def final(self) -> OptCoordinates:
        """
        Last set of coordinates

        -----------------------------------------------------------------------
        Returns:
            (autode.opt.OptCoordinates):
        """
        if len(self) < 1:
            raise IndexError('Cannot obtain the final set of coordinates from '
                             'an empty history')

        return self[-1]

    @property
    def minimum(self) -> OptCoordinates:
        """
        Minimum energy coordinates in the history

        -----------------------------------------------------------------------
        Returns:
            (autode.opt.OptCoordinates):
        """
        if len(self) == 0:
            raise IndexError('No minimum with no history')

        return self[np.argmin([coords.e for coords in self])]

    @property
    def contains_well(self) -> bool:
        r"""Does this history contain a well in the energy?::

          |
        E |    -----   /          <-- Does contain a well
          |         \/
          |_________________
               Iteration

        -----------------------------------------------------------------------
        Returns:
            (bool): Presence of an explicit minima
        """

        for idx in range(len(self)-1):
            if self[idx].e < self[idx+1].e:
                return True

        return False
