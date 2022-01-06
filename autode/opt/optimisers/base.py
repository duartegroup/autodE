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
            maxiter (int): Maximum number of iterations to perform

        Keyword Arguments:
            coords (autode.opt.coordinates.OptCoordinates | None): Coordinates
                  to use in the optimisation e.g. CartesianCoordinates. If None
                  then will initialise the coordinates from _species
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

            if self.iteration >= self._maxiter:
                logger.warning(f'Reached the maximum number of iterations '
                               f'*{self._maxiter}*. Did not converge')
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
        """Initialise the internal species and method. They must have the
         correct attributes

        -----------------------------------------------------------------------
         Raises:
             (ValueError): For incorrect type or attributes
         """
        self._method = method if method is not None else self._method
        self._species = species if species is not None else self._species

        if self._species is None or self._method is None:
            raise ValueError('Must have a species and a method to run an '
                             f'optimisation. Had: {self._species} and '
                             f'{self._method}')

        if not all(hasattr(self._species, attr) for attr in ('energy', 'name')):
            raise ValueError('Internal species required energy and name '
                             f'attributes but had {self._species}')

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
        cart_coords = self._coords.to('cart')
        self._species.coordinates = cart_coords

        grad = Calculation(name=f'{self._species.name}_opt_{self.iteration}',
                           molecule=self._species,
                           method=self._method,
                           keywords=self._method.keywords.grad,
                           n_cores=self._n_cores)
        grad.run()

        self._coords.e = self._species.energy = grad.get_energy()
        self._species.gradient = grad.get_gradients()
        grad.clean_up(force=True, everything=True)

        self._coords.update_g_from_cart_g(self._species.gradient)
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


class NDOptimiser(Optimiser, ABC):
    """Abstract base class for an optimiser in N-dimensions"""

    def __init__(self,
                 maxiter: int,
                 gtol:    GradientNorm,
                 etol:    PotentialEnergy,
                 coords:  Optional[OptCoordinates] = None):
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
                 maxiter: int = 500,
                 gtol:    Union[float, GradientNorm] = GradientNorm(1E-3, units='Ha Å-1'),
                 etol:    Union[float, PotentialEnergy] = PotentialEnergy(1E-4, units='Ha'),
                 coords:  Optional[OptCoordinates] = None,
                 n_cores: Optional[int] = None,
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

            kwargs (Any): Additional keyword arguments to pass on
        """

        optimiser = cls(maxiter=maxiter, gtol=gtol, etol=etol, coords=coords)
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
        Calculate ||∇E|| based on the current Cartesian gradient.

        -----------------------------------------------------------------------
        Returns:
            (autode.values.GradientNorm): Gradient norm. Infinity if the
                                          gradient is not defined
        """
        if self._coords is None:
            logger.warning('Had no coordinates - cannot determine ||∇E||')
            return GradientNorm(np.inf)

        if self._coords.to('cart').g is None:
            return GradientNorm(np.inf)

        return GradientNorm(np.linalg.norm(self._coords.to('cart').g))

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
        using a viable update strategy


        .. math::

            H_{l - 1} \rightarrow H_{l}

        """

        if self.iteration == 0:
            logger.info('First iteration so using exact inverse, H^-1')
            return np.linalg.inv(self._coords.h)

        coords_l, coords_k = self._coords, self._history.penultimate

        for update_type in self._hessian_update_types:
            updater = update_type(h_inv=coords_k.h_inv,
                                  s=coords_l.raw - coords_k.raw,
                                  y=coords_l.g - coords_k.g)

            if not updater.conditions_met:
                continue

            return updater.updated_h_inv

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
            raise RuntimeError('No minimum with no history')

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
