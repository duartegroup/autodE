import os
import pickle
from dataclasses import dataclass
import numpy as np

from abc import ABC, abstractmethod
from zipfile import ZipFile, is_zipfile
from collections import deque
from typing import (
    Type,
    List,
    Union,
    Optional,
    Callable,
    Any,
    TYPE_CHECKING,
    Iterator,
)

from autode.log import logger
from autode.config import Config
from autode.values import GradientRMS, PotentialEnergy, method_string, Distance
from autode.opt.coordinates.base import OptCoordinates
from autode.opt.optimisers.hessian_update import NullUpdate
from autode.exceptions import CalculationException
from autode.plotting import plot_optimiser_profile

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.wrappers.methods import Method
    from autode.opt.coordinates import OptCoordinates
    from autode.opt.optimisers.hessian_update import HessianUpdater


class BaseOptimiser(ABC):
    """Base abstract class for an optimiser"""

    @property
    @abstractmethod
    def converged(self) -> bool:
        """Has this optimisation converged"""

    @property
    @abstractmethod
    def last_energy_change(self) -> PotentialEnergy:
        """The energy change on between the final two optimisation cycles"""

    def run(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @property
    def final_coordinates(self):
        raise NotImplementedError


class Optimiser(BaseOptimiser, ABC):
    """Abstract base class for an optimiser"""

    def __init__(
        self,
        maxiter: int,
        coords: Optional["OptCoordinates"] = None,
        callback: Optional[Callable] = None,
        callback_kwargs: Optional[dict] = None,
    ):
        """
        Optimiser

        ----------------------------------------------------------------------
        Arguments:
            maxiter: Maximum number of iterations to perform

            coords: Coordinates to use in the optimisation
                    e.g. CartesianCoordinates. If None then will initialise the
                    coordinates from _species

            callback: Function that will be called after every step. First
                      called after initialisation and before the first step.
                      Takes the current coordinates (which have energy (e),
                      gradient (g) and hessian (h) attributes) as the only
                      positional argument

            callback_kwargs: Keyword arguments to pass to the callback function
        """
        if int(maxiter) <= 0:
            raise ValueError(
                "An optimiser must be able to run at least one "
                f"step, but tried to set maxiter = {maxiter}"
            )

        self._callback = _OptimiserCallbackFunction(callback, callback_kwargs)

        self._maxiter = int(maxiter)
        self._n_cores: int = Config.n_cores

        self._history = OptimiserHistory()

        self._coords = coords
        self._species: Optional["Species"] = None
        self._method: Optional["Method"] = None

    @classmethod
    @abstractmethod
    def optimise(
        cls,
        species: "Species",
        method: "Method",
        n_cores: Optional[int] = None,
        coords: Optional[OptCoordinates] = None,
        **kwargs,
    ) -> None:
        """
        Optimise a species using a method

        .. code-block:: Python

          >>> import autode as ade
          >>> mol = ade.Molecule(smiles='C')
          >>> Optimiser.optimise(mol,method=ade.methods.ORCA())
        """

    @property
    def optimiser_params(self) -> dict:
        """
        The parameters which are needed to intialise the optimiser and
        will be saved in the optimiser trajectory
        """
        return {"maxiter": self._maxiter}

    def run(
        self,
        species: "Species",
        method: "Method",
        n_cores: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        Run the optimiser. Updates species.atoms and species.energy

        ----------------------------------------------------------------------
        Arguments:
            species: Species to optimise

            method: Method to use to calculate energies/gradients/hessians.
                    Calculations will use method.keywords.grad for gradient
                    calculations

            n_cores: Number of cores to use for calculations. If None then use
                     autode.Config.n_cores

            name: The name of the optimisation save file
        """
        self._n_cores = n_cores if n_cores is not None else Config.n_cores
        self._initialise_species_and_method(species, method)
        assert self._species is not None, "Species must be set"

        if not self._space_has_degrees_of_freedom:
            logger.info("Optimisation is in a 0D space – terminating")
            return None

        if name is None:
            name = f"{self._species.name}_opt_trj.zip"

        self._history.open(filename=name)
        self._history.save_opt_params(self.optimiser_params)
        self._initialise_run()

        logger.info(
            f"Using {self._method} to optimise {self._species.name} "
            f"with {self._n_cores} cores using {self._maxiter} max "
            f"iterations"
        )
        logger.info("Iteration\t|∆E| / \\kcal mol-1 \t||∇E|| / Ha Å-1")

        while not self.converged:
            self._callback(self._coords)
            self._step()  # Update self._coords
            self._update_gradient_and_energy()  # Update self._coords.g
            self._log_convergence()

            if self._exceeded_maximum_iteration:
                break

        logger.info(f"Converged: {self.converged}, in {self.iteration} cycles")
        self._history.close()
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

    def _initialise_species_and_method(
        self,
        species: "Species",
        method: "Method",
    ) -> None:
        """
        Initialise the internal species and method. They must be the correct
        types

        -----------------------------------------------------------------------
         Raises:
             (ValueError): For incorrect types
        """
        from autode.species.species import Species
        from autode.wrappers.methods import Method

        if not isinstance(species, Species):
            raise ValueError(
                f"{species} must be a autoode.Species instance "
                f"but had {type(species)}"
            )

        if not isinstance(method, Method):
            raise ValueError(
                f"{method} must be a autoode.wrappers.base.Method "
                f"instance but had {type(method)}"
            )

        if species.constraints.n_cartesian > 0:
            raise NotImplementedError

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
        assert self._species and self._coords is not None and self._method

        from autode.calculations import Calculation

        # TODO: species.calc_grad() method

        # Calculations need to be performed in cartesian coordinates
        if self._coords is not None:
            self._species.coordinates = self._coords.to("cart")

        grad = Calculation(
            name=f"{self._species.name}_opt_{self.iteration}",
            molecule=self._species,
            method=self._method,
            keywords=self._method.keywords.grad,
            n_cores=self._n_cores,
        )
        grad.run()
        grad.clean_up(force=True, everything=True)

        if self._species.gradient is None:
            raise CalculationException(
                "Calculation failed to calculate a gradient. "
                "Cannot continue!"
            )

        self._coords.e = self._species.energy
        self._coords.update_g_from_cart_g(arr=self._species.gradient)
        return None

    def _update_hessian_gradient_and_energy(self) -> None:
        """
        Update the energy, gradient and Hessian using the method. Will
        transform from the current coordinates type to Cartesian coordinates
        to perform the calculation, then back. Uses a numerical Hessian if
        analytic Hessians are not implemented for this method. Does not
        perform a Hessian evaluation if the molecule's energy is evaluated
        at the same level of theory that would be used for the Hessian
        evaluation.

        -----------------------------------------------------------------------
        Raises:
            (autode.exceptions.CalculationException):
        """
        assert self._species and self._coords is not None and self._method
        should_calc_hessian = True

        if (
            _energy_method_string(self._species)
            == method_string(self._method, self._method.keywords.hess)
            and self._species.hessian is not None
        ):
            logger.info(
                "Have a calculated the energy at the same level of "
                "theory as this optimisation and a present Hessian. "
                "Not calculating a new Hessian"
            )
            should_calc_hessian = False

        self._update_gradient_and_energy()

        if should_calc_hessian:
            self._update_hessian()
        else:
            self._coords.update_h_from_cart_h(
                self._species.hessian.to("Ha Å^-2")  # type: ignore
            )
        return None

    def _update_hessian(self) -> None:
        """Update the Hessian of a species"""
        assert self._species and self._coords is not None and self._method

        species = self._species.new_species(
            name=f"{self._species.name}_opt_{self.iteration}"
        )
        species.coordinates = self._coords.to("cartesian")

        species.calc_hessian(
            method=self._method,
            keywords=self._method.keywords.hess,
            n_cores=self._n_cores,
        )
        assert species.hessian is not None, "Failed to calculate H"

        self._species.hessian = species.hessian.copy()
        self._coords.update_h_from_cart_h(self._species.hessian.to("Ha Å^-2"))

    @property
    def _space_has_degrees_of_freedom(self) -> bool:
        """Does this optimisation space have any degrees of freedom"""
        return True

    @property
    def _coords(self) -> Optional[OptCoordinates]:
        """
        Current set of coordinates this optimiser is using
        """
        if len(self._history) == 0:
            logger.warning("Optimiser had no history, thus no coordinates")
            return None

        return self._history.final

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
            self._history.add(value.copy())

        else:
            raise ValueError(
                "Cannot set the optimiser coordinates with " f"{value}"
            )

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

    @property
    def last_energy_change(self) -> PotentialEnergy:
        """Last ∆E found in this"""

        if self.iteration > 0:
            final_e = self._history.final.e
            penultimate_e = self._history.penultimate.e
            if final_e is not None and penultimate_e is not None:
                return PotentialEnergy(final_e - penultimate_e, units="Ha")

        if self.converged:
            logger.warning(
                "Optimiser was converged in less than two "
                "cycles. Assuming an energy change of 0"
            )
            return PotentialEnergy(0)

        return PotentialEnergy(np.inf)

    @property
    def final_coordinates(self) -> Optional["OptCoordinates"]:
        return None if len(self._history) == 0 else self._history.final

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
            logger.warning(
                f"Reached the maximum number of iterations "
                f"*{self._maxiter}*. Did not converge"
            )
            return True

        else:
            return False


class NullOptimiser(BaseOptimiser):
    """An optimiser that does nothing"""

    @property
    def converged(self) -> bool:
        return False

    @property
    def last_energy_change(self) -> PotentialEnergy:
        return PotentialEnergy(np.inf)

    def run(self, **kwargs: Any) -> None:
        pass

    @property
    def final_coordinates(self):
        raise RuntimeError("A NullOptimiser has no coordinates")


class NDOptimiser(Optimiser, ABC):
    """Abstract base class for an optimiser in N-dimensions"""

    def __init__(
        self,
        maxiter: int,
        conv_tol: "ConvergenceParams",
        coords: Optional[OptCoordinates] = None,
        **kwargs,
    ):
        """
        Geometry optimiser. Signature follows that in scipy.minimize so
        species and method are keyword arguments. Converged when both energy
        and gradient criteria are met.

        ----------------------------------------------------------------------
        Arguments:
            maxiter (int): Maximum number of iterations to perform

            conv_tol (ConvergenceParams): Convergence tolerances, indicating
                    thresholds for absolute energy change (|E_i+1 - E_i|),
                    RMS and max. gradients (∇E) and RMS and max. step size (Δx)

        See Also:

            :py:meth:`Optimiser <Optimiser.__init__>`
        """
        super().__init__(maxiter=maxiter, coords=coords, **kwargs)

        self.conv_tol = conv_tol
        self._hessian_update_types: List[Type[HessianUpdater]] = [NullUpdate]

    @property
    def conv_tol(self) -> "ConvergenceParams":
        """
        All convergence parameters for this optimiser. If numerical
        values are unset, they appear as infinity.

        Returns:
            (ConvergenceParams):
        """
        return self._conv_tol

    @conv_tol.setter
    def conv_tol(self, value):
        """
        Set the convergence parameters for this optimiser.

        Args:
            value:

        Returns:

        """
        # TODO: handle dicts?
        self._conv_tol = value

    @property
    def gtol(self) -> GradientRMS:
        """
        Gradient tolerance on |∇E| i.e. the root mean square of each component

        -----------------------------------------------------------------------
        Returns:
            (autode.values.GradientRMS):
        """
        return self._gtol

    @gtol.setter
    def gtol(self, value: Union[int, float, GradientRMS]):
        """Set the gradient tolerance"""

        if float(value) <= 0:
            raise ValueError(
                "Tolerance on the gradient (||∇E||) must be "
                f"positive. Had: gtol={value}"
            )

        self._gtol = GradientRMS(value)

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
            raise ValueError(
                "Tolerance on the energy change is absolute so "
                f"must be positive. Had etol = {value}"
            )

        self._etol = PotentialEnergy(value)

    @property
    def optimiser_params(self):
        """Optimiser params to save"""
        return {"maxiter": self._maxiter, "gtol": self.gtol, "etol": self.etol}

    @classmethod
    def optimise(
        cls,
        species: "Species",
        method: "Method",
        n_cores: Optional[int] = None,
        coords: Optional[OptCoordinates] = None,
        maxiter: int = 100,
        gtol: Any = GradientRMS(1e-3, units="Ha Å-1"),
        etol: Any = PotentialEnergy(1e-4, units="Ha"),
        **kwargs,
    ) -> None:
        """
        Convenience function for constructing and running an optimiser

        ----------------------------------------------------------------------
        Arguments:
            species (Species):

            method (autode.methods.Method):

            maxiter (int): Maximum number of iteration to perform

            gtol (float | autode.values.GradientNorm): Tolerance on RMS(|∇E|)
                 i.e. the root mean square of the gradient components. If
                 a float then assume units of Ha Å^-1

            etol (float | autode.values.PotentialEnergy): Tolerance on |∆E|
                 between two consecutive iterations of the optimiser

            coords (OptCoordinates | None): Coordinates to optimise in

            n_cores (int | None): Number of cores to run energy/gradient/hessian
                                  evaluations. If None then use ade.Config.n_cores

            kwargs (Any): Additional keyword arguments to pass to the
                          constructor

        Raises:
            (ValueError | RuntimeError):
        """

        optimiser = cls(
            maxiter=maxiter, gtol=gtol, etol=etol, coords=coords, **kwargs
        )
        optimiser.run(species, method, n_cores=n_cores)

        return None

    @property
    def _space_has_degrees_of_freedom(self) -> bool:
        return True if self._species is None else self._species.n_atoms > 1

    @property
    def converged(self) -> bool:
        """
        Is this optimisation converged? Must be converged based on both energy
        and gradient tolerance.

        -----------------------------------------------------------------------
        Returns:
            (bool): Converged?
        """
        if self._species is not None and self._species.n_atoms == 1:
            return True  # Optimisation 0 DOF is always converged

        if self._abs_delta_e < self.etol / 10:
            logger.warning(
                f"Energy change is overachieved. "
                f'{self.etol.to("kcal") / 10:.3E} kcal mol-1. '
                f"Signaling convergence"
            )
            return True

        return self._abs_delta_e < self.etol and self._g_norm < self.gtol

    def clean_up(self) -> None:
        """
        Clean up by removing the trajectory file on disk
        """
        self._history.clean_up()

    @classmethod
    def from_file(cls, filename: str) -> "NDOptimiser":
        """
        Create an optimiser from a trajectory file i.e. reload a saved state
        """
        hist = OptimiserHistory.load(filename)
        optimiser = cls(**hist.get_opt_params())
        optimiser._history = hist
        return optimiser

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
        assert (
            self._coords is not None
        ), "Must have coordinates to calculate ∆E"

        if len(self._history) < 2:
            logger.info("First iteration - returning |∆E| = ∞")
            return PotentialEnergy(np.inf)

        e1, e2 = self._coords.e, self._history.penultimate.e

        if e1 is None or e2 is None:
            logger.error(
                "Cannot determine absolute energy difference. Using |∆E| = ∞"
            )
            return PotentialEnergy(np.inf)

        return PotentialEnergy(abs(e1 - e2))  # type: ignore

    @property
    def _g_norm(self) -> GradientRMS:
        """
        Calculate RMS(∇E) based on the current Cartesian gradient.

        -----------------------------------------------------------------------
        Returns:
            (autode.values.GradientRMS): Gradient norm. Infinity if the
                                          gradient is not defined
        """
        if self._coords is None:
            logger.warning("Had no coordinates - cannot determine ||∇E||")
            return GradientRMS(np.inf)

        if self._coords.g is None:
            return GradientRMS(np.inf)

        return GradientRMS(np.sqrt(np.mean(np.square(self._coords.g))))

    def _log_convergence(self) -> None:
        """Log the convergence of the energy"""
        assert (
            self._coords is not None
        ), "Must have coordinates to log convergence"
        log_string = f"{self.iteration}\t"

        if len(self._history) > 1:
            assert self._coords.e and self._history.penultimate.e, "Need ∆E"
            de: PotentialEnergy = self._coords.e - self._history.penultimate.e
            log_string += f'{de.to("kcal mol-1"):.3f}\t{self._g_norm:.5f}'

        logger.info(log_string)
        return None

    def plot_optimisation(
        self,
        filename: Optional[str] = None,
        plot_energy: bool = True,
        plot_rms_grad: bool = True,
    ) -> None:
        """
        Draw the plot of the energies and/or rms_gradients of
        the optimisation so far

        ----------------------------------------------------------------------
        Args:
            filename (str): Name of the file to plot
            plot_energy (bool): Whether to plot energy
            plot_rms_grad (bool): Whether to plot RMS of gradient
        """
        assert self._species is not None, "Must have a species to plot"

        if self.iteration < 1:
            logger.warning("Less than 2 points, cannot draw optimisation plot")
            return None

        if not self.converged:
            logger.warning(
                "Optimisation is not converged, drawing a plot "
                "of optimiser profile until current iteration"
            )

        filename = (
            f"{self._species.name}_opt_plot.pdf"
            if filename is None
            else str(filename)
        )

        plot_optimiser_profile(
            self._history,
            filename=filename,
            plot_energy=plot_energy,
            plot_rms_grad=plot_rms_grad,
        )
        return None

    def print_geometries(self, filename: Optional[str] = None) -> None:
        """
        Writes the trajectory of the optimiser in .xyz format

        Args:
            filename (str|None): Name of the trajectory file (xyz),
                                 if not given, generates name from species
        """
        assert self._species is not None
        if self.iteration < 1:
            logger.warning(
                "Optimiser did no steps, not saving .xyz trajectory"
            )
            return None

        filename = (
            f"{self._species.name}_opt.trj.xyz"
            if filename is None
            else str(filename)
        )
        print_geometries_from(
            self._history, species=self._species, filename=filename
        )
        return None


@dataclass
class ConvergenceParams:
    """Various convergence parameters for Optimisers"""

    abs_d_e: PotentialEnergy
    rms_g: GradientRMS
    max_g: GradientRMS
    rms_s: Distance
    max_s: Distance
    strict: bool = False

    def meets_criteria(self, other: "ConvergenceParams") -> bool:
        """
        Check if the given values for the parameters meets the
        current set of criteria

        Args:
            other (ConvergenceParams): Another set of parameters

        Returns:
            (bool): True if criteria met, converged otherwise

        """
        abs_d_e = other.abs_d_e
        rms_g, max_g = other.rms_g, other.max_g
        rms_s, max_s = other.rms_s, other.max_s

        # everything converged
        if (
            abs_d_e < self.abs_d_e
            and rms_g < self.rms_g
            and max_g < self.max_g
            and rms_s < self.rms_s
            and max_s < self.max_s
        ):
            return True

        # Strict criteria
        if self.strict:
            return False

        # gradient, energy overachieved and step reasonably converged
        if (
            abs_d_e < self.abs_d_e / 2
            and rms_g < self.rms_g / 2
            and max_g < self.max_g / 1.5
            and rms_s < self.rms_s * 5
            and max_s < self.max_s * 10
        ):
            logger.warning(
                "Overachieved gradient and energy convergence, reasonable "
                "convergence on step size."
            )
            return True

        # only gradient criteria overachieved
        if (
            abs_d_e < self.abs_d_e * 1.5
            and rms_g < self.rms_g / 10
            and max_g < self.max_g / 5
            and rms_s < self.rms_s * 5
            and max_s < self.max_s * 5
        ):
            logger.warning(
                "Energy is almost converged. Gradient is one order of "
                "magnitude below convergence. Step size almost converged."
            )
            return True

        # everything except energy is converged
        if (
            abs_d_e < self.abs_d_e * 3
            and rms_g < self.rms_g
            and max_g < self.max_g
            and rms_s < self.rms_s
            and max_s < self.max_s
        ):
            logger.warning(
                "Gradients and step sizes have converged, reasonable "
                "convergence on energy."
            )
            return True

        return False


class OptimiserConvergence:
    """
    Class that defines some common convergence criteria for optimisation
    and parses user-provided convergence parameters
    """

    # NOTE: Taken from ORCA
    LOOSE = ConvergenceParams(
        abs_d_e=PotentialEnergy(3e-5, "Ha"),
        rms_g=GradientRMS(5e-4, "Ha/bohr").to("Ha/ang"),
        max_g=GradientRMS(2e-3, "Ha/bohr").to("Ha/ang"),
        rms_s=Distance(7e-3, "bohr").to("ang"),
        max_s=Distance(1e-2, "bohr").to("ang"),
    )
    NORMAL = ConvergenceParams(
        abs_d_e=PotentialEnergy(5e-6, "Ha"),
        rms_g=GradientRMS(1e-4, "Ha/bohr").to("Ha/ang"),
        max_g=GradientRMS(3e-4, "Ha/bohr").to("Ha/ang"),
        rms_s=Distance(2e-3, "bohr").to("ang"),
        max_s=Distance(4e-3, "bohr").to("ang"),
    )
    TIGHT = ConvergenceParams(
        abs_d_e=PotentialEnergy(1e-6, "Ha"),
        rms_g=GradientRMS(3e-5, "Ha/bohr").to("Ha/ang"),
        max_g=GradientRMS(1e-4, "Ha/bohr").to("Ha/ang"),
        rms_s=Distance(6e-4, "bohr").to("ang"),
        max_s=Distance(1e-3, "bohr").to("ang"),
    )
    VERYTIGHT = ConvergenceParams(
        abs_d_e=PotentialEnergy(2e-7, "Ha"),
        rms_g=GradientRMS(8e-6, "Ha/bohr").to("Ha/ang"),
        max_g=GradientRMS(3e-5, "Ha/bohr").to("Ha/ang"),
        rms_s=Distance(1e-4, "bohr").to("ang"),
        max_s=Distance(2e-4, "bohr").to("ang"),
    )

    @staticmethod
    def params_from_dict(options: dict) -> ConvergenceParams:
        """
        Convert a user supplied dictionary into valid convergence
        parameters

        Args:
            options (dict):

        Returns:
            (ConvergenceParams):
        """
        assert isinstance(options, dict)
        possible_attrs = [
            "abs_d_e",
            "rms_g",
            "max_g",
            "rms_s",
            "max_s",
            "strict",
        ]

        valid_options = {}
        for attr in possible_attrs:
            valid_options[attr] = options.get(attr, None)

        #  valid criteria must at least have RMS grad and delta E
        if valid_options["abs_d_e"] is None or valid_options["rms_g"] is None:
            raise ValueError(
                "Convergence criteria must define at least the absolute"
                "energy change tolerance (abs_d_e) and RMS gradient"
                "tolerance (rms_g)"
            )
        numerical_attrs = possible_attrs[:-1]
        # unset numerical criteria are set to infinity i.e. not considered
        for attr in numerical_attrs:
            if valid_options[attr] is None:
                valid_options[attr] = float("inf")

        return ConvergenceParams(**valid_options)


class OptimiserHistory:
    """
    Sequential trajectory of coordinates with a maximum length for
    storage on memory. Shunts data to disk if trajectory file is
    opened, otherwise old coordinates more than the maximum number
    are lost.
    """

    def __init__(self, maxlen: Optional[int] = 2) -> None:
        self._filename: Optional[str] = None  # filename with abs. path
        self._memory: deque = deque(maxlen=maxlen)  # coords in mem
        self._maxlen = maxlen if maxlen is not None else float("inf")
        self._is_closed = False  # whether trajectory is open
        self._len = 0  # count of total number of coords

    @property
    def final(self):
        """
        Last set of coordinates in memory

        -----------------------------------------------------------------------
        Returns:
            (OptCoordinates):
        """
        if len(self._memory) < 1:
            raise IndexError(
                "Cannot obtain the final set of "
                f"coordinates, memory is empty"
            )
        return self._memory[-1]

    @property
    def penultimate(self):
        """
        Last but one set of coordinates from memory (the penultimate set)

        -----------------------------------------------------------------------
        Returns:
            (OptCoordinates):
        """
        if len(self._memory) < 2:
            raise IndexError(
                "Cannot obtain the penultimate set of "
                f"coordinates, only had {len(self._memory)}"
            )
        return self._memory[-2]

    @property
    def _n_stored(self) -> int:
        """Number of coordinates stored on disk"""
        if self._filename is None:
            return 0

        with ZipFile(self._filename, "r") as file:
            names = file.namelist()
        n_coords = 0
        for name in names:
            if name.startswith("coords_") and int(name[7:]) >= 0:
                n_coords += 1
        return n_coords

    def __len__(self):
        """How many coordinates have been put into this trajectory"""
        return self._len

    def open(self, filename: str):
        """
        Initialise the trajectory file and write it on disk.

        Args:
            filename (str): The name of the trajectory file,
                            should be .zip, and NOT a path
        """
        if self._filename is not None:
            raise RuntimeError("Already initialised, cannot initialise again!")

        # filename should not be a path
        assert "\\" not in filename and "/" not in filename
        if not filename.lower().endswith(".zip"):
            filename += ".zip"

        if os.path.exists(filename):
            logger.warning(f"File {filename} already exists, overwriting")
            os.remove(filename)

        # get the full path so that it is robust to os.chdir
        self._filename = os.path.abspath(filename)

        # write a header like file to help identify
        with ZipFile(filename, "w") as file:
            with file.open("ade_opt_trj", "w") as fh:
                fh.write("Trajectory from autodE".encode("utf-8"))
        return None

    @classmethod
    def load(cls, filename: str):
        """
        Reload the state of the trajectory from a file

        Args:
            filename: The name of the trajectory .zip file,
                    could also be a relative path

        Returns:

        """
        trj = cls()
        if not filename.lower().endswith(".zip"):
            filename += ".zip"
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"The file {filename} does not exist!")
        if not is_zipfile(filename):
            raise ValueError(
                f"The file {filename} is not a valid trajectory file"
            )
        with ZipFile(filename, "r") as file:
            names = file.namelist()
            if "ade_opt_trj" not in names:
                raise ValueError(
                    f"The file {filename} is not an autodE trajectory!"
                )
        # handle paths with env vars or w.r.t. home dir
        trj._filename = os.path.abspath(
            os.path.expanduser(os.path.expandvars(filename))
        )
        trj._len = trj._n_stored
        trj._is_closed = True
        # load the last two into memory
        if trj._len < 2:
            load_idxs = [trj._len - 1]
        else:
            load_idxs = [trj._len - 2, trj._len - 1]
        with ZipFile(trj._filename, "r") as file:
            for idx in load_idxs:
                with file.open(f"coords_{idx}") as fh:
                    trj._memory.append(pickle.load(fh))

        return trj

    def clean_up(self):
        """Remove the disk file associated with this history"""
        os.remove(self._filename)
        return None

    def save_opt_params(self, params: dict):
        """
        Save optimiser parameters given as a dict into the trajectory
        savefile

        Args:
            params (dict):
        """
        assert isinstance(params, dict)
        if self._filename is None:
            raise RuntimeError("File not opened - cannot store data")

        # python's ZipFile does not allow overwriting files
        with ZipFile(self._filename, "a") as file:
            names = file.namelist()
            if "opt_params" in names:
                raise FileExistsError(
                    "Optimiser parameters are already stored -"
                    " cannot overwrite!"
                )
            with file.open("opt_params", "w") as fh:
                pickle.dump(params, fh, pickle.HIGHEST_PROTOCOL)

        return None

    def get_opt_params(self) -> dict:
        """
        Retrieve the stored optimiser parameters from the trajectory
        file

        Returns:
            (dict): Dictionary of optimiser parameters
        """
        if self._filename is None:
            raise RuntimeError("File not opened - cannot get data")

        # python's ZipFile does not allow overwriting files
        with ZipFile(self._filename, "r") as file:
            names = file.namelist()
            if "opt_params" not in names:
                raise FileNotFoundError("Optimiser parameters are not found!")
            with file.open("opt_params", "r") as fh:
                data = pickle.load(fh)

        return data

    def add(self, coords: Optional["OptCoordinates"]) -> None:
        """
        Add a new set of coordinates to this trajectory

        Args:
            coords (OptCoordinates): The set of coordinates to be added
        """
        if coords is None:
            return None
        elif not isinstance(coords, OptCoordinates):
            raise ValueError("item added must be OptCoordinates")

        if self._is_closed:
            raise RuntimeError("Cannot add to closed OptimiserHistory")

        self._len += 1
        # check if we need to push last coords to disk or can skip
        if len(self._memory) < self._maxlen or self._filename is None:
            self._memory.append(coords)
            return None

        n_stored = self._n_stored
        with ZipFile(self._filename, "a") as file:
            with file.open(f"coords_{n_stored}", "w") as fh:
                pickle.dump(self._memory[0], fh, pickle.HIGHEST_PROTOCOL)
        self._memory.append(coords)
        return None

    def close(self):
        """
        Close the Optimiser history by putting the coordinates still
        in memory onto disk
        """
        if self._filename is None:
            raise RuntimeError("Cannot close - had no trajectory file!")

        idx = self._n_stored
        with ZipFile(self._filename, "a") as file:
            for coords in self._memory:
                with file.open(f"coords_{idx}", "w") as fh:
                    pickle.dump(coords, fh, pickle.HIGHEST_PROTOCOL)
                idx += 1

        self._is_closed = True
        return None

    def __getitem__(self, item: int) -> Optional[OptCoordinates]:
        """
        Access a coordinate from this trajectory, either from stored
        data on disk, or from the memory. Only returns Cartesian
        coordinates to ensure type consistency.

        Args:
            item (int): Must be integer and not a slice

        Returns:
            (CartesianCoordinates|None): The coordinates if found, None
                        if the file does not exist and coordinate is not
                        in memory

        Raises:
            NotImplementedError: If slice is used
            IndexError: If requested index does not exist
        """
        if isinstance(item, slice):
            raise NotImplementedError
        elif isinstance(item, int):
            pass
        else:
            raise ValueError("Index has to be type int")

        if item < 0:
            item += self._len
        if item < 0 or item >= self._len:
            raise IndexError("Array index out of range")

        # read directly from memory if possible
        if item >= (self._len - self._maxlen):
            return self._memory[item - self._len]

        # have to read from disk now, return None if no file
        if self._filename is None:
            return None

        with ZipFile(self._filename, "r") as file:
            with file.open(f"coords_{item}") as fh:
                coords = pickle.load(fh)

        return coords

    def __iter__(self):
        """
        Iterate through the coordinates of this trajectory
        """
        for i in range(len(self)):
            yield self[i]

    def __reversed__(self):
        """
        Reversed iteration through the coordinates
        """
        for i in reversed(range(len(self))):
            yield self[i]


class ExternalOptimiser(BaseOptimiser, ABC):
    @property
    @abstractmethod
    def converged(self) -> bool:
        """Has this optimisation has converged"""

    @property
    @abstractmethod
    def last_energy_change(self) -> PotentialEnergy:
        """The final energy change in this optimisation"""


class _OptimiserCallbackFunction:
    def __init__(self, f: Optional[Callable], kwargs: Optional[dict]):
        """Callback function initializer"""

        self._f = f
        self._kwargs = kwargs if kwargs is not None else dict()

    def __call__(self, coordinates: Optional[OptCoordinates]) -> Any:
        """Call the function, if it exists"""

        if self._f is None:
            return None

        logger.info("Calling callback function")
        return self._f(coordinates, **self._kwargs)


def _energy_method_string(species: "Species") -> str:
    return "" if species.energy is None else species.energy.method_str


def print_geometries_from(
    coords_trj: Union[Iterator[OptCoordinates], OptimiserHistory],
    species: "Species",
    filename: str,
) -> None:
    """
    Print geometries from an iterator over a series of coordinates

    Args:
        coords_trj: The iterator for coordinates, can be OptimiserHistory
        species: The Species for which the coordinate history is generated
        filename: Name of file
    """
    from autode.species import Species

    assert isinstance(species, Species)

    if not filename.lower().endswith(".xyz"):
        filename = filename + ".xyz"

    if os.path.isfile(filename):
        logger.warning(f"{filename} already exists, overwriting...")
        os.remove(filename)

    # take a copy so that original is not modified
    tmp_spc = species.copy()
    for coords in coords_trj:
        tmp_spc.coordinates = coords.to("cart")
        tmp_spc.energy = coords.e
        tmp_spc.print_xyz_file(filename=filename, append=True)

    return None
