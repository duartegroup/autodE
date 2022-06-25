"""Constrained rational function optimisation"""
import numpy as np
from autode.log import logger
from autode.opt.coordinates import CartesianCoordinates, DICWithConstraints
from autode.opt.coordinates.internals import InverseDistances
from autode.opt.optimisers.rfo import RFOOptimiser
from autode.opt.optimisers.hessian_update import BFGSUpdate
from autode.opt.coordinates.primitives import (InverseDistance,
                                               ConstrainedInverseDistance)


class CRFOptimiser(RFOOptimiser):

    def __init__(self,
                 init_alpha:    float = 0.1,
                 *args,
                 **kwargs):
        """
        TODO

        -----------------------------------------------------------------------
        Arguments:
            init_alpha: Maximum step size

        See Also:
            :py:meth:`RFOOptimiser <RFOOptimiser.__init__>`
        """
        super().__init__(*args, **kwargs)

        self.alpha = float(init_alpha)
        self._hessian_update_types = [BFGSUpdate]

    def _step(self) -> None:
        """Partitioned rational function step"""

        self._coords.h = self._updated_h()



        # self._coords = self._coords + self._sanitised_step(delta_s)
        return None

    def _initialise_run(self) -> None:
        """Initialise the optimisation"""
        self._set_initial_coordinates()
        self._update_hessian_gradient_and_energy()
        return None

    def _set_initial_coordinates(self):
        """Set the initial coordinates to optimise in, formed using
        delocalized internals"""

        if self._species is None:
            raise RuntimeError("Cannot set initial coordinates. No species set")

        cartesian_coords = CartesianCoordinates(self._species.coordinates)

        self._coords = DICWithConstraints.from_cartesian(
            x=cartesian_coords,
            primitives=self._primitives
        )
        self._coords.zero_lagrangian_multipliers()
        return None

    @property
    def _primitives(self) -> InverseDistances:
        """Primitive internal coordinates in this molecule"""

        pic = InverseDistances()

        for i in range(self._species.n_atoms):
            for j in range(i+1, self._species.n_atoms):
                if (i, j) in self._species.constraints.distance:
                    r = self._species.constraints.distance[(i, j)]
                    pic.append(
                        ConstrainedInverseDistance(i, j, value=1./r)
                    )
                else:
                    pic.append(InverseDistance(i, j))

        return pic
