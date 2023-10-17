"""
'Reactive' potential energy surfaces that have support for saddle point
and transition state guess finding
"""
import numpy as np
from abc import ABC

from typing import (
    Iterator,
    Tuple,
    Optional,
    Dict,
    List,
    Union,
    Sequence,
    TYPE_CHECKING,
)

from autode.pes.pes_nd import PESnD
from autode.log import logger
from autode.values import Distance
from autode.transition_states.ts_guess import TSguess
from autode.pes.mep import peak_point

if TYPE_CHECKING:
    from autode.species.species import Species
    from autode.transition_states.ts_guess import TSguess


class ReactivePESnD(PESnD, ABC):
    def __init__(
        self,
        species: "Species",
        rs: Dict[Tuple[int, int], Union[Tuple, np.ndarray]],
        allow_rounding: bool = True,
    ):
        """
        Reactive potential energy surface in N-dimensions

        -----------------------------------------------------------------------
        Arguments:
            species: Species from which to perform the PES exploration

            rs: Set of atom index pairs defining distances, along with a
                representation of what values they should take in the scan

            allow_rounding: Allow rounding of the step size, if required
        """
        PESnD.__init__(self, species, rs, allow_rounding)

        self._gradients = np.full(
            shape=(*self.shape, self.ndim), fill_value=np.nan
        )
        self._hessians = np.full(
            shape=(*self.shape, self.ndim, self.ndim), fill_value=np.nan
        )

    def ts_guesses(
        self,
        product: Optional["Species"] = None,
        min_separation: Distance = Distance(0.5, units="Å"),
    ) -> Iterator["TSguess"]:
        """
        Generate TS guesses from the saddle points in the energy on this
        surface. Only those that are seperated by at least min_separation will
        be yielded in order of increasing energy.

        -----------------------------------------------------------------------
        Arguments:
            product:

            min_separation: Minimum separation on the surface between TS guess
                            structures.

        Yields:
            (autode.transition_states.ts_guess.TSguess):
        """
        if not self._has_energy(self.origin):
            logger.warning(
                "Initial point on the PES not calculated - have "
                "no transition state guesses"
            )
            return StopIteration

        if product is not None:
            # Find the one TS guess by traversing the minimum energy pathway
            # from the origin species (reactant state) to a product
            try:
                yield next(self._mep_ts_guess(product=product))

            except StopIteration:
                logger.warning(
                    "Found no TS guesses from the minimum energy " "path"
                )

            return StopIteration

        assert self._coordinates is not None, "Must have set coordinates"
        assert self._species is not None, "Must have set species"

        yielded_p: List[tuple] = []

        for idx, point in enumerate(self._sorted_saddle_points()):
            if any(
                self._distance(p, point) < min_separation for p in yielded_p
            ):
                continue

            species = self._species.new_species(name=f"ts_guess{idx}")
            species.coordinates = self._coordinates[point]

            yielded_p.append(point)
            yield TSguess.from_species(species)

        return StopIteration

    def _stationary_points(
        self, threshold: float = 0.05, require_energy: bool = True
    ) -> Iterator[Tuple]:
        """
        Stationary points on the surface, characterised by a zero gradient
        vector. On a finite surface the gradient (g) will never truly vanish,
        so this function will only return those with |g| less than a threshold
        for which all surrounding points have a larger |g|. Is this a
        sufficient condition? who knows.

        -----------------------------------------------------------------------
        Arguments:
            threshold: Maximum |g| (Ha / Å-1) which to consider a
                       stationary point.

        Yields:
            (tuple(int)): Indices of a stationary point
        """
        self._set_gradients()

        for point in self._points():
            if require_energy and not self._has_energy(point):
                continue

            grad = self._gradients[point]

            if np.linalg.norm(grad) > threshold:
                continue

            if self._is_minimum_in_gradient(point):
                yield point

        return StopIteration

    def _saddle_points(self, threshold: float = 0.2) -> Iterator[Tuple]:
        """
        Find all the saddle points on the surface

        -----------------------------------------------------------------------
        Arguments:
            threshold: Threshold on |λ_i|/max(|λ|), below which they are set
                       to zero. Used to discard small negative eigenvectors
                       which would not be negative on a finer surface

        Yields:
            (tuple(int)): Indices of a saddle point
        """
        for point in self._stationary_points():
            self._set_hessian(point)

            if self._is_saddle_point(point, threshold=threshold):
                yield point

        return StopIteration

    def _sorted_saddle_points(self) -> Sequence[Tuple]:
        """
        Iterator of saddle points sorted by their energy (low -> high)

        -----------------------------------------------------------------------
        Returns:
            (Iterator):
        """
        return sorted(self._saddle_points(), key=lambda p: self._energies[p])

    def _mep_ts_guess(self, product: "Species") -> Iterator["TSguess"]:
        """
        Find a TS guess by traversing the minimum energy pathway (MEP) on a
        discreet potential energy surface between the initial species
        the 'reactant' state, to a new product state. Points are identified
        based on graph isomorphisms

        -----------------------------------------------------------------------
        Arguments:
            product: Product (hopefully) on the surface

        Yields:
            (autode.transition_states.ts_guess.TSguess): TS guess

        Returns:
            (StopIteration): If there are no suitable TS guesses
        """
        assert self._coordinates is not None, "Must have set coordinates"
        assert self._species is not None, "Must have set species"

        reactant = self._species

        if reactant.graph is None or product.graph is None:
            logger.warning(
                "Products or reactants did not have a defined "
                "graph, thus the MEP could not be traversed"
            )
            return StopIteration

        product_point = self._point_with_isomorphic_graph_to(product)

        if product_point is None:
            logger.warning(
                "Could not find any point on the surface that had "
                f"the same connectivity as {product}"
            )
            return StopIteration

        ts_point = peak_point(
            energies=self._energies, point1=self.origin, point2=product_point
        )

        species = self._species.new_species()
        species.coordinates = self._coordinates[ts_point]

        yield TSguess.from_species(species)

    def _point_with_isomorphic_graph_to(
        self, species: "Species"
    ) -> Optional[Tuple]:
        """
        Find a point on this surface that is graph-isomorphic to a particular
        species. Attempt to return the lowest energy point.

        -----------------------------------------------------------------------
        Arguments:
            species:

        Returns:
            (tuple(int, ..) | None):
        """
        assert self._coordinates is not None, "Must have set coordinates"
        assert self._species is not None, "Must have set species"

        isomorphic_points = []

        for point in self._points():
            mol = self._species.new_species()
            mol.coordinates = self._coordinates[point]
            mol.reset_graph()

            if mol.has_same_connectivity_as(species):
                isomorphic_points.append(point)

        if len(isomorphic_points) == 0:
            logger.warning("No isomorphic points found")
            return None

        min_idx = np.argmin([self._energies[p] for p in isomorphic_points])
        return isomorphic_points[min_idx]

    def _set_hessian(self, point: Tuple) -> None:
        """
        Set the Hessian for a particular point in the surface, evaluated
        using finite differences

        -----------------------------------------------------------------------
        Arguments:
            point:
        """
        hessian = self._hessians[point]

        for i in range(self.ndim):
            for j in range(i, self.ndim):
                # Point plus 1 (pp) and point minus 1 (pm) in this dimension
                pp, pm = self._neighbour(point, j, +1), self._neighbour(
                    point, j, -1
                )

                hessian[i, j] = (
                    self._gradients[pp][i] - self._gradients[pm][i]
                ) / (self._r(pp, j) - self._r(pm, j))

                # Hessians are symmetric
                hessian[j, i] = hessian[i, j]

        return None

    def _set_gradients(self) -> None:
        r"""
        Set the numerical gradient for each point on the surface, in each
        dimension.

        .. math::

            \frac{\text{d}E}{\text{d}r_i}
            = \frac{E(R + \Delta r) - E(R - \Delta r)}{\Delta r}

        where :math:`\Delta r` is the difference between two points on
        the surface and :math:`R` is the nuclear positions, :math:`r_i` is a
        dimension on the dimension on the surface. If possible, central
        differences are used otherwise forwards or backwards finite differences
        are used. If neither possible then the gradient is set as np.nan
        """
        for p in self._points():
            grad = self._gradients[p]  # Gradient with shape: (ndim,)

            if not self._has_energy(p):
                logger.warning(
                    f"Cannot set the gradient for point: {p} as it "
                    "did not have an energy"
                )
                grad.fill(np.nan)
                continue

            for n in range(self.ndim):
                pm, pp = self._neighbour(p, n, +1), self._neighbour(p, n, -1)

                if not self._has_energy(pm):
                    pm = p

                if not self._has_energy(pp):
                    pp = p

                if pm == pp:
                    logger.warning(
                        "Cannot determine gradient. Neither "
                        "neighbour had an energy"
                    )
                    grad[n] = np.nan
                    continue

                grad[n] = (self._energies[pp] - self._energies[pm]) / (
                    self._r(pp, n) - self._r(pm, n)
                )

        return None

    def _is_minimum_in_gradient(self, point: Tuple) -> bool:
        """
        Is a particular point surrounded by points with larger gradients?
        Only checks ±1 in each dimension, NOT combinations (i.e. diagonals)
        and uses the norm of the gradient (|g|).

        -----------------------------------------------------------------------
        Arguments:
            point:

        Returns:
            (bool):
        """
        norm_grad = np.linalg.norm(self._gradients[point])

        for n in range(self.ndim):
            pm, pp = self._neighbour(point, n, +1), self._neighbour(
                point, n, -1
            )

            if not (self._is_contained(pm) and self._is_contained(pp)):
                return False

            for grad in (self._gradients[pm], self._gradients[pp]):
                if np.any(np.isnan(grad)):
                    # Cannot determine if it is a minimum with undefined NN
                    return False

                if np.linalg.norm(grad) < norm_grad:
                    return False

        return True

    def _is_saddle_point(self, point: Tuple, threshold: float = 0.2) -> bool:
        """
        Is this point in the surface a saddle point. In the ideal case of a
        continuous surface a saddle point has a single negative eigenvalue
        of the Hessian matrix. However,

        -----------------------------------------------------------------------
        Arguments:
            point: Point on the surface. Hessian must be set here

            threshold: Threshold on |λ_i|/|max(λ)|, below which they are set
                       to zero. Used to discard spurious imaginary modes

        Returns:
            (bool): If this point is a saddle point, to within a threshold
        """

        eigenvals = np.linalg.eigvals(self._hessians[point])

        tol = np.max(np.abs(eigenvals)) * threshold

        # Eigenvalues (λ) with an absolute value less than the tolerance are 0
        eigenvals[np.abs(eigenvals) < tol] = 0.0

        logger.warning(f"{point} had eigenvalues: {eigenvals}")
        return len([x for x in eigenvals if x < 0]) == 1

    @property
    def _tensors(self) -> Sequence[np.ndarray]:
        """Tensors in this PES"""
        attrs = ("_energies", "_gradients", "_hessians")

        return [getattr(self, a) for a in attrs if hasattr(self, a)]
