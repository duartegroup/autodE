# mypy: disable-error-code="has-type"
import numpy as np
from copy import deepcopy
from typing import Optional, Union, Sequence, List, TYPE_CHECKING
from abc import ABC, abstractmethod

from autode.log import logger
from autode.units import ang, nm, pm, m
from autode.values import ValueArray, PotentialEnergy

if TYPE_CHECKING:
    from autode.units import Unit
    from autode.values import Gradient
    from autode.hessians import Hessian


class OptCoordinates(ValueArray, ABC):
    """Coordinates used to perform optimisations"""

    implemented_units = [ang, nm, pm, m]

    def __new__(
        cls,
        input_array: Union[Sequence, np.ndarray],
        units: Union[str, "Unit"],
    ) -> "OptCoordinates":
        """New instance of these coordinates"""

        arr = super().__new__(cls, np.array(input_array), units)

        arr._e = None  # Energy
        arr._g = None  # Gradient: dE/dX
        arr._h = None  # Hessian:  d2E/dX_idX_j
        arr._h_inv = None  # Inverse of the Hessian: H^-1
        arr.B = None  # Wilson B matrix
        arr.B_T_inv = None  # Generalised inverse of B
        arr.U = np.eye(len(arr))  # Transform matrix
        arr.allow_unconverged_back_transform = True  # for internal coords

        return arr

    def __array_finalize__(self, obj: "OptCoordinates") -> None:
        """See https://numpy.org/doc/stable/user/basics.subclassing.html"""

        # fmt: off
        for attr in (
            "units",    "_e",   "_g",   "_h",
            "_h_inv",   "U",    "B",    "B_T_inv",
            "allow_unconverged_back_transform",
        ):
            self.__dict__[attr] = getattr(obj, attr, None)

        # fmt: on
        return None

    @property
    def indexes(self) -> List[int]:
        """Indexes of the coordinates in this set"""
        return list(range(len(self)))

    @property
    def raw(self) -> np.ndarray:
        """Raw numpy array of these coordinates"""
        return np.array(self, copy=True)

    @property
    def e(self) -> Optional[PotentialEnergy]:
        """
        Energy

        -----------------------------------------------------------------------
        Returns:
            (PotentialEnergy | None): E
        """
        return self._e

    @e.setter
    def e(self, value):
        """Set the energy"""
        self._e = None if value is None else PotentialEnergy(value)

    @property
    def g(self) -> Optional[np.ndarray]:
        r"""
        Gradient of the energy

        .. math::
            G = \nabla E
                \equiv
                \left\{\frac{\partial E}{\partial\boldsymbol{R}_{i}}\right\}

        where :math:`\boldsymbol{R}` are a general vector of coordinates.

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray | None): G
        """
        return self._g

    @g.setter
    def g(self, value: np.ndarray):
        """Set the gradient of the energy"""
        self._g = value

    @property
    def h(self) -> Optional[np.ndarray]:
        r"""
        Hessian (second derivative) matrix of the energy

        .. math::
            H = \begin{pmatrix}
                \frac{\partial^2 E}
                     {\partial\boldsymbol{R}_{0}\partial\boldsymbol{R}_{0}}
                & \cdots \\
                \vdots & \ddots
                \end{pmatrix}

        where :math:`\boldsymbol{R}` are a general vector of coordinates.

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray | None): H
        """
        if self._h is None and self._h_inv is not None:
            logger.info("Have H^-1 but no H, calculating H")
            self._h = np.linalg.inv(self._h_inv)

        return self._h

    @h.setter
    def h(self, value: np.ndarray):
        """Set the second derivatives of the energy"""
        if not self.h_or_h_inv_has_correct_shape(value):
            raise ValueError(
                f"Hessian must be an NxN matrix. "
                f"Had an array with shape: {value.shape}"
            )

        self._h = value

    @property
    def h_inv(self) -> Optional[np.ndarray]:
        """
        Inverse of the Hessian matrix

        .. math:: H^{-1}

        -----------------------------------------------------------------------
        Returns:
            (np.ndarray | None): H^{-1}
        """

        if self._h_inv is None and self._h is not None:
            logger.info(
                "Have Hessian but no inverse, so calculating "
                "explicit inverse"
            )
            self._h_inv = np.linalg.inv(self._h)

        return self._h_inv

    @h_inv.setter
    def h_inv(self, value: np.ndarray):
        """Set the inverse hessian matrix"""
        if not self.h_or_h_inv_has_correct_shape(value):
            raise ValueError(
                "Inverse Hessian must be an NxN matrix. "
                f"Had an array with shape: {value.shape}"
            )

        self._h_inv = value

    def h_or_h_inv_has_correct_shape(self, arr: Optional[np.ndarray]):
        """Does a Hessian or its inverse have the correct shape?"""
        if arr is None:
            return True  # None is always valid

        return arr.ndim == 2 and arr.shape[0] == arr.shape[1] == len(self)

    @abstractmethod
    def _update_g_from_cart_g(self, arr: Optional["Gradient"]) -> None:
        """Update the gradient dE/dR of from a Cartesian (cart) gradient"""

    def update_g_from_cart_g(
        self,
        arr: Optional["Gradient"],
    ) -> None:
        """Update the gradient from a Cartesian gradient, zeroing those atoms
        that are constrained"""
        assert (
            arr is None or len(arr.flatten()) % 3 == 0
        )  # Needs an Nx3 matrix

        return self._update_g_from_cart_g(arr)

    @abstractmethod
    def _update_h_from_cart_h(self, arr: Optional["Hessian"]) -> None:
        """Update the Hessian from a cartesian Hessian"""

    def update_h_from_cart_h(
        self,
        arr: Optional["Hessian"],
    ) -> None:
        """Update the Hessian from a cartesian Hessian with shape 3N x 3N for
        N atoms, zeroing the second derivatives if required"""
        return self._update_h_from_cart_h(arr)

    def make_hessian_positive_definite(self) -> None:
        """
        Make the Hessian matrix positive definite by shifting eigenvalues
        """
        self._h = _ensure_positive_definite(self.h, min_eigenvalue=1.0)
        return None

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of these coordinates"""

    @abstractmethod
    def to(self, *args, **kwargs) -> "OptCoordinates":
        """Transformation between these coordinates and another type"""

    @abstractmethod
    def iadd(self, value: np.ndarray) -> "OptCoordinates":
        """Inplace addition of some coordinates"""

    def __eq__(self, other):
        """Coordinates can never be identical..."""
        return False

    def __setitem__(self, key, value):
        """
        Set an item or slice in these coordinates. Clears the current
        gradient and Hessian as well as clearing setting the coordinates.
        Does NOT check if the current value is close to the current, thus
        the gradient and hessian shouldn't be cleared.
        """

        self.clear_tensors()
        return super().__setitem__(key, value)

    def __add__(self, other: Union[np.ndarray, float]) -> "OptCoordinates":
        """
        Addition of another set of coordinates. Clears the current
        gradient vector and Hessian matrix.

        -----------------------------------------------------------------------
        Arguments:
            other (np.ndarray): Array to add to the coordinates

        Returns:
            (autode.opt.coordinates.OptCoordinates): Shifted coordinates
        """
        new_coords = self.copy()
        new_coords.clear_tensors()
        new_coords.iadd(other)

        return new_coords

    def __sub__(self, other: Union[np.ndarray, float]) -> "OptCoordinates":
        """Subtraction"""
        return self.__add__(-other)

    def __iadd__(self, other: Union[np.ndarray, float]) -> "OptCoordinates":
        """Inplace addition"""
        self.clear_tensors()
        return self.__add__(other)

    def __isub__(self, other: Union[np.ndarray, float]) -> "OptCoordinates":
        """Inplace subtraction"""
        return self.__iadd__(-other)

    def clear_tensors(self) -> None:
        """
        Helper function for clearing the energy, gradient and Hessian for these
        coordinates. Called if the coordinates have been perturbed, making
        these quantities not accurate any more for the new coordinates
        """
        self._e, self._g, self._h = None, None, None
        return None

    def copy(self, *args, **kwargs) -> "OptCoordinates":
        return deepcopy(self)


def _ensure_positive_definite(
    matrix: np.ndarray, min_eigenvalue: float = 1e-10
) -> np.ndarray:
    """
    Ensure that the eigenvalues of a matrix are all >0 i.e. the matrix
    is positive definite. Will shift all values below min_eigenvalue to that
    value.

    ---------------------------------------------------------------------------
    Arguments:
        matrix: Matrix to make positive definite

        min_eigenvalue: Minimum value eigenvalue of the matrix

    Returns:
        (np.ndarray): Matrix with eigenvalues at least min_eigenvalue
    """

    if matrix is None:
        raise RuntimeError(
            "Cannot make a positive definite matrix - " "had no matrix"
        )

    lmd, v = np.linalg.eig(matrix)  # Eigenvalues and eigenvectors

    if np.all(lmd > min_eigenvalue):
        logger.info("Matrix was positive definite")
        return matrix

    logger.warning(
        "Matrix was not positive definite. "
        "Shifting eigenvalues to X and reconstructing"
    )
    lmd[lmd < min_eigenvalue] = min_eigenvalue
    return np.linalg.multi_dot((v, np.diag(lmd), v.T)).real
