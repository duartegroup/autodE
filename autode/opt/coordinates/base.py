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
    from typing import Type
    from autode.opt.optimisers.hessian_update import HessianUpdater


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

    def update_h_from_old_h(
        self,
        old_coords: "OptCoordinates",
        hessian_update_types: List["Type[HessianUpdater]"],
    ) -> None:
        r"""
        Update the Hessian :math:`H` from an old Hessian using an update
        scheme. Requires the gradient to be set, and the old set of
        coordinates with gradient to be available

        Args:
            old_coords (OptCoordinates): Old set of coordinates with
                        gradient and hessian defined
            hessian_update_types (list[type[HessianUpdater]]): A list of
                        hessian updater classes - the first updater that
                        meets the mathematical conditions will be used
        """
        assert self._g is not None
        assert isinstance(old_coords, OptCoordinates), "Wrong type!"
        assert old_coords._h is not None
        assert old_coords._g is not None
        idxs = self.active_mol_indexes

        for update_type in hessian_update_types:
            updater = update_type(
                h=old_coords._h,
                s=np.array(self) - np.array(old_coords),
                y=self._g - old_coords._g,
                subspace_idxs=idxs,
            )

            if not updater.conditions_met:
                logger.info(f"Conditions for {update_type} not met")
                continue

            new_h = updater.updated_h
            assert self.h_or_h_inv_has_correct_shape(new_h)
            self._h = new_h
            return None

        raise RuntimeError(
            "Could not update the Hessian - no suitable update strategies"
        )

    @property
    def rfo_shift(self) -> float:
        """
        Get the RFO diagonal shift factor λ for the molecular Hessian that
        can be applied (H - λI) to obtain the RFO downhill step. The shift
        is only calculated in active subspace

        Returns:
            (float): The shift parameter
        """
        assert self._h is not None
        # ignore constraint modes
        n, _ = self._h.shape
        idxs = self.active_mol_indexes
        hess = self._h[:, idxs][idxs, :]
        grad = self._g[idxs]

        h_n, _ = hess.shape
        # form the augmented Hessian in active subspace
        aug_h = np.zeros(shape=(h_n + 1, h_n + 1))

        aug_h[:h_n, :h_n] = hess
        aug_h[-1, :h_n] = grad
        aug_h[:h_n, -1] = grad

        # first non-zero eigenvalue
        aug_h_lmda = np.linalg.eigvalsh(aug_h)
        rfo_lmda = aug_h_lmda[0]
        assert abs(rfo_lmda) > 1.0e-10
        return rfo_lmda

    @property
    def min_eigval(self) -> float:
        """
        Obtain the minimum eigenvalue of the molecular Hessian in
        the active space

        Returns:
            (float): The minimum eigenvalue
        """
        assert self._h is not None
        n, _ = self._h.shape
        idxs = self.active_mol_indexes
        hess = self._h[:, idxs][idxs, :]

        eigvals = np.linalg.eigvalsh(hess)
        assert abs(eigvals[0]) > 1.0e-10
        return eigvals[0]

    def pred_quad_delta_e(self, new_coords: np.ndarray) -> float:
        """
        Calculate the estimated change in energy at the new coordinates
        based on the quadratic model (i.e. second order Taylor expansion)

        Args:
            new_coords(np.ndarray): The new coordinates

        Returns:
            (float): The predicted change in energy
        """
        assert self._g is not None and self._h is not None

        step = np.array(new_coords) - np.array(self)

        idxs = self.active_mol_indexes
        step = step[idxs]
        grad = self._g[idxs]
        hess = self._h[:, idxs][idxs, :]

        pred_delta = np.dot(grad, step)
        pred_delta += 0.5 * np.linalg.multi_dot((step, hess, step))
        return pred_delta

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

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """Number of constraints in these coordinates"""

    @property
    @abstractmethod
    def n_satisfied_constraints(self) -> int:
        """Number of constraints that are satisfied in these coordinates"""

    @property
    @abstractmethod
    def active_indexes(self) -> List[int]:
        """A list of indexes which are active in this coordinate set"""

    @property
    def active_mol_indexes(self) -> List[int]:
        """Active indexes that are actually atomic coordinates in the molecule"""
        return [i for i in self.active_indexes if i < len(self)]

    @property
    @abstractmethod
    def inactive_indexes(self) -> List[int]:
        """A list of indexes which are non-active in this coordinate set"""

    @property
    @abstractmethod
    def cart_proj_g(self) -> Optional[np.ndarray]:
        """
        The Cartesian gradient with any constraints projected out
        """

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
