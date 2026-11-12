import numpy as np

from typing import List
from autode.ext.ade_idpp import (
    get_interpolated_path,
    get_interp_path_length,
    get_relaxed_path,
)


class IDPP:
    """
    Image dependent pair potential (IDPP) objective function from
    https://arxiv.org/pdf/1406.1512.pdf

    .. math::

        S = Σ_i  Σ_{j>i} w(r_{ij}) (r_{ij}^{(k)} - r_{ij})^2

    where :math:`r_{ij}` is the distance between atoms i and j and
    :math:`r_{ij}^{(k)} = r_{ij}^{(1)} + k(r_{ij}^{(N)} - r_{ij}^{(1)})/N` for
    :math:`N` images. The weight function is :math:`w(r_{ij}) = r_{ij}^{-4}`,
    as suggested in the paper. Computation is done in C++ extension
    """

    def __init__(
        self,
        n_images: int,
        k_spr: float = 1.0,
        sequential: bool = True,
        rms_gtol: float = 1e-3,
        maxiter: int = 1000,
        add_img_maxgtol: float = 2e-3,
        add_img_maxiter: int = 200,
    ):
        """
        Initialise an IDPP calculation

        Args:
            n_images: Number of images requested
            k_spr: The spring constant
            sequential: Whether to use sequential IDPP
            rms_gtol: RMS gradient tolerance for path
            maxiter: Maximum number of iterations for path
            add_img_maxgtol: Maximum gradient tolerance for adding images
            add_img_maxiter: Maximum number of iterations for adding image
        """
        self._n_images = int(n_images)
        assert self._n_images > 2
        self._k_spr = float(k_spr)
        self._sequential = bool(sequential)
        self._rms_gtol = float(rms_gtol)
        self._maxiter = int(maxiter)
        self._add_img_maxgtol = float(add_img_maxgtol)
        self._add_img_maxiter = int(add_img_maxiter)

    def get_path(
        self, init_coords: np.ndarray, final_coords: np.ndarray
    ) -> np.ndarray:
        """
        Get the IDPP path between the intial and final coordinates

        Args:
            init_coords: Numpy array of initial coordinates
            final_coords: Numpy array of final coordinates

        Returns:
            (np.ndarray): Numpy array of coordinates of the
                        intermediate images in the path
        """
        if init_coords.shape != final_coords.shape:
            raise ValueError(
                "Initial and final coordinates must have the same shape"
            )

        return get_interpolated_path(
            init_coords,
            final_coords,
            self._n_images,
            sequential=self._sequential,
            k_spr=self._k_spr,
            rms_gtol=self._rms_gtol,
            maxiter=self._maxiter,
            add_img_maxgtol=self._add_img_maxgtol,
            add_img_maxiter=self._add_img_maxiter,
        )

    def get_path_length(
        self, init_coords: np.ndarray, final_coords: np.ndarray
    ) -> float:
        """
        Get the length of the IDPP path between the intial and final coordinates

        Args:
            init_coords: Numpy array of initial coordinates
            final_coords: Numpy array of final coordinates

        Returns:
            (float): Length of the IDPP path
        """
        return get_interp_path_length(
            init_coords,
            final_coords,
            self._n_images,
            sequential=self._sequential,
            k_spr=self._k_spr,
            rms_gtol=self._rms_gtol,
            maxiter=self._maxiter,
            add_img_maxgtol=self._add_img_maxgtol,
            add_img_maxiter=self._add_img_maxiter,
        )

    def relax_path(self, coords_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Relax the set of coordinates using the IDPP method

        Args:
            coords_list: List of coordinates (numpy arrays)

        Returns:
            (List[np.ndarray]): List of relaxed coordinates
        """
        n_images = len(coords_list)
        assert all(isinstance(c, np.ndarray) for c in coords_list)
        coords_len = len(coords_list[0].ravel())
        assert all(len(c.ravel()) == coords_len for c in coords_list)

        all_coords = np.array(coords_list).ravel()
        all_coords = get_relaxed_path(
            all_coords,
            n_images,
            sequential=self._sequential,
            k_spr=self._k_spr,
            rms_gtol=self._rms_gtol,
            maxiter=self._maxiter,
            add_img_maxgtol=self._add_img_maxgtol,
            add_img_maxiter=self._add_img_maxiter,
        )
        for i in range(n_images):
            coords_list[i] = all_coords[i * coords_len : (i + 1) * coords_len]
        return coords_list
