import numpy as np

from typing import TYPE_CHECKING
from autode.log import logger
import logging
from autode.values import PotentialEnergy
from ade_idpp import get_interpolated_path, get_interp_path_length

if TYPE_CHECKING:
    from autode.neb.original import Image, Images


class IDPP:
    """
    Image dependent pair potential (IDPP) objective function from
    https://arxiv.org/pdf/1406.1512.pdf

    .. math::

        S = Σ_i  Σ_{j>i} w(r_{ij}) (r_{ij}^{(k)} - r_{ij})^2

    where :math:`r_{ij}` is the distance between atoms i and j and
    :math:`r_{ij}^{(k)} = r_{ij}^{(1)} + k(r_{ij}^{(N)} - r_{ij}^{(1)})/N` for
    :math:`N` images. The weight function is :math:`w(r_{ij}) = r_{ij}^{-4}`,
    as suggested in the paper. Computation is done in C++ extension.
    """

    def __init__(
        self,
        n_images: int,
        k_spr: float = 1.0,
        sequential: bool = True,
        rms_gtol: float = 2e-3,
        maxiter: int = 1000,
        add_img_maxgtol: float = 6e-3,
        add_img_maxiter: int = 30,
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
            add_img_maxiter: Maximum number of iterations for adding images
        """
        assert n_images > 2, "Must have more than 2 images"
        self._n_images = int(n_images)
        assert k_spr > 0, "Spring constant must be positive"
        self._k_spr = float(k_spr)
        self._sequential = bool(sequential)
        assert rms_gtol > 0, "RMS gradient tolerance must be positive"
        self._rms_gtol = float(rms_gtol)
        assert maxiter > 0, "Maximum iterations must be positive"
        self._maxiter = int(maxiter)
        assert (
            add_img_maxgtol > 0
        ), "Add image gradient tolerance must be positive"
        self._add_img_maxgtol = float(add_img_maxgtol)
        assert (
            add_img_maxiter > 0
        ), "Add image maximum iterations must be positive"
        self._add_img_maxiter = int(add_img_maxiter)
        # TODO: make default arguments simpler and allow None

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
