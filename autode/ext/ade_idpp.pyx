# distutils: language = c++
# distutils: sources = [autode/ext/src/idpp.cpp, autode/ext/src/utils.cpp]

import numpy as np
from autode.log import logger
import logging
from autode.ext.wrappers cimport calculate_idpp_path

class IDPP:
    def __init__(
        self,
        n_images: int,
        k_spr: float = 0.1,
        sequential: bool = True,
        rms_gtol = 5e-4,
        maxiter = 2000,
    ):
        """
        Initialise an IDPP calculation

        Args:
            n_images: Number of images requested
            k_spr: The spring constant
            sequential: Whether to use sequential IDPP
            rms_gtol: RMS gradient tolerance for path
            maxiter: Maximum number of LBFGS iterations
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

    def get_path(
        self,
        init_coords: np.ndarray,
        final_coords: np.ndarray
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
        # access raw memory block of the arrays
        init_coords = np.ascontiguousarray(init_coords.flatten(), dtype=float)
        final_coords = np.ascontiguousarray(final_coords.flatten(), dtype=float)

        cdef double [:] init_view = init_coords
        cdef double [:] final_view = final_coords

        if init_coords.shape != final_coords.shape:
            raise ValueError(
                "Initial and final coordinates must have the same shape"
            )
        coords_len = init_coords.shape[0]
        assert coords_len % 3 == 0, "Coordinate array has incorrect dimensions"
        img_coordinates = np.zeros(
            shape=(self._n_images - 2) * coords_len, order="C", dtype=float
        )
        cdef double [:] img_coords_view = img_coordinates

        debug_print = logger.isEnabledFor(logging.DEBUG)

        calculate_idpp_path(
            &init_view[0],
            &final_view[0],
            int(coords_len),
            self._n_images,
            self._k_spr,
            self._sequential,
            &img_coords_view[0],
            bool(debug_print),
            self._rms_gtol,
            self._maxiter,
        )
        return img_coordinates