# distutils: language = c++
# distutils: sources = [autode/ext/src/idpp.cpp, autode/ext/src/utils.cpp]

import numpy as np
from autode.log import logger
import logging
from autode.ext.wrappers cimport calculate_idpp_path, IdppParams, get_path_length


cdef IdppParams handle_kwargs(kwargs) except *:
    """
    Obtain an IdppParams object from keyword arguments. Allowed keys in
    the kwargs dictionary are
        sequential (bool): Whether to use the sequential IDPP

        k_spr (float): The spring constant value

        rms_gtol (float): The RMS gradient tolerance for the path

        maxiter (int): Maximum number of iters for path

        add_img_maxgtol (float): Max. gradient tolerance for adding
                         new images (only for sequential)

        add_img_maxiter (int): Max. number of iters for adding new
                         images (only for sequential)

    Args:
        kwargs (dict): Dictionary of keyword arguments.

    Returns:
        (IdppParams):
    """
    cdef IdppParams calc_params
    calc_params.k_spr = kwargs.get('k_spr', 1.0)
    calc_params.sequential = kwargs.get('sequential', True)
    calc_params.rmsgtol = kwargs.get('rms_gtol', 2e-3)
    calc_params.maxiter = kwargs.get('maxiter', 1000)
    calc_params.add_img_maxgtol = kwargs.get('add_img_maxgtol', 0.005)
    calc_params.add_img_maxiter = kwargs.get('add_img_maxiter', 30)

    # change the debug option according to current logging level
    debug_print = logger.isEnabledFor(logging.DEBUG)
    calc_params.debug = debug_print

    return calc_params



def get_interpolated_path(
    init_coords,
    final_coords,
    n_images: int,
    **kwargs,
):
    """
    Obtain the interpolated path (using the IDPP method) from
    the coordinates of the reactant and product states

    Args:
        init_coords (np.ndarray): Initial coordinates
        final_coords (np.ndarray): Final coordinates
        n_images (int): Number of images requested

    Keyword Args:
        sequential (bool): Whether to use the sequential IDPP
        k_spr (float): The spring constant value
        rms_gtol (float): The RMS gradient tolerance for the path
        maxiter (int): Maximum number of iters for path
        add_img_maxgtol (float): Max. gradient tolerance for adding
                        new images (only for sequential)
        add_img_maxiter (int): Max. number of iters for adding new
                        images (only for sequential)

    Returns:
        (np.ndarray): Numpy array of coordinates of the
                    intermediate images in the path
    """
    if init_coords.shape != final_coords.shape:
        raise ValueError("Coordinates must have the same shape")

    # access raw memory block of the arrays
    init_coords = np.ascontiguousarray(
        init_coords.ravel(), dtype=np.double
    )
    final_coords = np.ascontiguousarray(
        final_coords.ravel(), dtype=np.double
    )
    cdef double [:] init_view = init_coords
    cdef double [:] final_view = final_coords

    coords_len = init_coords.shape[0]
    interm_img_coordinates = np.zeros(
        shape=((n_images - 2) * coords_len,),
        order="C",
        dtype=np.double
    )
    cdef double [:] img_coords_view = interm_img_coordinates
    cdef IdppParams params = handle_kwargs(kwargs)

    calculate_idpp_path(
        &init_view[0],
        &final_view[0],
        int(coords_len),
        n_images,
        &img_coords_view[0],
        params,
    )
    return interm_img_coordinates

def get_interp_path_length(
    init_coords,
    final_coords,
    n_images: int,
    **kwargs,
):
    """
    Obtain the length of the interpolated path (using the IDPP method) from
    the coordinates of the reactant and product states

    Args:
        init_coords (np.ndarray): Initial coordinates
        final_coords (np.ndarray): Final coordinates
        n_images (int): Number of images requested

    Keyword Args:
        sequential (bool): Whether to use the sequential IDPP
        k_spr (float): The spring constant value
        rms_gtol (float): The RMS gradient tolerance for the path
        maxiter (int): Maximum number of iters for path
        add_img_maxgtol (float): Max. gradient tolerance for adding
                        new images (only for sequential)
        add_img_maxiter (int): Max. number of iters for adding new
                        images (only for sequential)

    Returns:
        (float): Length of the interpolated path
    """
    if init_coords.shape != final_coords.shape:
        raise ValueError("Coordinates must have the same shape")

    # access raw memory block of the arrays
    init_coords = np.ascontiguousarray(
        init_coords.ravel(), dtype=np.double
    )
    final_coords = np.ascontiguousarray(
        final_coords.ravel(), dtype=np.double
    )
    cdef double [:] init_view = init_coords
    cdef double [:] final_view = final_coords
    coords_len = init_coords.shape[0]

    cdef IdppParams params = handle_kwargs(kwargs)

    return get_path_length(
        &init_view[0],
        &final_view[0],
        int(coords_len),
        n_images,
        params,
    )


class IDPP:
    def __init__(
        self,
        n_images: int,
        sequential: bool = True,
        k_spr: float = 1.0,
        rms_gtol: float = 2e-3,
        maxiter: int = 1000,
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
        init_coords = np.ascontiguousarray(
            init_coords.ravel(), dtype=np.double
        )
        final_coords = np.ascontiguousarray(
            final_coords.ravel(), dtype=np.double
        )

        cdef double [:] init_view = init_coords
        cdef double [:] final_view = final_coords

        if init_coords.shape != final_coords.shape:
            raise ValueError(
                "Initial and final coordinates must have the same shape"
            )
        coords_len = init_coords.shape[0]
        assert coords_len % 3 == 0, "Coordinate array has incorrect dimensions"
        img_coordinates = np.zeros(
            shape=((self._n_images - 2) * coords_len,),
            order="C",
            dtype=np.double
        )
        cdef double [:] img_coords_view = img_coordinates

        debug_print = logger.isEnabledFor(logging.DEBUG)

        cdef IdppParams params
        params.k_spr = self._k_spr
        params.sequential = self._sequential
        params.debug = debug_print
        params.maxiter = self._maxiter
        params.rmsgtol = self._rms_gtol
        # TODO: make the parameters configurable
        params.add_img_maxiter = 40
        params.add_img_maxgtol = 0.005

        calculate_idpp_path(
            &init_view[0],
            &final_view[0],
            int(coords_len),
            self._n_images,
            &img_coords_view[0],
            params,
        )
        return img_coordinates