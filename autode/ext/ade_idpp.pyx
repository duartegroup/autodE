# distutils: language = c++
# distutils: sources = [autode/ext/src/idpp.cpp, autode/ext/src/utils.cpp]

import numpy as np
from autode.log import logger
import logging
from autode.ext.wrappers cimport IdppParams, calculate_idpp_path, get_path_length, relax_path


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
    calc_params.k_spr = kwargs.get('k_spr')
    calc_params.sequential = kwargs.get('sequential')
    calc_params.rmsgtol = kwargs.get('rms_gtol')
    calc_params.maxiter = kwargs.get('maxiter')
    calc_params.add_img_maxgtol = kwargs.get('add_img_maxgtol')
    calc_params.add_img_maxiter = kwargs.get('add_img_maxiter')

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

def get_relaxed_path(
    all_coords,
    n_images: int,
    **kwargs,
):
    """
    Relax a set of images using the IDPP method
    Args:
        all_coords: Numpy array of all coordinates, including the
                    reactant and product
        n_images: Number of images supplied
    Keyword Args:
        sequential (bool): Whether to use the sequential IDPP
        k_spr (float): The spring constant value
        rms_gtol (float): The RMS gradient tolerance for the path
        maxiter (int): Maximum number of iters for path
        add_img_maxgtol (float): Max. gradient tolerance for adding
                        new images (only for sequential)
        add_img_maxiter (int): Max. number of iters for adding new
                        images (only for sequential)
    """
    all_coords_cont = np.ascontiguousarray(
        all_coords.ravel(), dtype=np.double
    )
    cdef double [:] all_coords_view = all_coords_cont
    assert all_coords_cont.shape[0] % n_images == 0
    coords_len = all_coords_cont.shape[0] // n_images
    assert coords_len % 3 == 0

    cdef IdppParams params = handle_kwargs(kwargs)

    relax_path(
        &all_coords_view[0],
        int(coords_len),
        n_images,
        params,
    )

    return all_coords_cont
