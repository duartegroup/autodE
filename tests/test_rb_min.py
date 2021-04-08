import numpy as np
from c_rb import minimised_rb_coords
from autode.geom import are_coords_reasonable


def _test_construction():

    coords = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.1]], dtype='f8')

    bonded= np.array([[False, True],
                      [True, False]], dtype=bool)

    r0 = np.array([[0.0, 1.0],
                   [1.0, 0.0]], dtype='f8')

    k = np.array([[0.0, 1.0],
                  [1.0, 0.0]], dtype='f8')

    c = np.array([[0.0, 0.1],
                  [0.1, 0.0]], dtype='f8')

    coords = minimised_rb_coords(py_coords=coords,
                                 py_bonded_matrix=bonded,
                                 py_r0_matrix=r0,
                                 py_k_matrix=k,
                                 py_c_matrix=c,
                                 py_exponent=4)

    # assert are_coords_reasonable(coords)
