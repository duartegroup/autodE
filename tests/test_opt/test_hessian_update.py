import numpy as np
from autode.opt.optimisers.hessian_update import (BFGSUpdate,
                                                  SR1Update,
                                                  NullUpdate)


def update_improves_hessian(updater, guess, true):

    return (updater.conditions_met
            and (np.linalg.norm(updater.updated_h - true)
                 < np.linalg.norm(guess - true)))


def update_improves_inv_hessian(updater, guess, true):

    return (updater.conditions_met
            and (np.linalg.norm(updater.updated_h_inv - true)
                 < np.linalg.norm(guess - true)))


def check_hessian_update(update_type, grad, h_true):

    for pair in [(0.3, 0.3), (0.5, 0.5)]:

        x = np.array(pair)

        h = h_true(*x)
        h_inv = np.linalg.inv(h)

        h_guess = np.eye(2)  # I_2 = I_2^(-1)
        h_inv_guess = np.eye(2)

        for increment in (0.1, 0.2, 0.3):

            x_new = x - increment
            updater = update_type(s=x_new - x,
                                  y=grad(*x_new) - grad(*x),
                                  h=h_guess,
                                  h_inv=h_inv_guess)

            assert update_improves_hessian(updater,
                                           guess=h_guess, true=h)
            assert update_improves_inv_hessian(updater,
                                               guess=h_inv_guess,
                                               true=h_inv)


def test_quadratic_hessian_update():
    """E = x^2 + y^2"""

    for update_type in (BFGSUpdate, SR1Update):
        check_hessian_update(update_type,
                             grad=lambda x, y: 2.0*np.array([x, y]),
                             h_true=lambda x, y: 2.0 * np.eye(2))


def test_quadratic2_hessian_update():
    """E = x^2 + y^2 + xy/2"""

    for update_type in (BFGSUpdate, SR1Update):
        check_hessian_update(update_type,
                             grad=lambda x, y: np.array([2*x+y/2, 2*y + x/2]),
                             h_true=lambda x, y: np.array([[2.0, 0.5],
                                                           [0.5, 2.0]]))


def test_polynomial_hessian_update():
    """E = x^5 + y^3 + xy^2/2"""

    # Hessian for more complex 2D surface fails to improve with an SR1 update
    check_hessian_update(BFGSUpdate,
                         grad=lambda x, y: np.array([5*x**4 + y**2/2,
                                                     3*y**2 + x*y]),
                         h_true=lambda x, y: np.array([[20 * x ** 3, y],
                                                       [y, 6*y + x]])
                         )


def test_null_update():
    
    updater = NullUpdate(h=10.*np.eye(4))
    
    # Null update has no conditions on update
    assert updater.conditions_met
    
    # and just returns the hessian back
    assert np.allclose(updater.updated_h,
                       10.*np.eye(4))
