import os
import pytest
import numpy as np
from ..testutils import work_in_zipped_dir
from autode.opt.optimisers.hessian_update import (
    BFGSUpdate,
    BFGSPDUpdate,
    SR1Update,
    NullUpdate,
    BofillUpdate,
)

here = os.path.dirname(os.path.abspath(__file__))


def update_improves_hessian(updater, guess, true):

    return updater.conditions_met and (
        np.linalg.norm(updater.updated_h - true) < np.linalg.norm(guess - true)
    )


def update_improves_inv_hessian(updater, guess, true):

    return updater.conditions_met and (
        np.linalg.norm(updater.updated_h_inv - true)
        < np.linalg.norm(guess - true)
    )


def check_hessian_update(update_type, grad, h_true):

    for pair in [(0.3, 0.3), (0.5, 0.5)]:

        x = np.array(pair)

        h = h_true(*x)
        h_inv = np.linalg.inv(h)

        h_guess = np.eye(2)  # I_2 = I_2^(-1)
        h_inv_guess = np.eye(2)

        for increment in (0.1, 0.2, 0.3):

            x_new = x - increment
            updater = update_type(
                s=x_new - x,
                y=grad(*x_new) - grad(*x),
                h=h_guess,
                h_inv=h_inv_guess,
            )

            assert update_improves_hessian(updater, guess=h_guess, true=h)
            assert update_improves_inv_hessian(
                updater, guess=h_inv_guess, true=h_inv
            )


def test_quadratic_hessian_update():
    """E = x^2 + y^2"""

    for update_type in (BFGSUpdate, SR1Update):
        check_hessian_update(
            update_type,
            grad=lambda x, y: 2.0 * np.array([x, y]),
            h_true=lambda x, y: 2.0 * np.eye(2),
        )


def test_quadratic2_hessian_update():
    """E = x^2 + y^2 + xy/2"""

    for update_type in (BFGSUpdate, SR1Update):
        check_hessian_update(
            update_type,
            grad=lambda x, y: np.array([2 * x + y / 2, 2 * y + x / 2]),
            h_true=lambda x, y: np.array([[2.0, 0.5], [0.5, 2.0]]),
        )


def test_polynomial_hessian_update():
    """E = x^5 + y^3 + xy^2/2"""

    # Hessian for more complex 2D surface fails to improve with an SR1 update
    check_hessian_update(
        BFGSUpdate,
        grad=lambda x, y: np.array(
            [5 * x**4 + y**2 / 2, 3 * y**2 + x * y]
        ),
        h_true=lambda x, y: np.array([[20 * x**3, y], [y, 6 * y + x]]),
    )


def test_null_update():

    updater = NullUpdate(h=10.0 * np.eye(4))

    # Null update has no conditions on update
    assert updater.conditions_met

    # and just returns the hessian back
    assert np.allclose(updater.updated_h, 10.0 * np.eye(4))


def test_updater_class():

    updater = NullUpdate()

    # Cannot update without either an initial H or H^(-1)
    with pytest.raises(RuntimeError):
        _ = updater.updated_h_inv

    with pytest.raises(RuntimeError):
        _ = updater.updated_h


def test_bofill_update():
    """Notation follows https://aip.scitation.org/doi/pdf/10.1063/1.1515483"""

    G_prev = np.array([[1.0, 0.1], [0.1, 1.0]])  # initial approximate Hessian

    dx = np.array([0.01, 0.03])
    dg = np.array([-0.02, -0.06])

    updater = BofillUpdate(h=G_prev, s=dx, y=dg)

    dg_Gdx = dg - np.dot(G_prev, dx)

    G_MS = G_prev + (np.outer(dg_Gdx, dg_Gdx) / np.dot(dg_Gdx, dx))

    G_PSB = (
        G_prev
        + ((np.outer(dg_Gdx, dx) + np.outer(dx, dg_Gdx)) / np.dot(dx, dx))
        - (
            (np.dot(dx, dg) - np.dot(dx, np.dot(G_prev, dx)))
            * np.outer(dx, dx)
            / np.dot(dx, dx) ** 2
        )
    )

    phi = 1.0 - (
        np.dot(dx, dg_Gdx) ** 2 / (np.dot(dx, dx) * np.dot(dg_Gdx, dg_Gdx))
    )

    G_bofill = (1.0 - phi) * G_MS + phi * G_PSB

    assert np.allclose(updater.updated_h, G_bofill, atol=1e-10)


def test_repr_and_strings():
    """
    Test that the hessian update types have defined representation
    and strings
    """

    for update_type in (
        BFGSUpdate,
        BFGSPDUpdate,
        SR1Update,
        NullUpdate,
        BofillUpdate,
    ):

        assert update_type().__repr__() is not None
        assert str(update_type()) is not None


class BFGSPDUpdateNoUpdate(BFGSPDUpdate):
    @property
    def _updated_h(self) -> np.ndarray:
        return self.h

    @property
    def _updated_h_inv(self) -> np.ndarray:
        return self.h_inv


@pytest.mark.parametrize(("eigval", "expected"), ([-1.0, False], [1.0, True]))
def test_bfgs_pd_update(eigval, expected):

    h = np.array([[1.0, 0.0], [0.0, eigval]])

    updater = BFGSPDUpdateNoUpdate(
        min_eigenvalue=0.1, h=h, s=np.array([0.1, 0.1]), y=np.array([0.1, 0.1])
    )

    # Needs to satisfy the secant equation
    assert updater.y.dot(updater.s) > 0
    assert updater.conditions_met == expected


@work_in_zipped_dir(os.path.join(here, "data", "hessians.zip"))
def test_bfgs_update_water():

    h = np.loadtxt("water_cart_hessian0.txt")

    x0 = [-0.0011, 0.3631, -0.0, -0.825, -0.1819, -0.0, 0.8261, -0.1812, 0.0]
    g0 = [
        -0.0026315238924962724,
        0.02788091552485463,
        3.5360899212838047e-16,
        -0.0498982397982622,
        -0.013121966502035046,
        -3.3717488622744373e-16,
        0.05252976369075942,
        -0.014758949022819396,
        -1.6434105900929746e-17,
    ]

    x1 = [
        -8.629582978216805e-05,
        0.37325030920738983,
        0.0,
        -0.7760070560089669,
        -0.1873002044203907,
        0.0,
        0.7760933518387491,
        -0.1859501047869992,
        0.0,
    ]
    g1 = [
        0.0008070291147249597,
        -0.007905970009730014,
        -1.5862959782497162e-16,
        0.0004597293468636392,
        0.003663285355006926,
        1.8209233374357918e-16,
        -0.001266758461588599,
        0.004242684654723089,
        -2.3462735918607574e-17,
    ]

    updater = BFGSUpdate(
        h=h, s=np.array(x1) - np.array(x0), y=np.array(g1) - np.array(g0)
    )

    new_bfgs_h = updater.updated_h
    new_true_h = np.loadtxt("water_cart_hessian1.txt")

    assert np.sum(np.abs(new_bfgs_h - new_true_h)) < np.sum(
        np.abs(h - new_true_h)
    )


def test_hessian_requires_at_least_one_index():

    h = np.eye(3)
    s = y = np.zeros(3)
    with pytest.raises(ValueError):
        updater = BFGSUpdate(h=h, s=s, y=y, subspace_idxs=[])
