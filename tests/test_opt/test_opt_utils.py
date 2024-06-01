from autode import Molecule
from autode.utils import work_in_tmp_dir
from autode.opt.coordinates import CartesianCoordinates
from autode.bracket.imagepair import _calculate_engrad_for_species
from autode.methods import XTB
from autode.opt.optimisers.utils import TruncatedTaylor, Polynomial2PointFit
from scipy.optimize import minimize
import numpy as np
from numpy.polynomial import Polynomial
from ..testutils import requires_working_xtb_install


@requires_working_xtb_install
@work_in_tmp_dir()
def test_truncated_taylor_surface():
    mol = Molecule(smiles="CCO")
    mol.calc_hessian(method=XTB())
    coords = CartesianCoordinates(mol.coordinates)
    coords.update_g_from_cart_g(mol.gradient)
    coords.update_h_from_cart_h(mol.hessian)
    coords.make_hessian_positive_definite()

    # for positive definite hessian, minimum of taylor surface would
    # be a simple Newton step
    minim = coords - (np.linalg.inv(coords.h) @ coords.g)

    # minimizing surface should give the same result
    surface = TruncatedTaylor(coords, coords.g, coords.h)
    res = minimize(
        method="CG",
        fun=surface.value,
        x0=np.array(coords),
        jac=surface.gradient,
    )

    assert res.success
    assert np.allclose(res.x, minim, rtol=1e-4)


@requires_working_xtb_install
@work_in_tmp_dir()
def test_cubic_interpolation():
    mol = Molecule(smiles="CCO")

    en1, grad1 = _calculate_engrad_for_species(mol, XTB(), 1)
    coords1 = CartesianCoordinates(mol.coordinates)
    coords1.e = en1
    coords1.update_g_from_cart_g(grad1)
    # take a steepest descent step
    step = -coords1.g * 0.1 / np.linalg.norm(coords1.g)
    coords2 = coords1 + step
    mol.coordinates = coords2
    en2, grad2 = _calculate_engrad_for_species(mol, XTB(), 1)
    coords2.e = en2
    coords2.update_g_from_cart_g(grad2)
    # cubic interp
    cubic_poly = Polynomial2PointFit.cubic_fit(coords1, coords2)
    # energy at a point between 0 and 1
    coords3 = coords1 + 0.25 * step
    mol.coordinates = coords3
    en3, _ = _calculate_engrad_for_species(mol, XTB(), 1)
    interp_en = cubic_poly(0.25)
    assert np.isclose(interp_en, en3, atol=1e-3)


def test_polynomial_max_min():
    # 1 - x + 2x^2 + x^3
    poly = Polynomial([1, -1, 2, 1])
    # known minimum = 0.215
    x_min = Polynomial2PointFit.get_extremum(poly, 0, 1)
    assert np.isclose(x_min, 0.215, atol=1e-3)
    # known maximum = -1.549
    x_max = Polynomial2PointFit.get_extremum(poly, -2, 0, get_max=True)
    assert np.isclose(x_max, -1.549, atol=1e-3)
    # order of bounds should not matter
    x_max = Polynomial2PointFit.get_extremum(poly, 0, -2, get_max=True)
    assert np.isclose(x_max, -1.549, atol=1e-3)
    # should not count inflection points
    poly = Polynomial([0, 0, 0, 1])
    x_min = Polynomial2PointFit.get_extremum(poly, -np.inf, np.inf)
    x_max = Polynomial2PointFit.get_extremum(
        poly, -np.inf, np.inf, get_max=True
    )
    assert x_min is None and x_max is None
