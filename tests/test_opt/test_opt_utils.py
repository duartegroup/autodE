from autode import Molecule
from autode.utils import work_in_tmp_dir
from autode.opt.coordinates import CartesianCoordinates
from autode.bracket.imagepair import _calculate_engrad_for_species
from autode.methods import XTB
from autode.opt.optimisers.utils import TruncatedTaylor, two_point_cubic_fit
from scipy.optimize import minimize
import numpy as np
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
    g1 = float(np.dot(coords1.g, step))
    g2 = float(np.dot(coords2.g, step))
    cubic_poly = two_point_cubic_fit(e0=en1, g0=g1, e1=en2, g1=g2)
    # energy at halfway point
    coords3 = coords1 + 0.5 * step
    mol.coordinates = coords3
    en3, _ = _calculate_engrad_for_species(mol, XTB(), 1)
    interp_en = cubic_poly(0.5)
    assert np.isclose(interp_en, en3, atol=1e-3)
