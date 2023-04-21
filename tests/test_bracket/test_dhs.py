import os
import numpy as np
from scipy.optimize import minimize
from autode import Molecule
from autode.methods import XTB
from autode.utils import work_in, work_in_tmp_dir
from autode.geom import calc_rmsd
from autode.bracket.dhs import (
    DHS,
    DistanceConstrainedOptimiser,
    TruncatedTaylor,
)
from autode.opt.coordinates import CartesianCoordinates
from autode import Config
from ..testutils import requires_with_working_xtb_install

here = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(here, "data")
# todo replace with zip later


@requires_with_working_xtb_install
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
        fun=surface.value,
        x0=np.array(coords),
        jac=surface.gradient,
    )

    assert res.success
    assert np.allclose(res.x, minim, rtol=1e-4)


@requires_with_working_xtb_install
@work_in(datadir)
def test_distance_constrained_optimiser():
    reactant = Molecule("da_reactant.xyz")
    product = Molecule("da_product.xyz")
    rct_coords = CartesianCoordinates(reactant.coordinates)
    prod_coords = CartesianCoordinates(product.coordinates)

    # displace product coordinates towards reactant
    dist_vec = prod_coords - rct_coords
    prod_coords = prod_coords - 0.1 * dist_vec
    distance = np.linalg.norm(prod_coords - rct_coords)
    product.coordinates = prod_coords

    opt = DistanceConstrainedOptimiser(
        pivot_point=rct_coords,
        maxiter=1,
        init_trust=0.2,
        gtol=1e-3,
        etol=1e-4,
    )
    opt.run(product, method=XTB())
    assert not opt.converged
    prod_coords_new = opt.final_coordinates

    # distance should not change
    new_distance = np.linalg.norm(prod_coords_new - rct_coords)
    assert np.isclose(new_distance, distance)
    # step should be less than trust radius
    step_size = np.linalg.norm(prod_coords_new - prod_coords)
    assert np.isclose(step_size, 0.2)

    opt = DistanceConstrainedOptimiser(
        pivot_point=rct_coords,
        maxiter=50,
        gtol=1e-3,
        etol=1e-4,
    )
    opt.run(product, method=XTB())
    assert opt.converged
    prod_coords_new = opt.final_coordinates
    new_distance = np.linalg.norm(prod_coords_new - rct_coords)
    assert np.isclose(new_distance, distance)


@requires_with_working_xtb_install
@work_in(datadir)
def test_dhs_single_step():
    step_size = 0.2
    reactant = Molecule("da_reactant.xyz")
    product = Molecule("da_product.xyz")

    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=2000,
        step_size=step_size,
        dist_tol=1.0,
    )

    dhs.imgpair.set_method_and_n_cores(engrad_method=XTB(), n_cores=1)
    dhs._method = XTB()
    dhs._initialise_run()

    imgpair = dhs.imgpair
    assert imgpair.left_coord.e is not None
    assert imgpair.right_coord.e is not None
    old_dist = imgpair.dist
    if imgpair.left_coord.e < imgpair.right_coord.e:
        lower_img = "left"
    else:
        lower_img = "right"
    # take a single step
    dhs._step()
    # step should be on lower energy image
    if lower_img == "left":
        assert len(imgpair._right_history) == 1
        assert len(imgpair._left_history) == 2
    else:
        assert len(imgpair._left_history) == 1
        assert len(imgpair._right_history) == 2
    new_dist = imgpair.dist
    # image should move exactly by step_size
    assert new_dist - old_dist == step_size


@requires_with_working_xtb_install
@work_in(datadir)
def test_dhs_diels_alder():
    set_dist_tol = 1.0  # angstrom

    reactant = Molecule("da_reactant.xyz")
    product = Molecule("da_product.xyz")
    # TS optimized with ORCA using xTB method
    true_ts = Molecule("da_ts_orca_xtb.xyz")

    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=2000,
        step_size=0.2,
        dist_tol=set_dist_tol,
    )

    dhs.calculate(method=XTB(), n_cores=Config.n_cores)
    assert dhs.converged
    peak = dhs.ts_guess

    rmsd = calc_rmsd(peak.coordinates, true_ts.coordinates)
    # Euclidean distance = rmsd * sqrt(n_atoms)
    distance = rmsd * np.sqrt(peak.n_atoms)

    # the true TS must be within the last two DHS images, therefore
    # the distance must be less than the distance tolerance
    # (assuming curvature of PES near TS not being too high)

    assert distance < set_dist_tol
