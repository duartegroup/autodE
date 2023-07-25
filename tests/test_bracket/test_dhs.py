import os
import numpy as np
import pytest
from scipy.optimize import minimize
from autode import Molecule
from autode.methods import XTB
from autode.values import PotentialEnergy
from autode.utils import work_in_tmp_dir
from autode.geom import calc_rmsd
from autode.bracket.dhs import (
    DHS,
    DHSGS,
    DistanceConstrainedOptimiser,
    TruncatedTaylor,
    DHSImagePair,
    ImageSide,
)
from autode.opt.coordinates import CartesianCoordinates
from autode import Config
from ..testutils import requires_working_xtb_install, work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
datazip = os.path.join(here, "data", "geometries.zip")


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
@work_in_zipped_dir(datazip)
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
        maxiter=1,  # just one step
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
    # linear interpolation is skipped on first step
    # should be less than or equal to trust radius
    step_size = np.linalg.norm(prod_coords_new - prod_coords)
    fp_err = 0.000001
    assert step_size <= 0.2 + fp_err * 0.2  # for floating point error

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


def test_dhs_image_pair():
    mol1 = Molecule(smiles="CCO")
    mol2 = mol1.new_species()
    imgpair = DHSImagePair(mol1, mol2)

    coords = imgpair.left_coords + 0.1

    # check the functions that get one side
    with pytest.raises(ValueError):
        imgpair.put_coord_by_side(coords, 2)

    imgpair.left_coords = coords
    with pytest.raises(ValueError):
        step = imgpair.get_last_step_by_side(2)

    # with "left" it should not cause any issues
    step = imgpair.get_last_step_by_side(ImageSide.left)
    assert isinstance(step, np.ndarray)

    with pytest.raises(ValueError):
        imgpair.get_coord_by_side(1)


def test_dhs_image_pair_ts_guess(caplog):
    mol1 = Molecule(smiles="CCO")
    imgpair = DHSImagePair(mol1, mol1.copy())

    imgpair.left_coords.e = PotentialEnergy(-0.144, "Ha")

    with caplog.at_level("ERROR"):
        peak = imgpair.ts_guess
    assert "Energy values are missing in the trajectory" in caplog.text

    imgpair.right_coords.e = PotentialEnergy(-0.145, "Ha")
    imgpair.left_coords.g = np.ones_like(imgpair.left_coords)  # spoof gradient
    peak = imgpair.ts_guess
    assert peak is not None

    assert np.allclose(peak.coordinates.flatten(), imgpair.left_coords)
    assert np.isclose(peak.energy, -0.144)
    assert np.allclose(peak.gradient.flatten(), imgpair.left_coords.g)


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_dhs_single_step():
    step_size = 0.2
    reactant = Molecule("da_reactant.xyz")
    product = Molecule("da_product.xyz")

    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=200,
        step_size=step_size,
        dist_tol=1.0,
    )

    dhs.imgpair.set_method_and_n_cores(method=XTB(), n_cores=1)
    dhs._method = XTB()
    dhs._initialise_run()

    imgpair = dhs.imgpair
    assert imgpair.left_coords.e is not None
    assert imgpair.right_coords.e is not None
    old_dist = imgpair.dist
    assert imgpair.left_coords.e > imgpair.right_coords.e

    # take a single step
    dhs._step()
    # step should be on lower energy image
    assert len(imgpair._left_history) == 1
    assert len(imgpair._right_history) == 2
    new_dist = imgpair.dist
    # image should move exactly by step_size
    assert np.isclose(old_dist - new_dist, step_size)


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_dhs_gs_single_step(caplog):
    step_size = 0.2
    reactant = Molecule("da_reactant.xyz")
    product = Molecule("da_product.xyz")

    # DHS-GS from end point of DHS (first step is always 100% DHS)
    dhs_gs = DHSGS(
        initial_species=reactant,
        final_species=product,
        maxiter=200,
        step_size=step_size,
        dist_tol=1.0,
        gs_mix=0.5,
    )

    dhs_gs.imgpair.set_method_and_n_cores(method=XTB(), n_cores=1)
    dhs_gs._method = XTB()
    dhs_gs._initialise_run()
    # take one step
    with caplog.at_level("INFO"):
        dhs_gs._step()
        dhs_gs._log_convergence()
    assert "DHS-GS" in caplog.text
    right_pred = dhs_gs._get_dhs_step(ImageSide.right)  # 50% DHS + 50% GS

    imgpair = dhs_gs.imgpair
    assert imgpair.left_coords.e > imgpair.right_coords.e
    assert len(imgpair._left_history) == 1
    assert len(imgpair._right_history) == 2

    hybrid_step = right_pred - imgpair.right_coords
    dhs_step = imgpair.dist_vec
    dhs_step = dhs_step / np.linalg.norm(dhs_step) * step_size
    gs_step = imgpair._right_history[-1] - imgpair._right_history[-2]

    assert np.allclose(hybrid_step, 0.5 * dhs_step + 0.5 * gs_step)


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_dhs_diels_alder():
    set_dist_tol = 1.0  # angstrom

    # Use almost converged images for quick calculation
    reactant = Molecule("da_rct_image.xyz")
    product = Molecule("da_prod_image.xyz")
    # TS optimized with ORCA using xTB method
    true_ts = Molecule("da_ts_orca_xtb.xyz")

    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=200,
        step_size=0.2,
        dist_tol=set_dist_tol,
        gtol=1.0e-3,
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

    # trajectories and default energy plot should be in "dhs" folder
    assert os.path.isfile("dhs/initial_species_DHS.trj.xyz")
    assert os.path.isfile("dhs/final_species_DHS.trj.xyz")
    assert os.path.isfile("dhs/total_trajectory_DHS.trj.xyz")
    assert os.path.isfile("dhs/DHS_path_energy_plot.pdf")

    # now run CI-NEB from end points
    dhs.run_cineb()
    assert dhs.imgpair._cineb_coords is not None
    assert dhs.imgpair._cineb_coords.e is not None
    peak = dhs.ts_guess

    rmsd = calc_rmsd(peak.coordinates, true_ts.coordinates)
    # Euclidean distance = rmsd * sqrt(n_atoms)
    new_distance = rmsd * np.sqrt(peak.n_atoms)

    # Now distance should be closer
    assert new_distance < distance
    assert new_distance < 0.6 * set_dist_tol

    # test graph plotting again, with all available options
    dhs.plot_energies("DHS_relative_dist.pdf", distance_metric="relative")
    dhs.plot_energies("DHS_by_index.pdf", distance_metric="index")
    dhs.plot_energies("DHS_dist_from_start.pdf", distance_metric="from_start")
    for filename in [
        "DHS_relative_dist.pdf",
        "DHS_by_index.pdf",
        "DHS_dist_from_start.pdf",
    ]:
        assert os.path.isfile(filename)


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_dhs_jumping_over_barrier(caplog):
    # Use almost converged images for quick calculation
    reactant = Molecule("da_rct_image.xyz")
    product = Molecule("da_prod_image.xyz")

    # run DHS with large step sizes, which will make one side jump
    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=200,
        step_size=0.6,
        dist_tol=0.3,  # smaller dist_tol also to make one side jump
        gtol=5.0e-4,
        barrier_check=True,
        cineb_at_conv=True,
    )
    with caplog.at_level("WARNING"):
        dhs.calculate(method=XTB(), n_cores=Config.n_cores)

    assert "One image has probably jumped over the barrier" in caplog.text
    assert not dhs.converged
    # CI-NEB should not be run if one image has jumped over
    assert "has not converged properly or one side has jumped" in caplog.text
    assert dhs.imgpair._cineb_coords is None


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_dhs_stops_if_microiter_exceeded(caplog):
    reactant = Molecule("da_rct_image.xyz")
    product = Molecule("da_prod_image.xyz")

    # run DHS with low maxiter
    dhs = DHS(
        initial_species=reactant,
        final_species=product,
        maxiter=10,
        step_size=0.2,
        dist_tol=1.0,
        gtol=5.0e-4,
        barrier_check=True,
    )
    with caplog.at_level("WARNING"):
        dhs.calculate(method=XTB(), n_cores=1)

    assert not dhs.converged
    text = "Reached the maximum number of micro-iterations"
    assert text in caplog.text


def test_method_names():
    # check all method names are properly written
    mol1 = Molecule(smiles="CCO")
    dhs = DHS(mol1, mol1.copy())
    assert dhs._name == "DHS"
    dhs_gs = DHSGS(mol1, mol1.copy())
    assert dhs_gs._name == "DHSGS"
