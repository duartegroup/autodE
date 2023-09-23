import os
import numpy as np
import pytest

from autode.species import Molecule
from autode.methods import XTB
from autode.bracket.ieip import IEIP, ElasticImagePair, IEIPMicroImagePair
from autode.bracket.ieip import _calculate_low_sp_energy_for_species
from autode.geom import calc_rmsd
from ..testutils import requires_working_xtb_install, work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
datazip = os.path.join(here, "data", "geometries.zip")


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_ieip_initialisation():
    rct = Molecule("da_reactant.xyz")
    prod = Molecule("da_product.xyz")
    ieip = IEIP(rct, prod)
    dist_orig = ieip.imgpair.dist
    ieip.imgpair.set_method_and_n_cores(method=XTB(), n_cores=1)
    ieip._initialise_run()
    # redistribution means one pair of coords should be added
    assert len(ieip.imgpair._left_history) == 2
    assert len(ieip.imgpair._right_history) == 2
    # the distance should be lower
    dist_curr = ieip.imgpair.dist
    assert dist_curr < dist_orig
    # the current geometries should be reasonable
    assert ieip.imgpair._left_image.has_reasonable_coordinates
    assert ieip.imgpair._right_image.has_reasonable_coordinates
    # the hessian and gradients should be set
    assert ieip.imgpair.left_coords.g is not None
    assert ieip.imgpair.right_coords.g is not None
    assert ieip.imgpair.left_coords.h is not None
    assert ieip.imgpair.right_coords.h is not None
    # the target distance and rms g should be set
    assert ieip._target_dist is not None
    assert ieip._target_rms_g is not None
    # there should be no ts guess without at least one iter
    assert ieip.ts_guess is None


@requires_working_xtb_install
def test_low_sp_func():
    mol = Molecule(smiles="CCO")
    en = _calculate_low_sp_energy_for_species(mol, method=XTB(), n_cores=1)
    assert en is not None
    assert hasattr(en, "units")


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_ieip_microiters():
    micro_step_size = 1e-4
    rct = Molecule("da_reactant.xyz")
    prod = Molecule("da_product.xyz")
    imgpair = ElasticImagePair(rct, prod)
    imgpair.set_method_and_n_cores(method=XTB(), n_cores=1)
    imgpair.update_both_img_engrad()
    imgpair.redistribute_imagepair(ll_neb_interp=False)
    imgpair.update_both_img_engrad()
    imgpair.update_both_img_hessian_by_calc()
    micro_imgpair = IEIPMicroImagePair(
        imgpair._left_image,
        imgpair.left_coords,
        imgpair.right_coords,
        micro_step_size=micro_step_size,
        target_dist=imgpair.dist * 0.9,
    )
    micro_imgpair.update_both_img_engrad()
    micro_imgpair.take_micro_step()
    # the step size should be equal to the provided step size
    assert np.isclose(micro_imgpair.max_displacement, micro_step_size)
    # need to update gradients from the Taylor surfaces before step
    with pytest.raises(AssertionError):
        micro_imgpair.take_micro_step()
    for _ in range(3):
        micro_imgpair.update_both_img_engrad()
        micro_imgpair.take_micro_step()
    # the previous gradient matrices should be flushed after three iters
    assert micro_imgpair._left_history[1].g is None
    assert micro_imgpair._right_history[1].g is None


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_ieip_diels_alder():
    set_dist_tol = 1.0  # Angstrom
    set_step_size = 0.2  # Angstrom
    # Use almost converged images for quick calculation
    reactant = Molecule("da_rct_image.xyz")
    product = Molecule("da_prod_image.xyz")
    # TS optimized with ORCA using xTB method
    true_ts = Molecule("da_ts_orca_xtb.xyz")
    ieip = IEIP(
        initial_species=reactant,
        final_species=product,
        use_ll_neb_interp=False,
        interp_fraction=3 / 4,
        dist_tol=set_dist_tol,
        maxiter=5,
        max_macro_step=set_step_size,
    )
    dist_curr = ieip.imgpair.dist
    ieip.imgpair.set_method_and_n_cores(XTB(), n_cores=1)
    ieip.imgpair.update_both_img_engrad()
    prev_de = abs(ieip.imgpair.left_coords.e - ieip.imgpair.right_coords.e)
    prev_dist = ieip.imgpair.dist
    # load hessian from txt to save time
    ieip.imgpair.left_coords.h = np.loadtxt("da_rct_image_hess.txt")
    ieip.imgpair.right_coords.h = np.loadtxt("da_prod_image_hess.txt")
    ieip._target_dist = dist_curr
    prev_rms_g = min(max(dist_curr / set_dist_tol, 1), 2) * ieip._gtol
    ieip._target_rms_g = prev_rms_g

    ieip._step()
    # step sizes should be within set size
    assert ieip.imgpair.last_left_step_size <= set_step_size
    assert ieip.imgpair.last_right_step_size <= set_step_size
    # energy should be more equalised after a step
    curr_de = abs(ieip.imgpair.left_coords.e - ieip.imgpair.right_coords.e)
    assert curr_de < prev_de

    # now take only 5 steps
    while not ieip.converged:
        ieip._step()
        if ieip._exceeded_maximum_iteration:
            break
    assert ieip._macro_iter == 5
    assert len(ieip.imgpair._left_history) == 7
    # check that distance is going down and the rms g is also tightened
    assert ieip.imgpair.dist <= prev_dist
    assert ieip._target_rms_g < prev_rms_g
    # interpolated guess should be reasonable even if not converged
    peak = ieip.ts_guess
    rmsd = calc_rmsd(peak.coordinates, true_ts.coordinates)
    assert rmsd < 0.2
