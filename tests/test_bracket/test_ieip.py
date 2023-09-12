import os
import numpy as np
import pytest

from autode.species import Molecule
from autode.methods import XTB
from autode.bracket.ieip import IEIP, ElasticImagePair, IEIPMicroImagePair
from autode.geom import calc_rmsd
from ..testutils import requires_working_xtb_install, work_in_zipped_dir

here = os.path.dirname(os.path.abspath(__file__))
datazip = os.path.join(here, "data", "geometries.zip")


@requires_working_xtb_install
@work_in_zipped_dir(datazip)
def test_ieip_redistribution():
    rct = Molecule("da_reactant.xyz")
    prod = Molecule("da_product.xyz")
    imgpair = ElasticImagePair(rct, prod)
    dist_orig = imgpair.dist
    imgpair.set_method_and_n_cores(method=XTB(), n_cores=1)
    imgpair.update_both_img_engrad()
    imgpair.redistribute_imagepair(interp_fraction=1 / 4)
    assert len(imgpair._left_history) == 2
    assert len(imgpair._right_history) == 2
    # the distance should be lower
    dist_curr = imgpair.dist
    assert dist_curr < dist_orig
    # the current geometries should be reasonable
    assert imgpair._left_image.has_reasonable_coordinates
    assert imgpair._right_image.has_reasonable_coordinates


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
    set_dist_tol = 0.8  # Angstrom
    set_step_size = 0.1  # Angstrom
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
        max_macro_step=set_step_size,
    )
    dist_curr = ieip.imgpair.dist
    ieip.imgpair.set_method_and_n_cores(XTB(), n_cores=1)
    ieip.imgpair.update_both_img_engrad()
    ieip.imgpair.update_both_img_hessian_by_calc()
    ieip._target_dist = dist_curr
    ieip._target_rms_g = min(max(dist_curr / set_dist_tol, 1), 2) * ieip._gtol

    ieip._step()
    assert ieip.imgpair.last_left_step_size < set_step_size
    assert ieip.imgpair.last_right_step_size < set_step_size
    while not ieip.converged:
        ieip._step()
        if ieip.imgpair.has_jumped_over_barrier:
            raise RuntimeError
        if ieip._exceeded_maximum_iteration:
            break

    assert ieip.converged
    assert ieip.imgpair.dist <= set_dist_tol

    peak = ieip.ts_guess
    rmsd = calc_rmsd(peak.coordinates, true_ts.coordinates)
    assert rmsd < 0.1
