import os
import numpy as np
import pytest

from autode.species import Molecule
from autode.methods import XTB
from autode.bracket.ieip import ElasticImagePair, IEIPMicroImagePair
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
    assert np.isclose(micro_imgpair.max_displacement, micro_step_size)
    # need to update gradients from the Taylor surfaces before step
    with pytest.raises(AssertionError):
        micro_imgpair.take_micro_step()
    for _ in range(3):
        micro_imgpair.update_both_img_engrad()
        micro_imgpair.take_micro_step()
    # the previous gradient matrices should be flushed after three
    assert micro_imgpair._left_history[1].g is None
    assert micro_imgpair._right_history[1].g is None
