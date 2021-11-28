import os
import numpy as np
import pytest
from .. import testutils
from autode.utils import work_in_tmp_dir
from autode.pes.relaxed import RelaxedPESnD
here = os.path.dirname(os.path.abspath(__file__))


def test_save_empty():

    pes = RelaxedPESnD()

    with pytest.raises(ValueError):
        pes.save('tmp')


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_save_1d():
    pes = RelaxedPESnD(rs={(0, 1): (0.1, 0.2, 3)})
    assert pes.shape == (3,)

    pes.save('tmp')
    # .npz should be automatically added
    assert os.path.exists('tmp.npz')

    with pytest.raises(Exception):
        pes.load('a_file_that_does_not_exist')

    pes.load('tmp.npz')
    assert pes.shape == (3,)

    # Should also be able to save the pure .txt file of energies
    pes.save('tmp.txt')
    assert os.path.exists('tmp.txt')
    assert np.allclose(np.loadtxt('tmp.txt'),
                       np.zeros(3))

    # Cannot reload from a .txt file
    with pytest.raises(ValueError):
        pes.load('tmp.txt')


@testutils.work_in_zipped_dir(os.path.join(here, 'data.zip'))
def test_save_plot():
    """Not easy to test what these look like, so just check that
    the plots exist..."""

    pes = RelaxedPESnD()

    for filename in ('pes1d_water.npz', 'pes2d_water.npz'):
        pes.load(filename)

        for interp_factor in (0, 2):
            pes.plot('tmp.pdf', interp_factor=interp_factor)
            assert os.path.exists('tmp.pdf')
            os.remove('tmp.pdf')
