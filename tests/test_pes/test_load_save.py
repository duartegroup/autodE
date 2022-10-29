import os
import numpy as np
import pytest
from .. import testutils
from .sample_pes import TestPES
from autode.utils import work_in_tmp_dir
from autode.pes.pes_nd import EnergyArray as Energies

here = os.path.dirname(os.path.abspath(__file__))


def test_save_empty():

    pes = TestPES(rs={})

    with pytest.raises(ValueError):
        pes.save("tmp")


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_save_1d():
    pes = TestPES(rs={(0, 1): (0.1, 0.2, 3)})
    assert pes.shape == (3,)

    pes.save("tmp")
    # .npz should be automatically added
    assert os.path.exists("tmp.npz")

    with pytest.raises(Exception):
        pes.load("a_file_that_does_not_exist")

    pes.load("tmp.npz")
    assert pes.shape == (3,)

    # Should also be able to save the pure .txt file of energies
    pes.save("tmp.txt")
    assert os.path.exists("tmp.txt")

    loaded_arr = np.loadtxt("tmp.txt")
    assert loaded_arr.shape == (3,)

    # Cannot reload from a .txt file
    with pytest.raises(ValueError):
        pes.load("tmp.txt")


def save_3d_as_text_file():

    pes = TestPES(
        rs={
            (0, 1): (0.1, 0.2, 3),
            (1, 2): (0.1, 0.2, 3),
            (2, 3): (0.1, 0.2, 3),
        }
    )
    pes._energies = Energies(np.ones(shape=(3, 3)))
    pes.save(filename="tmp.txt")

    # 3D, or more PESs should be flattened to be saved as a .txt file
    assert os.path.exists("tmp.txt")
    assert np.loadtxt("tmp.txt").shape == (9,)


@testutils.work_in_zipped_dir(os.path.join(here, "data.zip"))
def test_save_plot():
    """Not easy to test what these look like, so just check that
    the plots exist..."""

    pes = TestPES(rs={})

    for filename in ("pes1d_water.npz", "pes2d_water.npz"):
        pes.load(filename)

        with pytest.raises(Exception):
            # Cannot plot with a negative interpolation factor
            pes.plot("tmp.pdf", interp_factor=-1)

        for interp_factor in (0, 2):
            pes.plot("tmp.pdf", interp_factor=interp_factor)
            assert os.path.exists("tmp.pdf")
            os.remove("tmp.pdf")
