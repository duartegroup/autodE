import os
import pytest
from autode.utils import work_in_tmp_dir
from autode.pes.relaxed import RelaxedPESnD


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

    with pytest.raises(FileNotFoundError):
        pes.load('a_file_that_does_not_exist')

    pes.load('tmp.npz')
    assert pes.shape == (3,)
