import autode.pes_2d as pes_2d
import numpy as np
from autode.molecule import Molecule
from autode.reactions import Dissociation
from autode.wrappers.ORCA import orca
from autode.config import Config
import os

Config.high_qual_plots = False

here = os.path.dirname(os.path.abspath(__file__))


def test_polyfit2d():
    x = [6, 2, 18, 12]
    y = [4, 5, 6, 7]
    z = [14, 12, 30, 26]
    coeff_mat = pes_2d.polyfit2d(x, y, z, order=1)
    assert len(coeff_mat) == 2
    assert coeff_mat.shape[0] == 2
    assert coeff_mat.shape[1] == 2
    assert -0.005 < coeff_mat[0, 0] < 0.005
    assert 1.995 < coeff_mat[0, 1] < 2.005
    assert 0.995 < coeff_mat[1, 0] < 1.005
    assert -0.005 < coeff_mat[1, 1] < 0.005


def test_get_ts_guess_2dscan():
    os.chdir(os.path.join(here, 'data'))



    for filename in os.listdir(os.getcwd()):
        if filename.endswith(('.inp', '.png')) or 'animation' in filename:
            os.remove(filename)

    os.chdir(here)
