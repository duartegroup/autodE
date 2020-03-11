import autode.pes_2d as pes_2d
import numpy as np
from autode.molecule import Molecule
from autode.reactions import Dissociation
from autode.wrappers.ORCA import ORCA
from autode.config import Config
import os

Config.high_qual_plots = False

here = os.path.dirname(os.path.abspath(__file__))


def test_replace_none():
    a = [0, 1, 2, None, 4, 5, 6]
    no_none = list(pes_2d.replace_none(a))
    assert no_none == [0, 1, 2, 2, 4, 5, 6]


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


def test_closest_dist_to_saddle():
    dists = [(0, 0,), (1, 1), (2, 2), (3, 3)]
    r1 = 0.9
    r2 = 1.6
    assert pes_2d.get_closest_point_dists_to_saddle(r1, r2, dists) == (1, 1)


def test_find_2dpes_saddle_xyzs():
    os.chdir(os.path.join(here, 'data'))

    method = ORCA()
    dist_dict = {}
    dist_dict[(0.75, 0.75)] = ([['H', 3, 0, 0]], 0)
    dist_dict[(0.75, 1.5)] = ([['H', 1, 0, 0]], 10)
    dist_dict[(1.125, 1.125)] = ([['H', 4, 0, 0]], 5)
    dist_dict[(1.5, 0.75)] = ([['H', 2, 0, 0]], 10)
    dist_dict[(1.5, 1.5)] = ([['H', 0, 0, 0]], 0)
    saddle_xyzs = pes_2d.find_2dpes_saddlepoint_xyzs(dist_dict, 'test', 'test', method, n_points=4, order=1)
    assert type(saddle_xyzs) == tuple
    assert saddle_xyzs[0] == [['H', 4, 0, 0]]
    assert type(saddle_xyzs[1]) == tuple
    assert 1.124999 < saddle_xyzs[1][0] < 1.125001

    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.png') or 'animation' in filename:
            os.remove(filename)

    os.chdir(here)


def test_get_ts_guess_2dscan():
    os.chdir(os.path.join(here, 'data'))

    h4_xyzs = [['H', 0.0, 0.0, 0.0], ['H', 0.6, 0.0, 0.0],
               ['H', 0.0, -0.6, 0.0], ['H', 0.6, -0.6, 0.0]]
    mol = Molecule(name='h4', xyzs=h4_xyzs, mult=1)
    method = ORCA()

    ts_guess = pes_2d.get_ts_guess_2d(mol=mol, product=mol, active_bond1=(0, 2), active_bond2=(1, 3), n_steps=2, name='h4',
                                      reaction_class=Dissociation, method=method, keywords=Config.ORCA.keywords.low_opt,
                                      delta_dist1=0.2, delta_dist2=0.2, e_grid_points=4, polynomial_order=1, solvent_mol=None)

    assert len(ts_guess.active_bonds) == 2
    assert os.path.exists('h4_0_2_1_3_2dscan_ORCA.png')

    for filename in os.listdir(os.getcwd()):
        if filename.endswith(('.inp', '.png')) or 'animation' in filename:
            os.remove(filename)

    os.chdir(here)
