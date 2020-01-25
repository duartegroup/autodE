from autode.conformers import conf_gen
from autode.config import Config
import os

here = os.path.dirname(os.path.abspath(__file__))


def test_conf_gen():
    os.chdir(os.path.join(here, 'data'))
    xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]
    Config.n_cores = 1

    conf_list = conf_gen.gen_simanl_conf_xyzs(name='H2', init_xyzs=xyz_list, bond_list=[(0, 1)], stereocentres=[0], n_simanls=1)
    assert len(conf_list) == 1

    conf_list = conf_gen.gen_simanl_conf_xyzs(name='H2', init_xyzs=xyz_list, bond_list=[(0, 1)], stereocentres=None, n_simanls=2)
    assert len(conf_list) == 2

    conf_list2 = conf_gen.gen_simanl_conf_xyzs(name='h2', init_xyzs=xyz_list, bond_list=[(0, 1)], stereocentres=None, n_simanls=2)
    assert len(conf_list2) == 2
    assert conf_list2[0] == [['H', 0.0, 0.0, 0.0], ['H', -1.0, 0.0, 0.0]]

    os.chdir(here)
