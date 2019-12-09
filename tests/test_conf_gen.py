from autode.conformers import conf_gen
from autode.config import Config


def test_conf_gen():
    xyz_list = [['H', 0.0, 0.0, 0.0], ['H', 1.0, 0.0, 0.0]]
    Config.n_cores = 1

    conf_list = conf_gen.gen_simanl_conf_xyzs(
        name='H2', init_xyzs=xyz_list, bond_list=[(0, 1)], stereocentres=None, n_simanls=1)
    assert len(conf_list) == 1
