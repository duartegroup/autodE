from autode import config


def test_config():

    list_attr = ['scan_keywords', 'opt_keywords', 'opt_ts_keywords', 'sp_keywords', 'conf_opt_keywords']
    int_attr = ['orca_max_core', 'n_cores']
    str_attr = ['opt_ts_block', 'path_to_orca', 'path_to_xtb']
    config_attr = list_attr + int_attr + str_attr

    assert all([hasattr(config.Config, attr) for attr in config_attr])
    assert all([type(getattr(config.Config, attr)) == list for attr in list_attr])
    assert all([type(getattr(config.Config, attr)) == int for attr in int_attr])
    assert all([type(getattr(config.Config, attr)) == str for attr in str_attr])
