from autode import config


def test_config():

    orca_str_attr = ['opt_ts_block']
    orca_list_attr = ['scan_keywords', 'opt_keywords',
                      'opt_ts_keywords', 'sp_keywords', 'conf_opt_keywords']
    global_attr = ['max_core', 'n_cores']

    assert all([hasattr(config.Config, attr) for attr in global_attr])
    assert all([hasattr(config.Config.ORCA, attr)
                for attr in orca_list_attr + orca_str_attr])

    assert all([type(getattr(config.Config.ORCA, attr))
                == list for attr in orca_list_attr])
    assert all([type(getattr(config.Config.ORCA, attr))
                == str for attr in orca_str_attr])
