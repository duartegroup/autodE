from autode import config


def test_config():

    keywords_list_attr = ['low_opt', 'grad', 'opt', 'opt_ts', 'hess', 'sp']
    keywords_str_attr = ['optts_block']
    global_attr = ['max_core', 'n_cores']

    assert all([hasattr(config.Config, attr) for attr in global_attr])
    assert all([hasattr(config.Config.ORCA, attr) for attr in ['path', 'keywords']])

    assert all([type(getattr(config.Config.ORCA.keywords, attr)) == list for attr in keywords_list_attr])
    assert all([type(getattr(config.Config.ORCA.keywords, attr)) == str for attr in keywords_str_attr])
