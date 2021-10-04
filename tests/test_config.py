import pytest
from copy import deepcopy
from autode.config import Config
from autode.values import Allocation
from autode.wrappers.keywords import KeywordsSet
from autode.wrappers.keywords import Keywords


def test_config():

    keywords_attr = ['low_opt', 'grad', 'opt', 'opt_ts', 'hess', 'sp']
    global_attr = ['max_core', 'n_cores']

    assert all([hasattr(Config, attr) for attr in global_attr])
    assert all([hasattr(Config.ORCA, attr) for attr in ['path', 'keywords']])

    def assert_has_correct_keywords(keywords):
        for attribute in keywords_attr:
            assert hasattr(keywords, attribute)
            assert isinstance(getattr(keywords, attribute), Keywords)

    assert isinstance(Config.ORCA.keywords, KeywordsSet)
    assert_has_correct_keywords(keywords=Config.G09.keywords)
    assert_has_correct_keywords(keywords=Config.MOPAC.keywords)
    assert_has_correct_keywords(keywords=Config.XTB.keywords)
    assert_has_correct_keywords(keywords=Config.ORCA.keywords)
    assert_has_correct_keywords(keywords=Config.NWChem.keywords)


def test_maxcore_setter():

    _config = deepcopy(Config)

    # Cannot have a negative allocation
    with pytest.raises(ValueError):
        _config.max_core = -1

    # Default units are megabytes
    _config.max_core = 1
    assert int(_config.max_core.to('MB')) == 1

    # and should be able to convert MB -> GB
    _config.max_core = Allocation(1, units='GB')
    assert int(_config.max_core.to('MB')) == 1000
