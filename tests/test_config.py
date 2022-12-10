import numpy as np
import pytest
from copy import deepcopy
from autode.config import Config
from autode.values import Allocation, Distance
from autode.wrappers.keywords import KeywordsSet
from autode.wrappers.keywords import Keywords
from autode.transition_states.templates import get_ts_template_folder_path


def test_config():

    keywords_attr = ["low_opt", "grad", "opt", "opt_ts", "hess", "sp"]
    global_attr = ["max_core", "n_cores"]

    assert all([hasattr(Config, attr) for attr in global_attr])
    assert all([hasattr(Config.ORCA, attr) for attr in ["path", "keywords"]])

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
    assert int(_config.max_core.to("MB")) == 1
    assert "mb" in repr(_config.max_core.to("MB"))

    # and should be able to convert MB -> GB
    _config.max_core = Allocation(1, units="GB")
    assert int(_config.max_core.to("MB")) == 1000


@pytest.mark.parametrize("factor", (-0.1, 1.1, "a string"))
def test_invalid_freq_scale_factor(factor):

    with pytest.raises(Exception):
        Config.freq_scale_factor = factor


def test_unknown_attr():

    # Attributes not already present should raise an exception e.g. for
    # misspelling
    with pytest.raises(Exception):
        Config.maxcore = 1


def test_step_size_setter():

    _config = deepcopy(Config)

    # Distances cannot be negative
    with pytest.raises(ValueError):
        _config.max_step_size = -0.11

    # Setting the attribute should default to a Distance (Å)
    _config.max_step_size = 0.1
    assert np.isclose(_config.max_step_size.to("ang"), 0.1)

    # Setting in Bohr should convert to angstroms
    _config.max_step_size = Distance(0.2, units="a0")
    assert np.isclose(_config.max_step_size.to("ang"), 0.1, atol=0.02)


def test_invalid_get_ts_template_folder_path():

    Config.ts_template_folder_path = ""

    with pytest.raises(ValueError):
        _ = get_ts_template_folder_path(None)

    Config.ts_template_folder_path = None
