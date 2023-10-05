import numpy as np
import pytest
from copy import deepcopy
from autode.config import Config, _instantiate_config_opts, _ConfigClass
from autode.values import Allocation, Distance
from autode.wrappers.keywords import KeywordsSet
from autode.wrappers.keywords import Keywords, Functional, BasisSet
from autode.utils import _copy_into_current_config
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


def test_config_simple_copy():
    _config = deepcopy(Config)
    _config_restore = deepcopy(Config)

    _config.n_cores = 31
    _config.ORCA.keywords.low_sp.basis_set = "aug-cc-pVTZ"
    _config.NWChem.keywords.opt.functional = "B3LYP"

    assert Config.n_cores != 31
    assert Config.ORCA.keywords.low_sp.basis_set != BasisSet("aug-cc-pVTZ")
    assert Config.NWChem.keywords.opt.functional != Functional("B3LYP")

    _copy_into_current_config(_config)

    assert Config.n_cores == 31
    assert Config.ORCA.keywords.low_sp.basis_set == BasisSet("aug-cc-pVTZ")
    assert Config.NWChem.keywords.opt.functional == Functional("B3LYP")

    # restore original config
    _copy_into_current_config(_config_restore)


def test_exc_if_not_class_in_config_instantiate_func():
    # passing instance should raise exception
    with pytest.raises(ValueError):
        _instantiate_config_opts(_ConfigClass())

    # when passed class it should work
    test_config = _instantiate_config_opts(_ConfigClass)
    assert test_config is not None


def test_invalid_get_ts_template_folder_path():
    Config.ts_template_folder_path = ""

    with pytest.raises(ValueError):
        _ = get_ts_template_folder_path(None)

    Config.ts_template_folder_path = None
