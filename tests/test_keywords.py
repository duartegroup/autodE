from autode.wrappers.keywords import Keywords
from autode.wrappers.keywords import GradientKeywords
from autode.wrappers.keywords import OptKeywords
from autode.wrappers.keywords import HessianKeywords
from autode.wrappers.keywords import SinglePointKeywords
from autode.wrappers.functionals import pbe
from autode.wrappers.basis_sets import def2tzvp
from autode.config import Config
from copy import deepcopy


def test_keywords():

    Config.keyword_prefixes = True
    keywords = Keywords(keyword_list=None)
    assert keywords.keyword_list == []

    assert isinstance(GradientKeywords(None), Keywords)
    assert isinstance(OptKeywords(None), Keywords)
    assert isinstance(SinglePointKeywords(None), Keywords)

    keywords = Keywords(keyword_list=['test'])

    # Should not add a keyword that's already there
    keywords.append('test')
    assert len(keywords.keyword_list) == 1

    assert hasattr(keywords, 'copy')

    # Should have reasonable names
    assert 'opt' in str(OptKeywords(None)).lower()
    assert 'hess' in str(HessianKeywords(None)).lower()
    assert 'grad' in str(GradientKeywords(None)).lower()
    assert 'sp' in str(SinglePointKeywords(None)).lower()
    Config.keyword_prefixes = False


def test_set_keywordsset():

    kwset = deepcopy(Config.G09.keywords)
    assert hasattr(kwset, 'opt')

    kwset.set_opt_functional(pbe)
    assert kwset.opt.functional().lower() == 'pbe'
    assert kwset.opt_ts.functional().lower() == 'pbe'
    assert kwset.hess.functional().lower() == 'pbe'
    assert kwset.grad.functional().lower() == 'pbe'
    assert kwset.sp.functional().lower() != 'pbe'

    # Should now all be PBE functionals
    kwset.set_functional(pbe)
    assert kwset.sp.functional().lower() == 'pbe'

    kwset.set_opt_basis_set(def2tzvp)
    assert kwset.opt.basis_set().lower() == 'def2-tzvp'

    # Should admit no dispersion correction
    assert kwset.opt.dispersion() is not None
    kwset.set_dispersion(None)
    assert kwset.opt.dispersion() is None
    assert kwset.sp.dispersion() is None
