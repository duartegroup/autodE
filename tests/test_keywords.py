import pytest
from autode.wrappers.functionals import pbe
from autode.wrappers.dispersion import d3bj
from autode.wrappers.wf import hf
from autode.wrappers.basis_sets import def2tzvp, def2ecp
from autode.config import Config
from copy import deepcopy
from autode.wrappers.keywords import (Keywords,
                                      KeywordsSet,
                                      ECP,
                                      MaxOptCycles,
                                      GradientKeywords,
                                      OptKeywords,
                                      HessianKeywords,
                                      SinglePointKeywords)


def test_keywords():

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

    keywords = Keywords([pbe, def2tzvp, d3bj])
    assert len(keywords) == 3
    assert keywords.bstring is not None
    assert 'pbe' in keywords.bstring.lower()
    assert 'def2' in keywords.bstring.lower()
    assert 'd3bj' in keywords.bstring.lower()

    assert 'hf' in Keywords([hf, def2tzvp]).bstring.lower()


def test_set_keywordsset():

    kwset = deepcopy(Config.G09.keywords)
    assert hasattr(kwset, 'opt')

    kwset.set_opt_functional(pbe)
    assert kwset.opt.functional.lower() == 'pbe'
    assert kwset.opt_ts.functional.lower() == 'pbe'
    assert kwset.hess.functional.lower() == 'pbe'
    assert kwset.grad.functional.lower() == 'pbe'
    assert kwset.sp.functional.lower() != 'pbe'

    # Should now all be PBE functionals
    kwset.set_functional(pbe)
    assert kwset.sp.functional.lower() == 'pbe'

    kwset.set_opt_basis_set(def2tzvp)
    assert kwset.opt.basis_set.lower() == 'def2-tzvp'

    # Should admit no dispersion correction
    assert kwset.opt.dispersion is not None
    kwset.set_dispersion(None)
    assert kwset.opt.dispersion is None
    assert kwset.sp.dispersion is None


def test_ecp():

    kwds_set = KeywordsSet()
    assert kwds_set.opt.ecp is None

    ecp1 = ECP(name='tmp_ecp', min_atomic_number=10)
    ecp2 = ECP(name='tmp_ecp', min_atomic_number=20)

    assert not ecp1 == ecp2
    assert not ecp1 == 'tmp_ecp'
    ecp2.min_atomic_number = 10
    assert ecp1 == ecp2

    assert isinstance(def2ecp, ECP)
    kwds_set = KeywordsSet(ecp=def2ecp)

    for kwds in kwds_set:
        assert kwds.ecp is not None
        assert isinstance(kwds.ecp, ECP)
        assert kwds.ecp.min_atomic_number == 37

    kwds_set.set_ecp(None)
    for kwds in kwds_set:
        assert kwds.ecp is None


def test_max_opt_cycles():

    with pytest.raises(ValueError):
        _ = MaxOptCycles('a')

    kwds = OptKeywords()
    assert kwds.max_opt_cycles is None

    kwds.append(MaxOptCycles(10))
    assert kwds.max_opt_cycles == MaxOptCycles(10)

    kwds.max_opt_cycles = 20
    assert kwds.max_opt_cycles == MaxOptCycles(20)

    kwds.max_opt_cycles = MaxOptCycles(30)
    assert kwds.max_opt_cycles == MaxOptCycles(30)

    kwds.max_opt_cycles = None
    assert kwds.max_opt_cycles is None

    with pytest.raises(ValueError):
        kwds.max_opt_cycles = -1
