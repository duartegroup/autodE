import pytest
from autode.wrappers.keywords.functionals import pbe
from autode.wrappers.keywords.dispersion import d3bj
from autode.wrappers.keywords.wf import hf
from autode.wrappers.keywords.basis_sets import def2tzvp, def2ecp
from autode.config import Config
from copy import deepcopy
from autode.wrappers.keywords import (
    Keywords,
    KeywordsSet,
    ECP,
    Functional,
    BasisSet,
    DispersionCorrection,
    ImplicitSolventType,
    RI,
    WFMethod,
    MaxOptCycles,
    GradientKeywords,
    OptKeywords,
    HessianKeywords,
    SinglePointKeywords,
)


def test_keywords():

    keywords = OptKeywords(keyword_list=None)
    assert keywords._list == []

    assert isinstance(GradientKeywords(None), Keywords)
    assert isinstance(OptKeywords(None), Keywords)
    assert isinstance(SinglePointKeywords(None), Keywords)

    keywords = OptKeywords(keyword_list=["test"])
    assert "test" in str(keywords)
    assert "test" in repr(keywords)

    # Should not add a keyword that's already there
    keywords.append("test")
    assert len(keywords._list) == 1

    assert hasattr(keywords, "copy")

    # Should have reasonable names
    assert "opt" in repr(OptKeywords(None)).lower()
    assert "hess" in repr(HessianKeywords(None)).lower()
    assert "grad" in repr(GradientKeywords(None)).lower()
    assert "sp" in repr(SinglePointKeywords(None)).lower()

    assert "pbe" in repr(pbe).lower()

    keywords = OptKeywords([pbe, def2tzvp, d3bj])
    assert len(keywords) == 3
    assert keywords.bstring is not None
    assert "pbe" in keywords.bstring.lower()
    assert "def2" in keywords.bstring.lower()
    assert "d3bj" in keywords.bstring.lower()

    # Keywords have a defined order
    assert "pbe" in keywords[0].name.lower()

    assert "hf" in OptKeywords([hf, def2tzvp]).bstring.lower()


def test_wf_keywords_string():

    assert "hf" in OptKeywords([hf]).method_string.lower()


def test_set_keywordsset():

    kwset = deepcopy(Config.G09.keywords)
    assert hasattr(kwset, "opt")
    assert "keywords" in repr(kwset).lower()
    assert kwset.low_sp is not None

    kwset.set_opt_functional(pbe)
    assert kwset.opt.functional.lower() == "pbe"
    assert kwset.opt_ts.functional.lower() == "pbe"
    assert kwset.hess.functional.lower() == "pbe"
    assert kwset.grad.functional.lower() == "pbe"
    assert kwset.sp.functional.lower() != "pbe"

    # Should now all be PBE functionals
    kwset.set_functional(pbe)
    assert kwset.sp.functional.lower() == "pbe"
    assert kwset.low_sp.functional.lower() == "pbe"

    kwset.set_opt_basis_set(def2tzvp)
    assert kwset.opt.basis_set.lower() == "def2-tzvp"

    # Should admit no dispersion correction
    assert kwset.opt.dispersion is not None
    kwset.set_dispersion(None)
    assert kwset.opt.dispersion is None
    assert kwset.sp.dispersion is None


def test_keyword_repr():

    assert "basis" in repr(BasisSet("pbe")).lower()
    assert "disp" in repr(DispersionCorrection("d3")).lower()
    assert "func" in repr(Functional("pbe")).lower()
    assert "solv" in repr(ImplicitSolventType("cosmo")).lower()
    assert "resolution" in repr(RI("ri")).lower()
    assert "wavefunction" in repr(WFMethod("hf")).lower()
    assert "effective" in repr(def2ecp).lower()
    assert "max" in repr(MaxOptCycles(10)).lower()


def test_ecp():

    kwds_set = KeywordsSet()
    assert kwds_set.opt.ecp is None

    ecp1 = ECP(name="tmp_ecp", min_atomic_number=10)
    ecp2 = ECP(name="tmp_ecp", min_atomic_number=20)

    assert not ecp1 == ecp2
    assert not ecp1 == "tmp_ecp"
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
        _ = MaxOptCycles("a")

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


def test_type_init():

    keyword_set = Config.ORCA.keywords.copy()

    for opt_type in ("low_opt", "opt", "opt_ts"):
        kwds = getattr(keyword_set, opt_type)
        assert isinstance(kwds, OptKeywords)

    for opt_type in ("low_sp", "sp"):
        kwds = getattr(keyword_set, opt_type)
        assert kwds is None or isinstance(kwds, SinglePointKeywords)

    assert isinstance(keyword_set.hess, HessianKeywords)
    assert isinstance(keyword_set.grad, GradientKeywords)


def test_type_inference():
    """Ensure that setting keywords with lists retains their type"""

    keyword_set = Config.ORCA.keywords.copy()

    keyword_set.low_opt = ["a"]
    assert isinstance(keyword_set.low_opt, OptKeywords)

    keyword_set.low_opt = OptKeywords(["a", "different", "set"])
    assert isinstance(keyword_set.low_opt, OptKeywords)

    keyword_set.opt = ["a"]
    assert isinstance(keyword_set.opt, OptKeywords)

    keyword_set.opt_ts = ["a"]
    assert isinstance(keyword_set.opt_ts, OptKeywords)

    keyword_set.sp = ["a"]
    assert isinstance(keyword_set.sp, SinglePointKeywords)

    keyword_set.low_sp = ["a"]
    assert isinstance(keyword_set.low_sp, SinglePointKeywords)
    keyword_set.low_sp = None

    keyword_set.grad = ["a"]
    assert isinstance(keyword_set.grad, GradientKeywords)

    keyword_set.hess = ["a"]
    assert isinstance(keyword_set.hess, HessianKeywords)


def test_keywords_contain():

    kwds = SinglePointKeywords(["PBE", "Opt"])

    assert kwds.contain_any_of("pbe")
    assert kwds.contain_any_of("pbe", "opt")
    assert kwds.contain_any_of("opt")

    assert not kwds.contain_any_of("PBE0")


def test_functional_equality():

    assert Functional("PBE0") == Functional("PBE0")
    assert Functional("PBE0") != 1


def test_keyword_addition():

    a = OptKeywords("a")
    b = OptKeywords("b")

    assert "b" in (a + b)
    assert "b" in a + ["b"]

    with pytest.raises(ValueError):
        _ = a + 2

    assert isinstance(a + b, OptKeywords)
