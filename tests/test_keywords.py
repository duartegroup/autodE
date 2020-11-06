from autode.wrappers.keywords import Keywords
from autode.wrappers.keywords import GradientKeywords
from autode.wrappers.keywords import OptKeywords
from autode.wrappers.keywords import HessianKeywords
from autode.wrappers.keywords import SinglePointKeywords


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
