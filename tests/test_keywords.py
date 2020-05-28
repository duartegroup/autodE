from autode.wrappers.keywords import Keywords
from autode.wrappers.keywords import GradientKeywords
from autode.wrappers.keywords import OptKeywords
from autode.wrappers.keywords import SinglePointKeywords


def test_keywords():

    keywords = Keywords(keyword_list=None)
    assert keywords.keyword_list == []

    assert isinstance(GradientKeywords(None), Keywords)
    assert isinstance(OptKeywords(None), Keywords)
    assert isinstance(SinglePointKeywords(None), Keywords)

    keywords = Keywords(keyword_list=['test'])

    # Should not readd a keyword that's already there
    keywords.append('test')
    assert len(keywords.keyword_list) == 1

    assert hasattr(keywords, 'copy')
