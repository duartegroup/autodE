from autode.config import Config
from autode.wrappers.keywords import KeywordsSet
from autode.wrappers.keywords import Keywords


def test_config():

    keywords_attr = ['low_opt', 'grad', 'opt', 'opt_ts', 'hess', 'sp']
    global_attr = ['max_core', 'n_cores']

    assert all([hasattr(Config, attr) for attr in global_attr])
    assert all([hasattr(Config.ORCA, attr) for attr in ['path', 'keywords']])

    def correct_keywords(keywords):
        for attribute in keywords_attr:
            assert hasattr(keywords, attribute)
            assert isinstance(getattr(keywords, attribute), Keywords)

    assert isinstance(Config.ORCA.keywords, KeywordsSet)
    correct_keywords(keywords=Config.G09.keywords)
    correct_keywords(keywords=Config.MOPAC.keywords)
    correct_keywords(keywords=Config.XTB.keywords)
    correct_keywords(keywords=Config.ORCA.keywords)
    correct_keywords(keywords=Config.NWChem.keywords)


