from autode import log
import os

def test_log_level():
    os.environ['AUTODE_LOG_LEVEL'] = 'DEBUG'
    assert log.get_log_level() == log.logging.DEBUG

    os.environ['AUTODE_LOG_LEVEL'] = 'WARNING'
    assert log.get_log_level() == log.logging.WARNING

    os.environ['AUTODE_LOG_LEVEL'] = 'INFO'
    assert log.get_log_level() == log.logging.INFO
