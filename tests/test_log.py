from autode import log
import os


def test_log_level():

    # Default is debug
    if 'AUTODE_LOG_LEVEL' in os.environ:
        os.environ.pop('AUTODE_LOG_LEVEL')

    assert log.get_log_level() == log.logging.CRITICAL

    os.environ['AUTODE_LOG_LEVEL'] = 'DEBUG'
    assert log.get_log_level() == log.logging.DEBUG

    os.environ['AUTODE_LOG_LEVEL'] = 'WARNING'
    assert log.get_log_level() == log.logging.WARNING

    os.environ['AUTODE_LOG_LEVEL'] = 'INFO'
    assert log.get_log_level() == log.logging.INFO

    # Setting AUTODE_LOG_FILE to anything should log to a log file
    os.environ['AUTODE_LOG_FILE'] = 'true'
    assert log.log_to_log_file() is True
