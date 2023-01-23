import logging
import os

"""
Set up logging with the standard python logging module. Set the log level with
$AUTODE_LOG_LEVEL = {'', ERROR, WARNING, INFO, DEBUG}

i.e. export AUTODE_LOG_LEVEL=DEBUG

Also, set whether to log to a file with
$AUTODE_LOG_FILE = filename

"""


def get_log_level():
    """
    Get the logger level from the $AUTODE_LOG_LEVEL environment variable

    ---------------------------------------------------------------------------
    Returns:
        (int): Log level. Default is logging.CRITICAL == 50
    """

    try:
        log_level_str = os.environ["AUTODE_LOG_LEVEL"]
    except KeyError:
        log_level_str = ""

    if log_level_str == "DEBUG":
        return logging.DEBUG

    if log_level_str == "WARNING":
        return logging.WARNING

    if log_level_str == "INFO":
        return logging.INFO

    if log_level_str == "ERROR":
        return logging.ERROR

    return logging.CRITICAL


def log_to_log_file():
    """
    Should the log be piped into a file? Looks for $AUTODE_LOG_FILE being
    set and writes logs to that file. Also possible to pipe directly from
    stderr to a file with

    .. code-block:
        python script.py 2> ade.log

    ---------------------------------------------------------------------------
    Returns:
        (bool):
    """

    try:
        _ = os.environ["AUTODE_LOG_FILE"]
        return True
    except KeyError:
        return False


if log_to_log_file():
    logging.basicConfig(
        level=get_log_level(),
        filename=os.environ["AUTODE_LOG_FILE"],
        filemode="w",
        format="%(asctime)s %(name)-12s: %(levelname)-8s " "%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

else:
    logging.basicConfig(
        level=get_log_level(),
        format="%(name)-12s: %(levelname)-8s %(message)s",
    )
logger = logging.getLogger(__name__)

# Try and use colourful logs...
try:
    import coloredlogs

    coloredlogs.install(level=get_log_level(), logger=logger)
except ImportError:
    pass
