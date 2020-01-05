import os
from autode.log import logger
from functools import wraps


def work_in(dir_ext):
    """Execute a function in a different directory then clean up any temporary files"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):

            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext)

            if not os.path.isdir(dir_path):
                logger.info(f'Creating directory to store output files at {dir_path:}')
                os.mkdir(dir_path)

            os.chdir(dir_path)
            func(*args, **kwargs)
            clear_tmp_files()
            clear_xtb_files()
            os.chdir(here)

        return wrapped_function
    return func_decorator


def clear_xtb_files():
    xtb_files = ['xtbrestart', 'xtbopt.log', 'xtbopt.xyz',
                 'charges', 'wbo', '.xtboptok']
    if any(file in xtb_files for file in os.listdir(os.getcwd())):
        logger.info('Clearing xtb files')
    for filename in xtb_files:
        if os.path.exists(filename):
            os.remove(filename)


def clear_tmp_files():
    if any(file.endswith('.tmp') for file in os.listdir(os.getcwd())):
        logger.info('Clearing tmp files')
    for filename in os.listdir(os.getcwd()):
        if filename.endswith('.tmp'):
            os.remove(filename)
