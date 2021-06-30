import os
import shutil
from zipfile import ZipFile
from functools import wraps


def unzip_dir(zip_path):
    return work_in_zipped_dir(zip_path, chdir=False)


def work_in_zipped_dir(zip_path, chdir=True, fix_unique=False):
    """Extract some data from a compressed folder, change directories to it if
    required, run the function then, if required change directories back out
    and then delete the generated folder"""
    assert zip_path.endswith('.zip')

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Remove the .zip extension - rstrip doesn't seem to work
            # consistently(?)
            dir_path = zip_path[:-4]

            extract_path = os.path.split(dir_path)[0]
            here = os.getcwd()

            with ZipFile(zip_path, 'r') as zip_folder:
                zip_folder.extractall(extract_path)

            if chdir:
                os.chdir(dir_path)
            if fix_unique:
                os.environ['AUTODE_FIXUNIQUE'] = 'True'

            try:
                result = func(*args, **kwargs)

            finally:
                if chdir:
                    os.chdir(here)
                if fix_unique:
                    os.environ['AUTODE_FIXUNIQUE'] = 'False'

                shutil.rmtree(dir_path)

            return result

        return wrapped_function
    return func_decorator
