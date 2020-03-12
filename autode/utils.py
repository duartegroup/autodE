import os
from autode.log import logger
from autode.exceptions import NoAtomsInMolecule
from autode.exceptions import NoMolecularGraph
from autode.exceptions import NoCalculationOutput
from functools import wraps


def work_in(dir_ext):
    """Execute a function in a different directory"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):

            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext)

            if not os.path.isdir(dir_path):
                logger.info(
                    f'Creating directory to store output files at {dir_path:}')
                os.mkdir(dir_path)

            os.chdir(dir_path)
            func(*args, **kwargs)
            os.chdir(here)

        return wrapped_function
    return func_decorator


def requires_atoms():
    """A function requiring a number of atoms to run"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Species must be the first argument
            assert hasattr(args[0], 'n_atoms')

            if args[0].n_atoms == 0:
                raise NoAtomsInMolecule

            return func(*args, **kwargs)

        return wrapped_function
    return func_decorator


def requires_graph():
    """A function requiring a number of atoms to run"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Species must be the first argument
            assert hasattr(args[0], 'graph')

            if args[0].graph is None:
                raise NoMolecularGraph

            return func(*args, **kwargs)

        return wrapped_function
    return func_decorator


def requires_output():
    """A function requiring an output file and output file lines"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            # Species must be the first argument
            assert hasattr(args[0], 'output_filename')
            assert hasattr(args[0], 'output_file_exists')
            assert hasattr(args[0], 'output_file_lines')
            assert hasattr(args[0], 'rev_output_file_lines')

            if args[0].output_file_exists is False or args[0].output_file_lines is None:
                raise NoCalculationOutput

            return func(*args, **kwargs)

        return wrapped_function

    return func_decorator
