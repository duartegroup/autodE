from functools import wraps
import os
import shutil
from subprocess import Popen, DEVNULL, PIPE, STDOUT
from tempfile import mkdtemp
import multiprocessing
import multiprocessing.pool
from autode.exceptions import NoAtomsInMolecule
from autode.exceptions import NoCalculationOutput
from autode.exceptions import NoConformers
from autode.exceptions import NoMolecularGraph
from autode.log import logger


def run_external(params, output_filename):
    """
    Standard method to run a EST calculation with subprocess writing the
    output to the calculation output filename

    Arguments:
        output_filename (str):
        params (list(str)): e.g. [/path/to/method, input-filename]
    """

    with open(output_filename, 'w') as output_file:
        # /path/to/method input_filename > output_filename
        process = Popen(params, stdout=output_file, stderr=DEVNULL)
        process.wait()

    return None


def run_external_monitored(params, output_filename, break_word='MPI_ABORT',
                           break_words=None):
    """
    Run an external process monitoring the standard output and error for a
    word that will terminate the process

    Arguments:
        params (list(str)):
        output_filename (str):

    Keyword Arguments:
        break_word (str): String that if found will terminate the process
        break_words (list(str) | None): List of break_word-s
    """
    # Defining a set will override a single break word
    break_words = [break_word] if break_words is None else break_words

    def output_reader(process, out_file):
        for line in process.stdout:
            if any(string in line.decode('utf-8') for string in break_words):
                raise ChildProcessError

            print(line.decode('utf-8'), end='', file=out_file)

        return None

    with open(output_filename, 'w') as output_file:

        proc = Popen(params, stdout=PIPE, stderr=STDOUT)

        try:
            output_reader(proc, output_file)

        except ChildProcessError:
            logger.warning('External terminated')
            proc.terminate()
            return

    return None


def work_in(dir_ext):
    """Execute a function in a different directory"""

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):

            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext)

            if not os.path.isdir(dir_path):
                logger.info(f'Creating directory to store files: {dir_path:}')
                os.mkdir(dir_path)

            os.chdir(dir_path)
            result = func(*args, **kwargs)
            os.chdir(here)

            if len(os.listdir(dir_path)) == 0:
                logger.warning(f'Worked in {dir_path} but made no files '
                               f'- deleting')
                os.rmdir(dir_path)

            return result

        return wrapped_function
    return func_decorator


def work_in_tmp_dir(filenames_to_copy, kept_file_exts, use_ll_tmp=False):
    """Execute a function in a temporary directory.

    Arguments:
        filenames_to_copy (list(str)): Filenames to copy to the temp dir

        kept_file_exts (list(str): Filename extensions to copy back from
                       the temp dir
    """
    from autode.config import Config

    def func_decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            base_dir = Config.ll_tmp_dir if use_ll_tmp else None

            if base_dir is not None:
                assert os.path.exists(base_dir)

            tmpdir_path = mkdtemp(dir=base_dir)
            logger.info(f'Creating tmpdir to work in: {tmpdir_path}')

            logger.info(f'Copying {filenames_to_copy}')
            for filename in filenames_to_copy:
                if filename.endswith('_mol.in'):
                    # MOPAC needs the file to be called this
                    shutil.move(filename, os.path.join(tmpdir_path, 'mol.in'))
                else:
                    shutil.copy(filename, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)

            logger.info('Function   ...running')
            result = func(*args, **kwargs)
            logger.info('           ...done')

            for filename in os.listdir(tmpdir_path):

                if any([filename.endswith(ext) for ext in kept_file_exts]):
                    logger.info(f'Copying back {filename}')
                    shutil.copy(filename, here)

            os.chdir(here)

            logger.info('Removing temporary directory')
            shutil.rmtree(tmpdir_path)
            return result

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


def requires_conformers():
    """A function requiring the species to have a list of conformers"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):

            # Species must be the first argument
            assert hasattr(args[0], 'conformers')

            if args[0].conformers is None:
                raise NoConformers

            return func(*args, **kwargs)

        return wrapped_function
    return func_decorator


def requires_output():
    """A function requiring an output file and output file lines"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            # Calculation must be the first argument
            assert hasattr(args[0], 'output')

            if args[0].output.file_lines is None:
                raise NoCalculationOutput

            return func(*args, **kwargs)

        return wrapped_function

    return func_decorator


def timeout(seconds, return_value=None):
    """
    Function dectorator that times-out after a number of seconds

    Arguments:
        seconds (float):

    Keyword Arguments:
        return_value (Any): Value returned if the function times out

    Returns:
        (Any): Result of the function | return_value
    """
    def handler(queue, func, args, kwargs):
        queue.put(func(*args, **kwargs))

    def decorator(func):

        def wraps(*args, **kwargs):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(target=handler,
                                        args=(q, func, args, kwargs))
            p.start()
            p.join(timeout=seconds)
            if p.is_alive():
                p.terminate()
                p.join()
                return return_value

            else:
                return q.get()

        return wraps

    return decorator


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
    """Subclass of Pool to allow child multiprocessing"""

    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super().__init__(*args, **kwargs)

