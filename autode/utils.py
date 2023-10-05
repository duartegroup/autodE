import os
import sys
import platform
import shutil
import copy
import signal
import warnings
import contextlib
from time import time
from typing import Any, Optional, Sequence, List, Callable, TYPE_CHECKING
from functools import wraps
from subprocess import Popen, PIPE, STDOUT
from tempfile import mkdtemp
import multiprocessing

from autode.config import Config
from autode.log import logger
from autode.values import Allocation
from autode.exceptions import (
    AutodeException,
    NoAtomsInMolecule,
    NoCalculationOutput,
    NoConformers,
    NoMolecularGraph,
    MethodUnavailable,
    CouldNotGetProperty,
)

if TYPE_CHECKING:
    from autode.reactions.reaction import Reaction
    from autode.config import _ConfigClass


@contextlib.contextmanager
def temporary_config():
    """
    Context manager to temporarily change autodE's Config. When it
    exits, the Config will be restored to whatever it was before
    calling the context manager.

    Example usage:

    .. code-block:: Python

    >>> import autode as ade
    >>> from autode import Config
    >>> Config.hcode = 'ORCA'
    >>> with ade.temporary_config():
    ...     # change some config vars
    ...     Config.n_cores = 16
    ...     Config.ORCA.keywords.sp.functional = 'B3LYP'
    ...     # then do some calculations
    ...     # ------
    >>> # When context manager returns, Config should be restored to what it was before
    >>> assert Config.n_cores == 4
    >>> assert str(Config.ORCA.keywords.sp.functional).lower() == 'pbe0'

    """
    original_config_data = copy.deepcopy(Config.__dict__)

    try:
        yield
    finally:
        Config.__dict__.update(original_config_data)

    return None


def _copy_into_current_config(
    parent_config: "_ConfigClass",
) -> None:
    """
    Copies an instance of Config into current process. Required to set the
    process pool workers' Config to the same state as the parent, when
    not forking the interpreter. To be only run on initializing workers

    Args:
        parent_config: Parent config instance that will be copied into
         present process
    """
    Config.__dict__.update(parent_config.__dict__)


def get_total_memory() -> int:
    """Returns total amount of physical memory available in bytes"""
    if sys.platform == "win32" or platform.system() == "Windows":
        return _get_total_memory_on_windows()
    else:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def _get_total_memory_on_windows() -> int:
    """Use WinAPI to get total memory on Windows machines"""
    from ctypes import Structure, c_int32, c_uint64, sizeof, byref, windll  # type: ignore

    # Use Win32 API : https://stackoverflow.com/questions/31546309/
    class MemoryStatusEx(Structure):
        _fields_ = [
            ("length", c_int32),
            ("memoryLoad", c_int32),
            ("totalPhys", c_uint64),
            ("availPhys", c_uint64),
            ("totalPageFile", c_uint64),
            ("availPageFile", c_uint64),
            ("totalVirtual", c_uint64),
            ("availVirtual", c_uint64),
            ("availExtendedVirtual", c_uint64),
        ]

        def __init__(self):
            super().__init__()
            self.length = sizeof(self)

    win_mem = MemoryStatusEx()
    if windll.kernel32.GlobalMemoryStatusEx(byref(win_mem)):
        return int(win_mem.totalPhys)
    else:
        raise OSError


def check_sufficient_memory(func: Callable):
    """Decorator to check that enough memory is available for a calculation"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        physical_mem = None
        required_mem = int(Config.n_cores) * Config.max_core

        try:
            physical_mem = Allocation(get_total_memory(), units="bytes")
        except (ValueError, OSError):
            logger.warning("Cannot check physical memory")

        if physical_mem is not None and physical_mem < required_mem:
            raise RuntimeError(
                "Cannot run function - insufficient memory. Had"
                f' {physical_mem.to("GB")} GB but required '
                f'{required_mem.to("GB")} GB'
            )

        return func(*args, **kwargs)

    return wrapped_function


@check_sufficient_memory
def run_external(
    params: List[str], output_filename: str, stderr_to_log: bool = True
):
    """
    Standard method to run a EST calculation with subprocess writing the
    output to the calculation output filename

    ---------------------------------------------------------------------------
    Arguments:
        params: e.g. [/path/to/method, input-filename]

        output_filename: Filename to output stdout to

        stderr_to_log: Should the stderr be added to the logged warnings?
    """

    with open(output_filename, "w") as output_file:
        # /path/to/method input_filename > output_filename
        process = Popen(params, stdout=output_file, stderr=PIPE)
        if process.stderr is not None:
            with process.stderr:
                for line in iter(process.stderr.readline, b""):
                    if stderr_to_log:
                        logger.warning("STDERR: %r", line.decode())

        process.wait()

    return None


@check_sufficient_memory
def run_external_monitored(
    params: Sequence[str],
    output_filename: str,
    break_word: str = "MPI_ABORT",
    break_words: Optional[List[str]] = None,
):
    """
    Run an external process monitoring the standard output and error for a
    word that will terminate the process

    ---------------------------------------------------------------------------
    Arguments:
        params (list(str)):

        output_filename (str):

        break_word (str): String that if found will terminate the process

        break_words (list(str) | None): List of break_word-s
    """
    # Defining a set will override a single break word
    break_words = [break_word] if break_words is None else break_words

    def output_reader(process, out_file):
        for line in process.stdout:
            if any(word in line.decode("utf-8") for word in break_words):
                raise ChildProcessError

            print(line.decode("utf-8"), end="", file=out_file)

        return None

    with open(output_filename, "w") as output_file:
        proc = Popen(params, stdout=PIPE, stderr=STDOUT)

        try:
            output_reader(proc, output_file)

        except ChildProcessError:
            logger.warning("External terminated")
            proc.terminate()
            return None

    return None


def work_in(dir_ext: str) -> Callable:
    """Execute a function in a different directory"""

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()
            dir_path = os.path.join(here, dir_ext)

            if not os.path.isdir(dir_path):
                logger.info(f"Creating directory to store files: {dir_path:}")
                os.mkdir(dir_path)

            os.chdir(dir_path)
            try:
                result = func(*args, **kwargs)
            finally:
                os.chdir(here)

                if len(os.listdir(dir_path)) == 0:
                    logger.warning(
                        f"Worked in {dir_path} but made no files "
                        f"- deleting"
                    )
                    cleanup_after_timeout()
                    os.rmdir(dir_path)

            return result

        return wrapped_function

    return func_decorator


def work_in_tmp_dir(
    filenames_to_copy: Optional[Sequence[str]] = None,
    kept_file_exts: Optional[Sequence[str]] = None,
    use_ll_tmp: bool = False,
) -> Callable:
    """Execute a function in a temporary directory.

    -----------------------------------------------------------------------
    Arguments:
        filenames_to_copy: Filenames to copy to the temp dir

        kept_file_exts: Filename extensions to copy back from the temp dir

        use_ll_tmp (bool): If true then use autode.config.Config.ll_tmp_dir
    """
    from autode.config import Config

    if filenames_to_copy is None:
        filenames_to_copy = []

    if kept_file_exts is None:
        kept_file_exts = []

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            here = os.getcwd()

            base_dir = Config.ll_tmp_dir if use_ll_tmp else None

            if base_dir is not None:
                assert os.path.exists(base_dir)

            tmpdir_path = mkdtemp(dir=base_dir)
            logger.info(f"Creating tmpdir to work in: {tmpdir_path}")

            if len(filenames_to_copy) > 0:
                logger.info(f"Copying {filenames_to_copy}")

            for filename in filenames_to_copy:
                if filename.endswith("_mol.in"):
                    # MOPAC needs the file to be called this
                    shutil.move(filename, os.path.join(tmpdir_path, "mol.in"))
                else:
                    shutil.copy(filename, tmpdir_path)

            # Move directories and execute
            os.chdir(tmpdir_path)

            try:
                logger.info("Function   ...running")
                result = func(*args, **kwargs)
                logger.info("           ...done")

                for filename in os.listdir(tmpdir_path):
                    if any([filename.endswith(ext) for ext in kept_file_exts]):
                        logger.info(f"Copying back {filename}")
                        shutil.copy(filename, here)

            finally:
                os.chdir(here)

                logger.info("Removing temporary directory")
                cleanup_after_timeout()
                shutil.rmtree(tmpdir_path)

            return result

        return wrapped_function

    return func_decorator


def log_time(prefix: str = "Executed in: ", units: str = "ms") -> Callable:
    """A function requiring a number of atoms to run"""

    if units.lower() == "s" or units.lower() == "seconds":
        s_to_units = 1.0

    elif units.lower() == "ms" or units.lower() == "milliseconds":
        s_to_units = 1000.0

    else:
        raise ValueError(f"Unsupported time unit: {units}")

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            start_time = time()

            result = func(*args, **kwargs)

            logger.info(
                f"{prefix} "
                f"{(time() - start_time) * s_to_units:.2f} {units}"
            )

            return result

        return wrapped_function

    return func_decorator


def requires_atoms(func: Callable) -> Callable:
    """A function requiring a number of atoms to run"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        # Species must be the first argument
        assert hasattr(args[0], "n_atoms")

        if args[0].n_atoms == 0:
            raise NoAtomsInMolecule

        return func(*args, **kwargs)

    return wrapped_function


def requires_graph(func: Callable) -> Callable:
    """A function requiring a number of atoms to run"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        # Species must be the first argument
        assert hasattr(args[0], "graph")

        if args[0].graph is None:
            raise NoMolecularGraph

        return func(*args, **kwargs)

    return wrapped_function


def requires_conformers(func: Callable) -> Callable:
    """A function requiring the species to have a list of conformers"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        # Species must be the first argument
        assert hasattr(args[0], "n_conformers")

        if args[0].n_conformers == 0:
            raise NoConformers

        return func(*args, **kwargs)

    return wrapped_function


def requires_hl_level_methods(func: Callable) -> Callable:
    """A function requiring both high and low-level methods to be available"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        from autode.methods import get_lmethod, get_hmethod

        suffix = "neither was available."

        try:
            _ = get_lmethod()

            # Have a low-level method, so the high-level must not be available
            suffix = "the high-level was not available."
            _ = get_hmethod()

        except MethodUnavailable:
            raise MethodUnavailable(
                f"Function *{func.__name__}* requires both"
                f" a high and low-level method but "
                f"{suffix}"
            )

        return func(*args, **kwargs)

    return wrapped_function


def requires_output(func: Callable) -> Callable:
    """A function requiring an output file and output file lines"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        # Calculation must be the first argument
        assert hasattr(args[0], "output")

        if args[0].output.file_lines is None:
            raise NoCalculationOutput

        return func(*args, **kwargs)

    return wrapped_function


def requires_output_to_exist(func: Callable) -> Callable:
    """Calculation method requiring the output filename to be set"""

    @wraps(func)
    def wrapped_function(*args, **kwargs):
        calc = args[0]

        if not calc.output.exists:
            raise CouldNotGetProperty(
                f"Could not get property from "
                f"{calc.name}. Has .run() been called?"
            )
        return func(*args, **kwargs)

    return wrapped_function


def no_exceptions(func) -> Any:
    """Calculation method requiring the output filename to be set"""

    @wraps(func)
    def wrapped_function(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except (ValueError, IndexError, TypeError, AutodeException):
            return None

    return wrapped_function


def _cleanup_after_exp_timeout_win() -> None:
    """
    If experimental timeout has been used, the ProcessPool has to be shutdown
    otherwise File permission errors will be caused on Windows
    """
    if Config.use_experimental_timeout:
        pool = loky.get_reusable_executor()
        pool.shutdown()


def _timeout_experimental(
    seconds: float, return_value: Optional[Any] = None
) -> Any:
    """
    Function decorator that times-out after a number of seconds, if
    Config.use_experimental_timeout = True, otherwise no timeout
    (experimental version, works on Windows)

    Args:
        seconds: The number of seconds to timeout
        return_value: Value returned if the function times out

    Returns:
        (Any): result of function | return_value
    """

    def func_runner(connector, func, args, kwargs):
        connector.send(os.getpid())
        return func(*args, **kwargs)

    def decorator(func):
        def wrapper(*args, **kwargs):
            # if user does not want it, there is no timeout
            if not Config.use_experimental_timeout:
                return func(*args, **kwargs)

            pool = loky.get_reusable_executor(max_workers=1)
            conn1, conn2 = multiprocessing.Pipe()
            job = pool.submit(func_runner, conn1, func, args, kwargs)
            job_pid = conn2.recv()
            try:
                res = job.result(timeout=seconds)
                return res
            except loky.TimeoutError:
                if os.getpid() != job_pid:
                    os.kill(job_pid, signal.SIGTERM)
                pool.shutdown(wait=True)
                return return_value

        return wrapper

    return decorator


def _timeout_default(
    seconds: float, return_value: Optional[Any] = None
) -> Any:
    """
    Function decorator that times-out after a number of seconds
    (default version, uses forking)

    Args:
        seconds: The number of seconds to timeout
        return_value: Value returned if the function times out

    Returns:
        (Any): result of function | return_value
    """

    def handler(queue, func, args, kwargs):
        queue.put(func(*args, **kwargs))

    def decorator(func):
        def wrapper(*args, **kwargs):
            q = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=handler, args=(q, func, args, kwargs)
            )

            if multiprocessing.current_process().daemon:
                # Cannot run a subprocess in a daemon process - timeout is not
                # possible
                return func(*args, **kwargs)

            elif isinstance(
                multiprocessing.get_context(),
                multiprocessing.context.ForkContext,
            ):
                p.start()

            else:
                logger.error("Failed to wrap function")
                return func(*args, **kwargs)

            p.join(timeout=seconds)

            if p.is_alive():
                p.kill()
                p.join()
                return return_value

            else:
                return q.get()

        return wrapper

    return decorator


def hashable(_method_name: str, _object: Any):
    """Multiprocessing requires hashable top-level functions to be executed,
    so convert a method into a top-level function"""
    return getattr(_object, _method_name)


def run_in_tmp_environment(**kwargs) -> Callable:
    """
    Apply a set of environment variables, execute a function and reset them
    """

    class EnvVar:
        def __init__(self, name, val):
            self.name = str(name)
            self.val = os.getenv(str(name), None)
            self.new_val = str(val)

    env_vars = [EnvVar(k, v) for k, v in kwargs.items()]

    def func_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **_kwargs):
            for env_var in env_vars:
                logger.info(f"Setting the {env_var.name} to {env_var.new_val}")
                os.environ[env_var.name] = env_var.new_val

            result = func(*args, **_kwargs)

            for env_var in env_vars:
                if env_var.val is None:
                    # Remove from the environment
                    os.environ.pop(env_var.name)
                else:
                    # otherwise set it back to the old value
                    os.environ[env_var.name] = env_var.val

            return result

        return wrapped_function

    return func_decorator


def deprecated(func: Callable) -> Callable:
    @wraps(func)
    def wrapped_function(*args, **kwargs):
        warnings.warn(
            "This function is deprecated and will be removed "
            "in autodE v1.5.0",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapped_function


def checkpoint_rxn_profile_step(name: str) -> Callable:
    """
    Decorator for a function that will save a checkpoint file with the reaction state
    at that point in time. If the checkpoint exists then the state will be reloaded
    and the execution skipped
    """

    def func_decorator(func: Callable[["Reaction"], Any]):
        @wraps(func)
        def wrapped_function(reaction: "Reaction"):
            filepath = os.path.join(
                "checkpoints", f"{str(reaction)}_{name}.chk"
            )
            if os.path.exists(filepath):
                reaction.load(filepath)
                return

            start_time = time()
            result = func(reaction)

            if (
                time() - start_time < 1.0
            ):  # If execution is < 1s don't checkpoint
                return result

            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints")

            reaction.save(filepath)

            return result

        return wrapped_function

    return func_decorator


class StringDict:
    r"""
    Immutable dictionary stored as a single string. For example::
        'a = b  c = d'
    """
    _value_type: type = str

    def __init__(self, string: str, delim: str = " = "):
        self._string = string
        self._delim = delim

    def __getitem__(self, item: str) -> Any:
        split_string = self._string.split(f"{item}{self._delim}")
        try:
            return self._value_type(split_string[1].split()[0])

        except (ValueError, IndexError) as e:
            raise IndexError(
                f"Failed to extract {item} from {self._string} "
                f"using delimiter *{self._delim}*"
            ) from e

    def __contains__(self, item: str) -> bool:
        split_string = self._string.split(f"{item}{self._delim}")
        return len(split_string) == 2

    def get(self, item: str, default: Any) -> Any:
        """Get an item or return a default"""

        try:
            return self[item]
        except IndexError:
            return default

    def __str__(self):
        return self._string


class NumericStringDict(StringDict):
    _value_type = float


if platform.system() == "Windows":
    # On Win64 or Win32, use loky
    import loky
    import loky.backend.resource_tracker

    try:
        loky.backend.context.set_start_method("loky")
    except RuntimeError:
        logger.warning("Loky context has already been set")
    # start the resource tracker early to fix file permission errors
    if isinstance(
        loky.backend.context.get_context(), loky.backend.context.LokyContext
    ):
        loky.backend.resource_tracker.ensure_running()

    class ProcessPool(loky.ProcessPoolExecutor):
        def __init__(
            self,
            max_workers=None,
            job_reducers=None,
            result_reducers=None,
            timeout=None,
            context=None,
            env=None,
        ):
            super().__init__(
                max_workers=max_workers,
                job_reducers=job_reducers,
                result_reducers=result_reducers,
                timeout=timeout,
                context=context,
                initializer=_copy_into_current_config,
                initargs=(Config,),
                env=env,
            )

    timeout = _timeout_experimental
    cleanup_after_timeout = _cleanup_after_exp_timeout_win

else:
    # On Linux, macOS, use concurrent futures, which has multiprocessing backend
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        logger.warning("Multiprocessing context has already been set")

    from concurrent.futures import ProcessPoolExecutor

    ProcessPool = ProcessPoolExecutor  # type: ignore
    timeout = _timeout_default

    def cleanup_after_timeout():
        pass
