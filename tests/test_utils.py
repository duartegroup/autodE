import shutil
import time
import pytest
import platform
import os
from autode import utils
from autode.calculations import Calculation
from autode.species.molecule import Molecule
from autode.conformers import Conformer
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.keywords import OptKeywords
from subprocess import Popen, TimeoutExpired
import multiprocessing as mp
from autode import exceptions as ex
from autode.mol_graphs import is_isomorphic
from autode.utils import (
    work_in_tmp_dir,
    log_time,
    requires_graph,
    ProcessPool,
    temporary_config,
)
from autode.wrappers.keywords.keywords import Functional
from autode.config import Config
from .testutils import requires_with_working_xtb_install


here = os.path.dirname(os.path.abspath(__file__))


def test_clear_files():
    @utils.work_in("test")
    def make_test_files():
        open("aaa.tmp", "a").close()

    make_test_files()
    assert os.path.exists("test/aaa.tmp")
    os.remove("test/aaa.tmp")
    os.rmdir("test")


def test_reset_dir_on_error():
    @utils.work_in("tmp_path")
    def raise_error():
        assert 0

    here = os.getcwd()
    try:
        raise_error()
    except AssertionError:
        pass

    assert here == os.getcwd()


def test_monitored_external():

    echo = ["echo", "test"]
    if platform.system() == "Windows":
        echo = ["cmd", "/c", *echo]  # echo is cmd prompt builtin

    utils.run_external_monitored(params=echo, output_filename="test.txt")

    assert os.path.exists("test.txt")
    assert "test" in open("test.txt", "r").readline()
    os.remove("test.txt")

    # If the break word is in the stdout or stderr then the process should exit
    echo = ["echo", "ABORT\ntest"]
    if platform.system() == "Windows":
        echo = ["cmd", "/c", *echo]

    utils.run_external_monitored(
        params=echo, output_filename="test.txt", break_word="ABORT"
    )

    assert len(open("test.txt", "r").readline()) == 0
    os.remove("test.txt")


def test_work_in_temp_dir():

    # Make a test python file echoing 'test' and printing a .dat file
    with open("echo_test.py", "w") as test_file:
        print('print("test")', file=test_file)
        print(
            'other_file = open("other_file.dat", "w")\n'
            'other_file.write("test")\n'
            "other_file.close",
            file=test_file,
        )

    # Working in a temp directory running an external command
    @utils.work_in_tmp_dir(
        filenames_to_copy=["echo_test.py"], kept_file_exts=[".txt"]
    )
    def test():
        params = ["python", "echo_test.py"]
        utils.run_external(params=params, output_filename="test.txt")

    # Call the decorated function
    test()

    # Decorator should only copy back the .txt file back, and not the .dat
    assert os.path.exists("test.txt")
    assert not os.path.exists("other_file.dat")

    os.remove("echo_test.py")
    os.remove("test.txt")


def test_reset_tmp_dir_on_error():
    @utils.work_in_tmp_dir()
    def raise_error():
        assert 0

    here = os.getcwd()
    try:
        raise_error()
    except AssertionError:
        pass

    assert here == os.getcwd()


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_calc_output():

    calc = Calculation(
        name="test",
        molecule=Molecule(smiles="C"),
        method=MOPAC(),
        keywords=OptKeywords(["PM7"]),
    )

    # A function that ficticously requires output
    @utils.requires_output
    def test(calculation):
        print(calculation.molecule.n_atoms)

    with pytest.raises(ex.NoCalculationOutput):
        test(calc)

    # Calling the same function with some calculation output should not raise
    # a not calculation output error
    calc.output.filename = "tmp.out"
    with open(calc.output.filename, "w") as out_file:
        print("some", "example", "output", sep="\n", file=out_file)
    test(calc)


def test_conformers():

    methane = Molecule(name="methane", smiles="C")

    # Function requiring a molecule having a conformer attribute
    @utils.requires_conformers
    def test(mol):
        print(mol.conformers[0].n_atoms)

    with pytest.raises(ex.NoConformers):
        test(methane)

    # Populating a species conformers should allow this function to be called
    methane.conformers = [Conformer(name="conf0", atoms=methane.atoms)]
    test(methane)


def test_work_in_empty():
    @utils.work_in("tmp_dir")
    def test_function():
        # Function makes no files so the directory should be deleted
        print("test")

    test_function()
    assert not os.path.exists("tmp_dir")

    @utils.work_in("tmp_dir")
    def test_function_files():
        # Function makes  files so the directory should be deleted
        with open("tmp.txt", "w") as out_file:
            print("test", file=out_file)

    test_function_files()
    # Directory should now be retained
    assert os.path.exists("tmp_dir")

    # Remove the created files and directory
    os.remove("tmp_dir/tmp.txt")
    os.rmdir("tmp_dir")


def test_work_in_compatible_with_experimental_timeout():
    if platform.system() != "Windows":
        return
    # cleanup should take care of all processes unused if the
    # function finished within time limit

    @utils.timeout(seconds=3)
    def sleep_2s():
        return time.sleep(2)

    @utils.work_in("tmp_dir")
    def use_exp_timeout():
        sleep_2s()

    with temporary_config():
        Config.use_experimental_timeout = True
        use_exp_timeout()


def test_cleanup_after_timeout():
    if platform.system() != "Windows":
        return
    # cleanup should take care of all processes unused

    @utils.timeout(seconds=3)
    def sleep_2s():
        return time.sleep(2)

    with temporary_config():
        Config.use_experimental_timeout = True
        sleep_2s()
        assert len(mp.active_children()) != 0
        utils.cleanup_after_timeout()
        assert len(mp.active_children()) == 0


def test_timeout():

    if platform.system() == "Windows":
        Config.use_experimental_timeout = True

    def sleep_2s():
        return time.sleep(2)

    start_time = time.time()
    sleep_2s()
    assert time.time() - start_time > 1.9

    @utils.timeout(seconds=1)
    def sleep_2s():
        return time.sleep(2)

    # Decorated function should timeout and return in under two seconds
    start_time = time.time()
    sleep_2s()
    assert time.time() - start_time < 2

    @utils.timeout(seconds=10)
    def return_string():
        return "test"

    # Should not raise a TimeoutError if the function executes fast
    start_time = time.time()
    assert return_string() == "test"
    assert time.time() - start_time < 10

    if platform.system() == "Windows":
        Config.use_experimental_timeout = False


def test_repeated_timeout_win_loky():
    if platform.system() != "Windows":
        return None
    # With experimental timeout, triggering timeout
    # repeatedly should not cause deadlocks or hangs
    # from executor manager thread

    @utils.timeout(seconds=1)
    def sleep_2s():
        return time.sleep(2)

    with temporary_config():
        Config.use_experimental_timeout = True
        start_time = time.time()
        sleep_2s()
        sleep_2s()
        sleep_2s()
        assert time.time() - start_time < 6


@work_in_tmp_dir(filenames_to_copy=[], kept_file_exts=[])
def test_spawn_multiprocessing_posix():

    if platform.system() == "Windows":
        return None

    with open("tmp.py", "w") as py_file:
        print(
            "import multiprocessing as mp",
            'mp.set_start_method("spawn", force=True)',
            "import autode as ade",
            "def mol():",
            '    return ade.Molecule(atoms=[ade.Atom("H"), ade.Atom("H", x=0.7)])',
            'if __name__ == "__main__":',
            "    with mp.Pool(2) as pool:",
            "        res = [pool.apply_async(mol) for _ in range(2)]",
            "        mols = [r.get() for r in res]",
            sep="\n",
            file=py_file,
        )

    process = Popen(["python", "tmp.py"])

    # Executing the script should not take more than a second, if the function
    # hangs then it should timeout after 10s
    try:
        process.wait(timeout=10)
    except TimeoutExpired:
        raise AssertionError

    os.remove("tmp.py")


def test_spawn_multiprocessing_graph_posix():
    # Test spawn method for POSIX only
    if platform.system() == "Windows":
        return None

    mp.set_start_method("spawn", force=True)

    # Isomorphism should still be able to be checked
    h2o_a, h2o_b = Molecule(smiles="O"), Molecule(smiles="O")
    assert is_isomorphic(h2o_a.graph, h2o_b.graph)

    mp.set_start_method("fork", force=True)


def test_spawn_loky_graph_win():

    if platform.system() != "Windows":
        return

    import loky

    loky.backend.context.set_start_method("spawn", force=True)

    # Isomorphism should still be able to be checked
    h2o_a, h2o_b = Molecule(smiles="O"), Molecule(smiles="O")
    assert is_isomorphic(h2o_a.graph, h2o_b.graph)

    loky.backend.context.set_start_method("loky", force=True)


def test_config_in_worker_proc():
    # check that the config is able to be passed to child processes
    # mainly for windows, but still nice to check for posix

    with temporary_config():
        Config.n_cores = 9
        Config.ORCA.keywords.sp.functional = "B3LYP"
        with ProcessPool(max_workers=2) as pool:
            job = pool.submit(worker_fn)
            _ = job.result()


def worker_fn():
    assert Config.n_cores == 9
    assert Config.ORCA.keywords.sp.functional == Functional("B3LYP")


def test_temporary_config_context_manager():
    old_n_cores = Config.n_cores
    old_orca_funct = Config.ORCA.keywords.sp.functional

    with temporary_config():
        Config.n_cores = 9
        Config.ORCA.keywords.sp.functional = "B3LYP"
        # test the values have been changed in external function
        worker_fn()

    # Config should be restored after exit
    assert Config.n_cores == old_n_cores
    assert Config.ORCA.keywords.sp.functional == old_orca_funct


def test_temporary_config_in_worker_proc():
    # check that the context manager works if workers
    # are created inside the context manager
    old_n_cores = Config.n_cores
    old_orca_funct = Config.ORCA.keywords.sp.functional

    with temporary_config():
        Config.n_cores = 9
        Config.ORCA.keywords.sp.functional = "B3LYP"
        with ProcessPool(max_workers=2) as pool:
            job = pool.submit(worker_fn)
            _ = job.result()

    assert Config.n_cores == old_n_cores
    assert Config.ORCA.keywords.sp.functional == old_orca_funct


def test_time_units():

    with pytest.raises(ValueError):
        log_time(units="X")  # invalid time format


def test_requires_graph():
    @requires_graph
    def f(mol):
        return mol.graph.number_of_edges()

    with pytest.raises(Exception):
        m = Molecule()
        m.graph = None

        f(m)  # No graph


def test_tmp_env():

    os.environ["OMP_NUM_THREADS"] = "1"

    @utils.run_in_tmp_environment(
        tmp_key_str="tmp_value", tmp_key_int=1, OMP_NUM_THREADS=9999
    )
    def f():
        assert os.environ["tmp_key_int"] == "1"
        assert os.environ["tmp_key_str"] == "tmp_value"
        assert os.environ["OMP_NUM_THREADS"] == "9999"

    f()
    assert "tmp_key_int" not in os.environ
    assert "tmp_key_str" not in os.environ
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_string_dict():

    d = utils.StringDict("a = b solvent = water")
    assert "a" in str(d)
    assert "solvent" in str(d)
    assert d["a"] == "b"


def test_requires_xtb_install():

    path_env_var = os.environ.pop("PATH")
    assert shutil.which("xtb") is None

    test_list = []

    @requires_with_working_xtb_install
    def tmp_function():
        test_list.append("executed")

    # If XTB is not in $PATH then the function should not execute
    assert len(test_list) == 0
    os.environ["PATH"] = path_env_var
