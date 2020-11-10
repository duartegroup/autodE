from autode import utils
from autode.calculation import Calculation
from autode.species.molecule import Molecule
from autode.conformers import Conformer
from autode.wrappers.MOPAC import MOPAC
from autode.wrappers.keywords import Keywords
from autode.exceptions import NoCalculationOutput
from autode.exceptions import NoConformers
import pytest
import os


def test_clear_files():

    @utils.work_in('test')
    def make_test_files():
        open('aaa.tmp', 'a').close()

    make_test_files()
    assert os.path.exists('test/aaa.tmp')
    os.remove('test/aaa.tmp')
    os.rmdir('test')


def test_monitored_external():

    echo = ['echo', 'test']
    utils.run_external_monitored(params=echo,
                                 output_filename='test.txt')

    assert os.path.exists('test.txt')
    assert 'test' in open('test.txt', 'r').readline()
    os.remove('test.txt')

    # If the break word is in the stdout or stderr then the process should exit
    echo = ['echo', 'ABORT\ntest']
    utils.run_external_monitored(params=echo,
                                 output_filename='test.txt',
                                 break_word='ABORT')

    assert len(open('test.txt', 'r').readline()) == 0
    os.remove('test.txt')


def test_work_in_temp_dir():

    # Make a test python file echoing 'test' and printing a .dat file
    with open('echo_test.py', 'w') as test_file:
        print('print("test")', file=test_file)
        print('other_file = open("other_file.dat", "w")\n'
              'other_file.write("test")\n'
              'other_file.close',
              file=test_file)

    # Working in a temp directory running an external command
    @utils.work_in_tmp_dir(filenames_to_copy=['echo_test.py'],
                           kept_file_exts=['.txt'])
    def test():
        params = ['python', 'echo_test.py']
        utils.run_external(params=params, output_filename='test.txt')

    # Call the decorated function
    test()

    # Decorator should only copy back the .txt file back, and not the .dat
    assert os.path.exists('test.txt')
    assert not os.path.exists('other_file.dat')

    os.remove('echo_test.py')
    os.remove('test.txt')


def test_calc_output():

    calc = Calculation(name='test',
                       molecule=Molecule(smiles='C'),
                       method=MOPAC(),
                       keywords=Keywords(['PM7']))

    # A function that ficticously requires output
    @utils.requires_output()
    def test(calculation):
        print(calculation.molecule.n_atoms)

    with pytest.raises(NoCalculationOutput):
        test(calc)

    # Calling the same function with some calculation output should not raise
    # a not calculation output error
    calc.output.file_lines = ['some', 'example', 'output']
    test(calc)


def test_conformers():

    methane = Molecule(name='methane', smiles='C')

    # Function requiring a molecule having a conformer attribute
    @utils.requires_conformers()
    def test(mol):
        print(mol.conformers[0].n_atoms)

    with pytest.raises(NoConformers):
        test(methane)

    # Populating a species conformers should allow this function to be called
    methane.conformers = [Conformer(name='conf0', atoms=methane.atoms)]
    test(methane)


def test_work_in_empty():

    @utils.work_in('tmp_dir')
    def test_function():
        # Function makes no files so the directory should be deleted
        print('test')

    test_function()
    assert not os.path.exists('tmp_dir')

    @utils.work_in('tmp_dir')
    def test_function_files():
        # Function makes  files so the directory should be deleted
        with open('tmp.txt', 'w') as out_file:
            print('test', file=out_file)

    test_function_files()
    # Directory should now be retained
    assert os.path.exists('tmp_dir')

    # Remove the created files and directory
    os.remove('tmp_dir/tmp.txt')
    os.rmdir('tmp_dir')
