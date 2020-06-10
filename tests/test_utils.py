from autode import utils
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
