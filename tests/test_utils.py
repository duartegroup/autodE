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
