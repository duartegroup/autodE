from autode import utils
import os


def test_clear_files():

    @utils.work_in('test')
    def make_test_files():
        open('aaa.tmp', 'a').close()
        open('xtbrestart', 'a').close()

    assert not os.path.exists('aaa.tmp')
    assert not os.path.exists('xtbrestart')
