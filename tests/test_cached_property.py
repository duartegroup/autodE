from autode.utils import cached_property
from time import time, sleep


class Class:

    @cached_property
    def some_slow_function(self):
        sleep(1)
        return 'hello'


def test_cached_property():

    tmp = Class()
    start_time = time()
    _ = tmp.some_slow_function
    assert time() - start_time > 0.9  # executes in around a second

    # But further calls should be quick
    start_time = time()
    _ = tmp.some_slow_function
    assert time() - start_time < 0.9
