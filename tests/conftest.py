import os
import autode as ade
import pytest


@pytest.fixture(scope="function", autouse=True)
def autouse_fixture():
    """Fixture to execute before and after a test is run"""

    # For ORCA/Gaussian etc. calculations to be skipped there needs to be no
    # attempt to make calculation names unique if they have a different input,
    # so set the appropriate flag
    os.environ["AUTODE_FIXUNIQUE"] = "False"

    # Run all the tests on a single core
    ade.Config.n_cores = 1

    # Frequencies are all with unity scaling
    ade.Config.freq_scale_factor = 1.0

    yield  # test happens here

    # Teardown
