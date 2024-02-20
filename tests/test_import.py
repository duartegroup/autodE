"""Tests for the import speed of autode."""
import sys

import pytest

SLOW_IMPORTS = ["matplotlib"]


@pytest.fixture
def unimport_slow_imports():
    """Remove modules in ``SLOW_IMPORTS`` from ``sys.modules``."""
    for module in SLOW_IMPORTS:
        if module in sys.modules:
            del sys.modules[module]


@pytest.mark.usefixtures("unimport_slow_imports")
def test_slow_imports_during_tab_completion():
    """Check that importing autode does not import certain python modules that would make import slow."""

    # Let's double check that the undesired imports are not already loaded
    for modulename in SLOW_IMPORTS:
        assert (
            modulename not in sys.modules
        ), f"Module `{modulename}` was not properly unloaded"

    import autode

    for modulename in SLOW_IMPORTS:
        assert (
            modulename not in sys.modules
        ), f"Detected loaded module {modulename} after autode import"
