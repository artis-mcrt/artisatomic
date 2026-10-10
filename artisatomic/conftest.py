"""pytest settings for the tests of artisatomic."""

import pytest


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Fail the test run if a test gave a warning, so that CI shows a polars deprecation.

    A filter that turns a warning into an error does not work for a warning from the Rust core of
    polars. polars then prints the error and continues, and pytest shows no warning. So the hook
    counts the warnings at the end of the run.
    """
    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is not None and terminalreporter.stats.get("warnings") and session.exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
