"""Check that gamma command arguments precede filesystem and network operations."""

from pathlib import Path

import pytest

from artisatomic import download_gammaspec_betaminus_alpha as gamma


@pytest.mark.parametrize(("argument", "status"), [("--help", 0), ("--unknown", 2)])
def test_arguments_stop_before_conversion(monkeypatch, capsys, argument, status):
    """Help and invalid arguments must leave the data and output files unchanged."""

    def fail_operation(*_args: object, **_kwargs: object):
        msg = "The argument check must precede conversion."
        raise AssertionError(msg)

    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles", argument])
    monkeypatch.setattr(Path, "mkdir", fail_operation)
    monkeypatch.setattr(gamma.at, "get_path", fail_operation)
    monkeypatch.setattr(gamma.requests, "Session", fail_operation)
    with pytest.raises(SystemExit) as error:
        gamma.main()
    assert error.value.code == status
    captured = capsys.readouterr()
    assert "usage:" in captured.out + captured.err
