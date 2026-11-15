"""Check the argument and decay table checks of the gamma command, which precede each network request."""

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


@pytest.mark.parametrize(
    ("duplicated", "table"), [("betaminusdecays.txt", "beta minus table"), ("alphadecays.txt", "alpha table")]
)
def test_a_nuclide_with_two_rows_stops_the_command(monkeypatch, tmp_path, duplicated, table):
    """A decay table with two rows for one nuclide stops the command before a network request."""
    rows = {"betaminusdecays.txt": "44 21 1.0 0.5 0.3 0.2 100.0\n", "alphadecays.txt": "210 84 1 0 2.0 5 0 5 0 0\n"}
    rows[duplicated] *= 2
    for filename, text in rows.items():
        (tmp_path / filename).write_text(text, encoding="utf-8")

    def fail_request(*_args: object, **_kwargs: object):
        msg = "The table check must precede the network request."
        raise AssertionError(msg)

    monkeypatch.setattr("sys.argv", ["makeartisgammaspecfiles"])
    monkeypatch.setattr(gamma, "PYDIR", tmp_path / "artisatomic")
    monkeypatch.setattr(gamma.at, "get_path", lambda _name: tmp_path)
    monkeypatch.setattr(gamma.requests, "Session", fail_request)
    with pytest.raises(ValueError, match=f"The {table} holds a nuclide more than one time"):
        gamma.main()
