"""Check the minimum point count before the command reads atomic data."""

import pytest

from artisatomic import cli


def test_reject_one_point_before_conversion(tmp_path, monkeypatch, capsys):
    """One point gives identical integration limits in ARTIS, so the command must reject it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "-nphixspoints", "1"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "-nphixspoints must be 2 or more" in capsys.readouterr().err
    assert not (tmp_path / "artis_files").exists()
