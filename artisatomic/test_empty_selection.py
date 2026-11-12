"""Check selections that contain elements with no ions."""

import json

import pytest

from artisatomic import cli


@pytest.mark.parametrize("selection", [[], [[26, []]], [[26, []], [27, []]]])
def test_reject_selection_without_ions(tmp_path, monkeypatch, selection):
    """The command rejects an empty ion selection before it creates output files."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artisatomicionhandlers.json").write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles"])
    with pytest.raises(ValueError, match="No ions selected"):
        cli.main()
    assert not (tmp_path / "artis_files").exists()


def test_keep_empty_element_beside_selected_ion(tmp_path, monkeypatch):
    """An empty element remains valid when another element has an ion."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artisatomicionhandlers.json").write_text(
        json.dumps([[26, []], [34, [[5, "gsnist"]]]]), encoding="utf-8"
    )
    monkeypatch.setattr("sys.argv", ["makeartisatomicfiles", "--nophixs"])
    cli.main()
    rows = (tmp_path / "artis_files" / "compositiondata.txt").read_text(encoding="utf-8").splitlines()
    assert rows[3].split()[:4] == ["26", "0", "0", "0"]
    assert rows[4].split()[:4] == ["34", "1", "5", "5"]
