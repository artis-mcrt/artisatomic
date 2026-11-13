"""Check each source rate before the Floers reader adds contributions."""

import pytest

from artisatomic.readfloers25data import read_transitions_file


@pytest.mark.parametrize("bad_rate", ["-1", "nan", "inf", "-inf"])
def test_reject_invalid_contribution(tmp_path, bad_rate):
    """A positive contribution must not conceal an invalid contribution to the same transition."""
    path = tmp_path / "transitions.txt"
    path.write_text(
        f"Test table\n--\n--\n--\n Lower Upper A Type\n 0 1 2 E1\n 0 1 {bad_rate} M1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"data row 2.*transitions.txt.*finite and non-negative"):
        read_transitions_file(path)


def test_keep_zero_contribution(tmp_path):
    """A zero source rate remains valid."""
    path = tmp_path / "transitions.txt"
    path.write_text("Test table\n--\n--\n--\n Lower Upper A Type\n 0 1 0 E1\n", encoding="utf-8")
    transitions, _ = read_transitions_file(path)
    assert transitions["A"].to_list() == [0.0]
