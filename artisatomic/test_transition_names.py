"""Check each transition name independently of the total row count."""

import polars as pl
import pytest

from artisatomic.output import add_level_ids_forbidden


@pytest.mark.parametrize("side", ["namefrom", "nameto"])
def test_reject_cancelled_match_counts(side):
    """A missing match and a duplicate match must not cancel."""
    levels = pl.DataFrame({"levelid": [0, 1, 2], "levelname": ["a", "b", "b"], "parity": [0, 1, 1]})
    columns = {"namefrom": ["a", "a"], "nameto": ["a", "a"], "A": [1.0, 2.0]}
    columns[side] = ["b", "missing"]
    transitions = pl.DataFrame(columns)
    with pytest.raises(ValueError, match=f"{side} value.*must match one level"):
        add_level_ids_forbidden(levels, transitions)


def test_allow_duplicate_names_without_name_references():
    """A duplicate name does not affect a transition that uses level ids."""
    levels = pl.DataFrame({"levelid": [0, 1, 2], "levelname": ["a", "b", "b"], "parity": [0, 1, 1]})
    transitions = pl.DataFrame({"lowerlevel": [0], "upperlevel": [2], "A": [1.0]})
    result = add_level_ids_forbidden(levels, transitions)
    assert result["upperlevel"].to_list() == [2]
