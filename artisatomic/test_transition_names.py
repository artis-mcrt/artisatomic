"""Check that each level name of a transition matches one level."""

import polars as pl
import pytest

from artisatomic.output import add_level_ids_forbidden


@pytest.mark.parametrize("side", ["namefrom", "nameto"])
def test_reject_missing_name_beside_shared_name(side):
    """A name with no level and a name with two levels keep the row count, and the function stops."""
    levels = pl.DataFrame({"levelid": [0, 1, 2], "levelname": ["a", "b", "b"], "parity": [0, 1, 1]})
    columns = {"namefrom": ["a", "a"], "nameto": ["a", "a"], "A": [1.0, 2.0]}
    columns[side] = ["b", "missing"]
    with pytest.raises(
        ValueError, match=rf"1 transitions have a {side} that matches no level\. The first is 'missing'"
    ):
        add_level_ids_forbidden(levels, pl.DataFrame(columns))


def test_allow_shared_name_that_no_transition_uses():
    """Two levels can share a name if no transition has that name."""
    levels = pl.DataFrame({"levelid": [0, 1, 2, 3], "levelname": ["a", "b", "c", "c"], "parity": [0, 1, 0, 0]})
    transitions = pl.DataFrame({"namefrom": ["a"], "nameto": ["b"], "A": [1.0]})
    result = add_level_ids_forbidden(levels, transitions)
    assert result.select("lowerlevel", "upperlevel").row(0) == (0, 1)
