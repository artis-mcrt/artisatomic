"""Check source values before the cross section reduction can conceal an invalid value."""

import numpy as np
import pytest

from artisatomic.base import output_xgrid
from artisatomic.phixs import reduce_phixs_tables_worker


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("energies", [[1.0, 1.001, 1.002, 5.0], [1.0, 5.0, 6.0, 7.0], [0.0, 1.0, 2.0, 3.0]])
def test_reject_invalid_source_value(value, energies):
    """An invalid value must fail inside a bin and beyond the output range."""
    table = np.column_stack((energies, [1.0, value, 1.0, 1.0]))
    with pytest.raises(ValueError, match=r"source cross section.*row 2.*source.*level"):
        reduce_phixs_tables_worker(6000.0, output_xgrid(100, 0.03), table, key="level", label="source")
