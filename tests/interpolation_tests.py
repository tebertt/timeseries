import pytest

import numpy as np
import numpy.typing as npt

from src.timeseries import TimeSeries
from src.timeseries import interpolate

@pytest.mark.parametrize("time_series_name", [
    ("time_series_constant")
])
@pytest.mark.parametrize("rh_data_name, error", [
    ("time_series_constant", None),
    ("time_series_constant_small", None),
    ("time_series_constant_offset", None),
    ("data_array_constant", None),
    ("data_array_constant_small", None),
    ("data_value", TypeError)
])

def interpolate_test(request: pytest.FixtureRequest, time_series_name: str, rh_data_name: str, error: 'type[Exception] | None'):

    time_series = request.getfixturevalue(time_series_name)
    rh_data = request.getfixturevalue(rh_data_name)

    if error is None:

        # Perform calculation
        result1, result2 = interpolate(time_series, rh_data)

        # size = max(len(time_series), len(rh_data))

        assert np.array_equal(result1.time, result2.time)
        assert len(result1.data) == len(result2.data)

    else:

        with pytest.raises(error):
            # Raise error if expected
            interpolate(time_series, rh_data)
