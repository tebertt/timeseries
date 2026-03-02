import pytest

from src.timeseries import TimeSeries, downsample

# Parameters
pytestmark = [
    pytest.mark.parametrize("downsample_method, error", [
        (None, None),
        ("lttb", None),
        ("min max", None),
        ("min max lttb", None),
        ("every nth", None),
        # ("m4", None),
        ("wrong", ValueError)
    ]),
    pytest.mark.parametrize("time_series_name", [
        "time_series_linear",
        "time_series_linear_offset",
        "time_series_sine",
        "time_series_sine_offset",
    ])
]

def downsample_function_time_series_test(
    request: pytest.FixtureRequest,
    downsample_method: str,
    error: 'type[Exception] | None',
    time_series_name: str,
    size_value_small: int,
):
    """
    Test downsampling function on a time series.
    """
    time_series: TimeSeries = request.getfixturevalue(time_series_name)

    if error is None:

        if downsample_method is None:
            ds_time_series = downsample(time_series, size_value_small)
        else:
            ds_time_series = downsample(time_series, size_value_small, downsample_method)

        assert len(ds_time_series) == size_value_small

    else:
        with pytest.raises(error):
            downsample(time_series, size_value_small, downsample_method)

def test_downsample_method_time_series_test(
    request: pytest.FixtureRequest,
    downsample_method: str,
    error: 'type[Exception] | None',
    time_series_name: str,
    size_value_small: int,
):
    """
    Test downsampling method from time series.
    """
    time_series: TimeSeries = request.getfixturevalue(time_series_name)

    if error is None:

        if downsample_method is None:
            time_series.downsample(size_value_small)
        else:
            time_series.downsample(size_value_small, downsample_method)

        assert len(time_series) == size_value_small

    else:
        with pytest.raises(error):
            time_series.downsample(size_value_small, downsample_method)