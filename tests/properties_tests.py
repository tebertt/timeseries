import pytest

import numpy as np
import numpy.typing as npt

from src.timeseries import TimeSeries

def init_test(time_array: npt.NDArray[np.float64], data_array_constant: npt.NDArray[np.float64]) -> None:
    """Test the __init__ method."""

    time_series = TimeSeries(time_array, data_array_constant)

    assert np.array_equal(time_series.time, time_array)
    assert np.array_equal(time_series.data, data_array_constant)

def init_array_length_error_test(size_value: int, start_time: float, end_time: float, data_value: float, offset_value: float) -> None:
    """Test the __init__ method with arrays of different lengths."""

    time = np.linspace(start_time, end_time, size_value)
    data = np.full(size_value + int(offset_value), data_value)

    with pytest.raises(ValueError):
        TimeSeries(time, data)

def init_array_time_error_test(size_value: int, start_time: float, end_time: float, data_value: float):
    """Test the __init__ method with time that should raise an error."""

    time = np.linspace(end_time, start_time, size_value)
    data = np.full(size_value, data_value)

    with pytest.raises(ValueError):
        TimeSeries(time, data)

def time_setter_test(time_array: npt.NDArray[np.float64], data_array_constant: npt.NDArray[np.float64]) -> None:
    """
    Test the time setter of the TimeSeries object.
    """
    time_series = TimeSeries()

    time_series.time = time_array

    assert np.array_equal(time_series.time, time_array)

def data_setter_test(time_array: npt.NDArray[np.float64], data_array_constant: npt.NDArray[np.float64]) -> None:
    """
    Test the data setter of the TimeSeries object.
    """
    time_series = TimeSeries()

    time_series.data = data_array_constant

    assert np.array_equal(time_series.data, data_array_constant)

def sample_frequency_test(time_series_constant: TimeSeries):
    """
    Test the sample frequency of a TimeSeries object.
    """
    # Calculate expected sample frequency
    expected = 500
    
    # Get actual sample frequency
    actual_sample_frequency = time_series_constant.fs
    
    # Assert that the actual sample frequency matches the expected value
    assert np.isclose(actual_sample_frequency, expected, rtol=1)

def length_test(size_value: int, time_series_constant: TimeSeries):
    """
    Test the length of the TimeSeries object.
    """
    assert len(time_series_constant) == size_value

def absolute_value_test(size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test the absolute value of the TimeSeries object.
    """

    data = np.full(size_value, -data_value)
    abs_data = np.full(size_value, data_value)

    time_series = TimeSeries(time_array, data)

    result = abs(time_series)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, abs_data)

def negative_test(size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test the negative of the TimeSeries object.
    """

    neg_data = np.full(size_value, -data_value)

    result = -time_series_constant

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, neg_data)