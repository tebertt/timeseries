import pytest

import operator
from typing import Callable

import numpy as np
import numpy.typing as npt

from src.timeseries import TimeSeries

# Operator Fixtures
pytestmark = pytest.mark.parametrize("operator", [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow
])

def operation_time_series_no_interpolation_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on a time series without interpolation.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, time_series_constant)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_time_series_offset_interpolation_test(operator: Callable, data_value: float, time_series_constant: TimeSeries, time_series_constant_offset: TimeSeries):
    """
    Test arithmetic functionality on a time series with interpolation.
    """

    size = 501

    data = np.full(size, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, time_series_constant_offset)

    assert np.array_equal(result.data, data)

def operation_time_series_reverse_offset_interpolation_test(operator: Callable, data_value: float, time_series_constant: TimeSeries, time_series_constant_offset: TimeSeries):
    """
    Test arithmetic functionality on a time series with interpolation.
    """

    size = 501

    data = np.full(size, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant_offset, time_series_constant)

    assert np.array_equal(result.data, data)

def operation_time_series_different_size_interpolation_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries, time_series_constant_small: TimeSeries):
    """
    Test arithmetic functionality on a time series with interpolation.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, time_series_constant_small)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_time_series_different_size_offset_interpolation_test(operator: Callable, data_value: float, time_series_constant: TimeSeries, time_series_constant_small_offset: TimeSeries):
    """
    Test arithmetic functionality on a time series with interpolation.
    """

    size = 501

    data = np.full(size, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, time_series_constant_small_offset)

    assert np.array_equal(result.data, data)

def operation_array_no_interpolation_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], data_array_constant: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on an array without interpolation.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, data_array_constant)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)


def operation_array_interpolation_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], data_array_constant_small: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on an array without interpolation.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, data_array_constant_small)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_array_empty_test(operator: Callable, time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on an array without interpolation.
    """

    # Perform calculation and expect an error
    with pytest.raises(ValueError):
        operator(time_series_constant, np.ndarray([]))

def operation_reverse_array_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test the reverse arithmetic functionality on an array.
    """
    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(np.full(size_value, data_value), time_series_constant)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_float_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on a float.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, data_value)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_reverse_float_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64],time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on an int.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(data_value, time_series_constant)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_int_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test arithmetic functionality on an int.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(time_series_constant, int(data_value))

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_reverse_int_test(operator: Callable, size_value: int, data_value: float, time_array: npt.NDArray[np.float64], time_series_constant: TimeSeries):
    """
    Test the reverse arithmetic functionality on an int.
    """

    data = np.full(size_value, operator(data_value, data_value))

    # Perform calculation
    result: TimeSeries = operator(int(data_value), time_series_constant)

    assert np.array_equal(result.time, time_array)
    assert np.array_equal(result.data, data)

def operation_error_test(operator: Callable, time_series_constant: TimeSeries):
    """
    Test arithmetic functionality without interpolation.
    """

    # Perform calculation and expect an error
    with pytest.raises(TypeError):
        operator(time_series_constant, "one")

def operation_reverse_error_test(operator: Callable, time_series_constant: TimeSeries):
    """
    Test arithmetic functionality without interpolation.
    """

    # Perform calculation and expect an error
    with pytest.raises(TypeError):
        operator("one", time_series_constant)
