import pytest

import numpy as np
import numpy.typing as npt

from src.timeseries import TimeSeries

# Value Fixtures
@pytest.fixture()
def size_value() -> int:
    """
    Fixture to define the size of the time series for testing.
    """
    return 1000

# Value Fixtures
@pytest.fixture()
def size_value_small() -> int:
    """
    Fixture to define the size of the time series for testing.
    """
    return 20

@pytest.fixture()
def start_time() -> float:
    """
    Fixture to define the start time for testing.
    """
    return -1.0

@pytest.fixture()
def end_time() -> float:
    """
    Fixture to define the end time for testing.
    """
    return 1.0

@pytest.fixture()
def data_value() -> float:
    """
    Fixture to define a constant data value for testing.
    """
    return 2.0

@pytest.fixture()
def offset_value() -> float:
    """
    Fixture to define an offset value for testing.
    """
    return 1.0

# Array Fixtures
@pytest.fixture()
def time_array(size_value: int, start_time: float, end_time: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a time array for testing.
    """
    return np.linspace(start_time, end_time, size_value)

@pytest.fixture()
def data_array_constant(size_value: int, data_value: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a data array for testing.
    """
    return np.full(size_value, data_value)

@pytest.fixture()
def data_array_linear(size_value: int, data_value: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a data array for testing.
    """
    # return np.full(size_value, data_value)
    return np.linspace(-data_value, data_value, size_value)

@pytest.fixture()
def data_array_sine(size_value: int, data_value: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a data array for testing.
    """
    # return np.full(size_value, data_value)
    return np.linspace(-data_value, data_value, size_value)

# Array Fixtures
@pytest.fixture()
def time_array_small(size_value_small: int, start_time: float, end_time: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a time array for testing.
    """
    return np.linspace(start_time, end_time, size_value_small)

@pytest.fixture()
def data_array_constant_small(size_value_small: int, data_value: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a data array for testing.
    """
    return np.full(size_value_small, data_value)

@pytest.fixture()
def data_array_linear_small(size_value_small: int, data_value: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a data array for testing.
    """
    return np.linspace(-data_value, data_value, size_value_small)

@pytest.fixture()
def data_array_sine_small(size_value_small: int, data_value: float) -> npt.NDArray[np.float64]:
    """
    Fixture to create a data array for testing.
    """
    return np.sin(np.linspace(-data_value, data_value, size_value_small))

# Time Series Fixtures
@pytest.fixture()
def time_series_constant(time_array: npt.NDArray[np.float64], data_array_constant: npt.NDArray[np.float64]) -> TimeSeries:
    """
    Fixture to create a constant time series for testing.
    """
    return TimeSeries(time_array, data_array_constant)

@pytest.fixture()
def time_series_constant_offset(time_array: npt.NDArray[np.float64], data_array_constant: npt.NDArray[np.float64], offset_value: float) -> TimeSeries:
    """
    Fixture to create a constant time series for testing.
    """
    return TimeSeries(time_array + offset_value, data_array_constant)

@pytest.fixture()
def time_series_constant_small(time_array_small: npt.NDArray[np.float64], data_array_constant_small: npt.NDArray[np.float64]):
    """
    Fixture to create a constant time series for testing.
    """
    return TimeSeries(time_array_small, data_array_constant_small)

@pytest.fixture()
def time_series_constant_small_offset(offset_value: float, time_array_small: npt.NDArray[np.float64], data_array_constant_small: npt.NDArray[np.float64]):
    """
    Fixture to create a constant time series for testing.
    """
    return TimeSeries(time_array_small + offset_value, data_array_constant_small)

@pytest.fixture()
def time_series_linear(time_array: npt.NDArray[np.float64], data_array_linear: npt.NDArray[np.float64]) -> TimeSeries:
    """
    Fixture to create a linear time series for testing.
    """
    return TimeSeries(time_array, data_array_linear)

@pytest.fixture()
def time_series_linear_offset(time_array: npt.NDArray[np.float64], data_array_linear: npt.NDArray[np.float64], offset_value: float) -> TimeSeries:
    """
    Fixture to create a linear time series for testing.
    """
    return TimeSeries(time_array + offset_value, data_array_linear)

@pytest.fixture()
def time_series_linear_small(time_array_small: npt.NDArray[np.float64], data_array_linear_small: npt.NDArray[np.float64]):
    """
    Fixture to create a linear time series for testing.
    """
    return TimeSeries(time_array_small, data_array_linear_small)

@pytest.fixture()
def time_series_linear_small_offset(offset_value: float, time_array_small: npt.NDArray[np.float64], data_array_linear_small: npt.NDArray[np.float64]):
    """
    Fixture to create a linear time series for testing.
    """
    return TimeSeries(time_array_small + offset_value, data_array_linear_small)

@pytest.fixture()
def time_series_sine(time_array: npt.NDArray[np.float64], data_array_sine: npt.NDArray[np.float64]) -> TimeSeries:
    """
    Fixture to create a sine time series for testing.
    """
    return TimeSeries(time_array, data_array_sine)

@pytest.fixture()
def time_series_sine_offset(time_array: npt.NDArray[np.float64], data_array_sine: npt.NDArray[np.float64], offset_value: float) -> TimeSeries:
    """
    Fixture to create a sine time series for testing.
    """
    return TimeSeries(time_array + offset_value, data_array_sine)

@pytest.fixture()
def time_series_sine_small(time_array_small: npt.NDArray[np.float64], data_array_sine_small: npt.NDArray[np.float64]):
    """
    Fixture to create a sine time series for testing.
    """
    return TimeSeries(time_array_small, data_array_sine_small)

@pytest.fixture()
def time_series_sine_small_offset(offset_value: float, time_array_small: npt.NDArray[np.float64], data_array_sine_small: npt.NDArray[np.float64]):
    """
    Fixture to create a sine time series for testing.
    """
    return TimeSeries(time_array_small + offset_value, data_array_sine_small)
