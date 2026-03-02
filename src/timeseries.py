"""The TimeSeries class."""

from typing import Tuple, Callable, Literal
from dataclasses import dataclass, field
from operator import add, sub, mul, truediv, pow

import numpy as np
import numpy.typing as npt
import scipy.interpolate
from tsdownsample import (
    LTTBDownsampler,
    MinMaxDownsampler,
    MinMaxLTTBDownsampler,
    EveryNthDownsampler,
    # M4Downsampler
)

from .validators import (
    validate_time_series,
    ensure_flags,
    validate_1d_numpy_array
)

DOWNSAMPLE_METHODS = Literal[
    "lttb",
    "min max",
    "min max lttb",
    "every nth",
    # "m4"
]

@dataclass
class TimeSeries:
    """A time series data type."""
    _time: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    _data: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64))
    
    __numpy_ufunc__ = None # Numpy up to 13.0
    __array_ufunc__ = None # Numpy 13.0 and above

    def __post_init__(self):
        """Performs operations on the data after the object is initialized."""
        validate_time_series(self)
        self._time = ensure_flags(self._time)
        self._data = ensure_flags(self._data)

    @property
    def time(self) -> npt.NDArray[np.float64]:
        """Returns the time array."""
        return self._time

    @time.setter
    def time(self, value: npt.NDArray[np.float64]):
        """Sets the time array."""
        validate_1d_numpy_array(value)
        value = ensure_flags(value)
        self._time = value

    @property
    def data(self) -> npt.NDArray[np.float64]:
        """Returns the data array."""
        return self._data

    @data.setter
    def data(self, value: npt.NDArray[np.float64]):
        """Sets the data array."""
        validate_1d_numpy_array(value)
        value = ensure_flags(value)
        self._data = value

    @property
    def fs(self) -> float:
        """Returns the sampling frequency of the time series."""
        return 1 / np.mean(np.diff(self.time)) # type: ignore

    # Implement the len() function for this class:
    def __len__(self):
        """Returns the length of the time series."""
        return self.time.size

    def __abs__(self):
        """Returns the absolute value of the time series."""
        return TimeSeries(self.time, np.abs(self.data))

    def __neg__(self):
        """Returns the negation of the time series."""
        return TimeSeries(self.time, -self.data)

    def __operation_helper(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
        operation: Callable,
    ) -> 'TimeSeries':
        """Helper function to perform an operation on the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.
            operation: The operation to perform.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The result of the operation.
        """
        if isinstance(other, (TimeSeries, np.ndarray)):
            self, other = interpolate(self, other)
            return TimeSeries(self.time, operation(self.data, other.data))
        elif isinstance(other, (int, float)):
            return TimeSeries(self.time, operation(self.data, other))
        else:
            raise TypeError('The operand must be a TimeSeries, array, float, or int.')

    def __reverse_operation_helper(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
        operation: Callable,
    ) -> 'TimeSeries':
        """Helper function to perform an operation in reverse on the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.
            operation: The operation to perform.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The result of the operation.
        """
        if isinstance(other, (TimeSeries, np.ndarray)):
            self, other = interpolate(self, other)
            return TimeSeries(self.time, operation(other.data, self.data))
        elif isinstance(other, (int, float)):
            return TimeSeries(self.time, operation(other, self.data))
        else:
            raise TypeError('The operand must be a TimeSeries, array, float, or int.')

    def __add__(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the sum of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The sum of the time series and another time series or float.
        """

        return self.__operation_helper(other, add)

    def __radd__(
        self,
        other: 'float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the sum of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The sum of the time series and another time series, array, float or int.
        """

        return self.__reverse_operation_helper(other, add)

    def __sub__(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the difference of the time series and another time series, array, float, or int.

        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.
        Returns:
            The difference of the time series and another time series, array, float or int.
        """
        return self.__operation_helper(other, sub)

    def __rsub__(
        self,
        other: 'float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the difference of the time series and another time series, array, float, or int.

        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The difference of the time series and another time series, array, float or int.
        """
        return self.__reverse_operation_helper(other, sub)

    def __mul__(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the product of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The product of the time series and another time series or float.
        """
        return self.__operation_helper(other, mul)

    def __rmul__(
        self,
        other: 'float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the product of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The product of the time series and another time series or float.
        """
        return self.__reverse_operation_helper(other, mul)

    def __truediv__(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the quotient of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The quotient of the time series and another time series or float.
        """
        return self.__operation_helper(other, truediv)

    def __rtruediv__(
        self,
        other: 'float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the quotient of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The quotient of the time series and another time series or float.
        """
        return self.__reverse_operation_helper(other, truediv)

    def __pow__(
        self,
        other: 'TimeSeries | float | int | npt.NDArray[np.float64]',
    ) -> 'TimeSeries':
        """Returns the power of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The power of the time series and another time series or float.
        """
        return self.__operation_helper(other, pow)

    def __rpow__(
        self,
        other: 'TimeSeries',
    ) -> 'TimeSeries':
        """Returns the power of the time series and another time series, array, float, or int.
        
        Args:
            other: The other operand.

        Raises:
            TypeError: If the other operand is not a TimeSeries, array, float, or int.

        Returns:
            The power of the time series and another time series or float.
        """
        return self.__reverse_operation_helper(other, pow)
    
    def downsample(self, target_number_of_points: int, method: DOWNSAMPLE_METHODS = "min max lttb") -> 'TimeSeries':
        """Downsamples the time series using the selected method.

        Args:
            target_number_of_points (int): The target number of points.
            method (str): The method to downsample with. Can be one of:
                - "lttb": Largest Triangle Three Buckets
                - "min max": Min-Max Downsampling
                - "min max lttb": Min-Max and LTTB Downsampling
                - "every nth": Every Nth Downsampling
                - "m4": M4 Downsampling

        Returns:
            TimeSeries: The downsampled time series.
        """

        ds = downsample(self, target_number_of_points, method)

        self.time = ds.time
        self.data = ds.data

# Interpolation Functions
def match_time_series_ranges(
    time_series1: TimeSeries,
    time_series2: TimeSeries,
) -> Tuple[TimeSeries, TimeSeries]:
    """Match the time ranges of two sets of time, data arrays.

    Args:
        time_series1 (TimeSeries): The first time series
        time_series2 (TimeSeries): The second time series

    Returns:
        Tuple[TimeSeries, TimeSeries]: The two time series with the same time ranges
    """
    # If either array is empty, return all empty arrays
    if len(time_series1.time) == 0 or len(time_series2.time) == 0:
        return (TimeSeries(), TimeSeries())

    # Check if the start times are approximately the same
    # by checking if the first element in time_series1.time
    # is between the first and second elements in time_series2.time
    # or if the first element in time_series2.time
    # is between the first and second elements in time_series1.time
    start_times_equal = False
    if (
        time_series1.time[0] >= time_series2.time[0]
        and time_series1.time[0] <= time_series2.time[1]
    ):
        start_times_equal = True
    elif (
        time_series2.time[0] >= time_series1.time[0]
        and time_series2.time[0] <= time_series1.time[1]
    ):
        start_times_equal = True

    # Check if the end times are approximately the same
    end_times_equal = False
    if (
        time_series1.time[-1] >= time_series2.time[-1]
        and time_series1.time[-1] <= time_series2.time[-2]
    ):
        end_times_equal = True
    elif (
        time_series2.time[-1] >= time_series1.time[-1]
        and time_series2.time[-1] <= time_series1.time[-2]
    ):
        end_times_equal = True

    if start_times_equal and end_times_equal:
        return (time_series1, time_series2)

    array1_time = time_series1.time
    array1_data = time_series1.data
    array2_time = time_series2.time
    array2_data = time_series2.data

    # If the start times are not approximately the same, trim the longer one:
    if not start_times_equal:
        if time_series1.time[0] < time_series2.time[0]:
            # Find the first element in time_series1.time that is greater than the
            #   first element in time_series2.time
            # and trim time_series1.time and array1_data to that index
            closest_index = np.argmin(np.abs(time_series1.time - time_series2.time[0]))
            array1_time = time_series1.time[closest_index:]
            array1_data = time_series1.data[closest_index:]
        else:
            # Find the first element in time_series2.time that is greater than the
            #   first element in time_series1.time
            # and trim time_series2.time and array2_data to that index
            closest_index = np.argmin(np.abs(time_series2.time - time_series1.time[0]))
            array2_time = time_series2.time[closest_index:]
            array2_data = time_series2.data[closest_index:]

    # If the end times are not approximately the same, trim the longer one:
    if not end_times_equal:
        if time_series1.time[-1] < time_series2.time[-1]:
            # Find the first element in time_series2.time that is less than the
            #   last element in time_series1.time
            # and trim time_series2.time and array2_data to that index
            closest_index = np.argmin(np.abs(time_series2.time - time_series1.time[-1]))
            array2_time = time_series2.time[:closest_index + 1]
            array2_data = time_series2.data[:closest_index + 1]
        else:
            # Find the first element in time_series1.time that is less than the
            #   last element in time_series2.time
            # and trim time_series1.time and array1_data to that index
            closest_index = np.argmin(np.abs(time_series1.time - time_series2.time[-1]))
            array1_time = time_series1.time[:closest_index + 1]
            array1_data = time_series1.data[:closest_index + 1]

    return (TimeSeries(array1_time, array1_data), TimeSeries(array2_time, array2_data))

def array_to_time_series(
    data: npt.NDArray[np.float64],
    start_time: float,
    end_time: float,
) -> TimeSeries:
    """Convert a data array to a time series with evenly spaced time data.

    Args:
        data (npt.NDArray[np.float64]): Data to convert to a time series
        start_time (float): Start time of the time series
        end_time (float): End time of the time series

    Returns:
        TimeSeries: Time series with the given data and time range
    """
    if data.size <= 0:
        raise ValueError("Data array must not be empty.")

    time = np.linspace(start_time, end_time, data.size)
    return TimeSeries(time, data)

def interpolate_to_time_series(
    time_series: TimeSeries,
    interpolated_time_data: npt.NDArray[np.float64],
) -> TimeSeries:
    """Interpolate the data of a time series to match the given time data.

    Args:
        time_series (TimeSeries): Time series to interpolate
        interpolated_time_data (npt.NDArray[np.float64]): Time data to interpolate to

    Returns:
        TimeSeries: Interpolated time series
    """
    interpolator = scipy.interpolate.interp1d(
        time_series.time,
        time_series.data,
        kind="linear",
        fill_value="extrapolate", # type: ignore
    )
    interpolated_data = interpolator(interpolated_time_data)
    return TimeSeries(interpolated_time_data, interpolated_data)


def interpolate(time_series1: TimeSeries, time_series2: 'TimeSeries | npt.NDArray[np.float64]') -> Tuple[TimeSeries, TimeSeries]:
    """Interpolate the data of two time series to match each other.
    The lower rate time series will be interpolated to match the higher rate time series.

    Args:
        time_series1 (TimeSeries): The first time series
        time_series2 (TimeSeries | npt.NDArray[np.float64]): The second time series

    Returns:
        Tuple[TimeSeries, Timeseries]: The result of the interpolated time series
    """
    if isinstance(time_series2, TimeSeries):

        # Match the time ranges of the two time series by trimming
        time_series1, time_series2 = match_time_series_ranges(
            time_series1,
            time_series2,
        )

    elif isinstance(time_series2, np.ndarray):

        # If time_series2 is a numpy array, convert it to a TimeSeries base on the time range of time_series1
        time_series2 = array_to_time_series(
            time_series2,
            time_series1.time[0],
            time_series1.time[-1],
        )

    else:
        raise TypeError("Time series 2 must be a TimeSeries or a numpy array")

    # Match the lower rate time series to the higher rate time series
    # by interpolating the lower rate time series to the time data of the higher rate timeseries
    if not np.array_equal(time_series1.time, time_series2.time):

        # If the second time series is longer, interpolate the first time series
        if time_series1.time.size < time_series2.time.size:
            time_series1 = interpolate_to_time_series(time_series1, time_series2.time)

        # If the first time series is longer or they are the same length but not equal, interpolate the second time series
        else:
            time_series2 = interpolate_to_time_series(time_series2, time_series1.time)

    return (time_series1, time_series2)

def downsample(time_series: TimeSeries, target_number_of_points: int, method: DOWNSAMPLE_METHODS = "min max lttb") -> TimeSeries:
    """Downsamples the time series using the selected method.

    Args:
        target_number_of_points (int): The target number of points.
        method (str): The method to downsample with. Can be one of:
            - "lttb": Largest Triangle Three Buckets
            - "min max": Min-Max Downsampling
            - "min max lttb": Min-Max and LTTB Downsampling
            - "every nth": Every Nth Downsampling

    Returns:
        TimeSeries: The downsampled time series.
    """
    # Check if the data has fewer points than the target number of points:
    if len(time_series) <= target_number_of_points:
        return time_series

    # Determine downsample method
    if method.lower() == "lttb":
        s_ds = LTTBDownsampler().downsample(
            time_series.time,
            time_series.data,
            n_out=target_number_of_points,
            parallel=False,
        )
    elif method.lower() == "min max":
        s_ds = MinMaxDownsampler().downsample(
            time_series.time,
            time_series.data,
            n_out=target_number_of_points,
            parallel=True,
        )
    elif method.lower() == "min max lttb":
        s_ds = MinMaxLTTBDownsampler().downsample(
            time_series.time,
            time_series.data,
            n_out=target_number_of_points,
            parallel=True,
        )
    elif method.lower() == "every nth":
        s_ds = EveryNthDownsampler().downsample(
            time_series.data,
            n_out=target_number_of_points,
            parallel=True,
        )
    # elif method.lower() == "m4":
    #     s_ds = M4Downsampler().downsample(
    #         time_series.time,
    #         time_series.data,
    #         n_out=target_number_of_points,
    #         parallel=True,
#     )
    else :
        raise ValueError(f"Invalid downsample method: {method}")

    # Create downsampled time series
    downsampled_time_series = TimeSeries(
        time_series.time[s_ds],
        time_series.data[s_ds],
    )

    # Return the downsampled time series
    return downsampled_time_series