"""
Functions for validating properties of timeseries objects.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .timeseries import TimeSeries

from typing import Any

import numpy as np
import numpy.typing as npt

# List of [flags](https://numpy.org/doc/stable/reference/generated/numpy.require.html#numpy.require)
# that the array must have.
REQUIRED_FLAGS = ['C_CONTIGUOUS']

def ensure_flags(array: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Ensures that the given array has the required flags.

    Args:
        array (npt.NDArray[np.float64]): The array to validate.

    Returns:
        npt.NDArray[np.float64]: The input array.
    """
    return np.require(array, requirements=REQUIRED_FLAGS)

def validate_1d_numpy_array(array: Any):
    """Validates that the given array is a 1D NumPy array.

    Args:
        array: The array to validate.

    Raises:
        ValueError: If the input is not a NumPy array.
        ValueError: If the input is not 1D.
    """
    if not isinstance(array, np.ndarray):
        raise ValueError(f"The array must be np.ndarray. Got type: {type(array)}")
    if array.ndim != 1:
        raise ValueError(f"The array must be 1D. Got shape: {array.shape}")
    
def validate_time_series(time_series: TimeSeries):
    """Validates that the given time series is valid.

    Args:
        time_series: The time series to validate.

    Raises:
        ValueError: If the time series is not valid.
    """
    validate_1d_numpy_array(time_series.time)
    validate_1d_numpy_array(time_series.data)

    # Check that the time and data arrays are the same length.
    if len(time_series.time) != len(time_series.data):
        raise ValueError('The time and data arrays must be the same length.')

    # Check that the time array is monotonically increasing.
    if not np.all(np.diff(time_series.time) > 0):
        raise ValueError('The time array must be monotonically increasing.')
