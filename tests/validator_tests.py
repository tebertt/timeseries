import pytest

from typing import Any

import numpy as np

from src import validators

@pytest.mark.parametrize(
    ("input", "error_type"),
    [
        (1, ValueError),
        ("a", ValueError),
        ([1, 2], ValueError),
        ({"a": 1}, ValueError),
        (np.array([1, 2, 3]), None),
        (np.array([1]), None),
        (np.array([[1, 2], [3, 4]]), ValueError),
        (np.array([[1, 2, 3]]), ValueError),
    ] # type: ignore
)
def validate_1d_numpy_array_test(
    input: Any, error_type: 'type[Exception] | None'
) -> None:
    """Test the validate_1d_numpy_array function."""
    if error_type is not None:
        with pytest.raises(error_type):
            validators.validate_1d_numpy_array(input)
    else:
        validators.validate_1d_numpy_array(input)
