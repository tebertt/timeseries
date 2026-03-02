# TimeSeries

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

TimeSeries a data structure in Python for using and manipulating time series data.

## Structure

TimeSeries has 2 arrays:
- time
- data

The indices for each array correspond to the time and data for each point.

## Functionality

TimeSeries can perform multiple mathematical operations in Python.
It can perform these operations with numbers, arrays, and other TimeSeries objects.

### Unary
- Absolute Value
- Negative
- Length

### Binary 
- Addition
- Subtraction
- Multiplicaion
- Division

### Functions
- Downsample (Can choose method)
- Sampling Frequency

### Interpolation
In addition to its functionality, the TimeSeries data structure can also interpolate in mathematical operations.
If the TimeSeries are two different lengths, the smaller one will be interpolated up to meet the size of the larger one using `scipy.interpolate.interp1d`. 
