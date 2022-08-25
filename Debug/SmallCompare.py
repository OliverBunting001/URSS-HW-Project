import numpy as np
import cupy as cp

x = cp.array([1, 0, -1])
y = np.array([1, 0, -1])

x = x/(sum(x * x))**0.5
y = y/(np.dot(y, y))**0.5

cp.testing.assert_array_almost_equal(x,y, decimal=18)