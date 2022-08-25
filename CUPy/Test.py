import cupy as cp
import numpy as np
x = cp.arange(6).reshape(2, 3).astype('f')
print(x)

print(x.sum(axis=1))