import cupy as cp
import numpy as np









a = np.zeros(12)
b = np.zeros((3, 4))

print(a, b)

for i in range(len(a)):
    a[i]=i+1

a=np.reshape(a, (3, 4))


print(a)


