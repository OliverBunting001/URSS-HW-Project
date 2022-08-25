from itertools import product
from matplotlib import pyplot as plt
import numpy as np
x=10
y=11
z=12
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection="3d")
space = np.array([*product(range(x), range(y), range(z))]) # all possible triplets of numbers from 0 to N-1
volume = np.random.rand(x, y, z) # generate random data
ax.scatter(space[:,0], space[:,1], space[:,2], c=space/12, s=volume*300)