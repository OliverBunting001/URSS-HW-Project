import cupy as cp 
import matplotlib.pyplot as plt
import time
#import cv2 as cv
import inputs as inn
from tqdm import tqdm
eps=0.000000000001

X0i=inn.X0i
Xg=inn.Xg
a0 = inn.a0
nu = inn.nu
t0 = inn.t0
n = cp.array(inn.n)
g = cp.array(inn.g)
s = cp.array(inn.s)
z = cp.array(inn.z)
b0 = cp.array(inn.b0)
u = cp.array(inn.u)
dt = inn.dt
pix2nm = inn.pix2nm
pad = inn.pad
blursigma = inn.blursigma
#print(100000000000*s)

def normalise(x):
    norm = x / (cp.dot(x, x) ** 0.5)
    return norm

#u = normalise(u)
z = normalise(z)
n = normalise(n)


u1=u*1



#%%

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import inputs as inn

eps = 0.000000000001


################################## Choose method
   


toc = time.perf_counter()


X0i=inn.X0i #scalar
Xg=inn.Xg #complex number
a0 = inn.a0 #scalar
nu = inn.nu
t0 = inn.t0
n = np.array(inn.n)
g = np.array(inn.g)
s = np.array(inn.s)
z = np.array(inn.z)
b0 = np.array(inn.b0)
u = np.array(inn.u)
dt = inn.dt
pix2nm = inn.pix2nm
pad = inn.pad
blursigma = inn.blursigma
#u = u / (np.dot(u, u) ** 0.5)
z = z / (np.dot(z, z) ** 0.5)
n = n / (np.dot(n, n) ** 0.5)
#print(100000000*u, 100000000*z, 100000000*n)
# we want n pointing to the same side of the foil as z
if np.dot(n, z) < 0:  # they're antiparallel, reverse n
    n = -n
# also we want u pointing to the same side of the foil as z
if np.dot(u, z) < 0:  # they're antiparallel, reverse u and b
    u = -u
    b0 = -b0

cp.testing.assert_array_almost_equal(u,u1, decimal=18)










