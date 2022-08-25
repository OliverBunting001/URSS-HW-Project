import cupy as cp
import random as rand
import numpy as np
import time

vec_add=cp.ElementwiseKernel(
    'S x, T y',
    'S z',
    'z=x+y',
    'vec_add')

long_single=True
short_many=False





if long_single==True:
    no_loop=1
    vec_rank=40000
if short_many==False:
    no_loop=20000
    vec_rank=40




x = cp.zeros(vec_rank)
y = cp.zeros(vec_rank)
z = cp.zeros(vec_rank)
cptime=0
nptime=0


for loop in range(no_loop):
    for i in range(vec_rank):
        x[i]=rand.randint(0, 300)
        y[i]=rand.randint(0, 300)
        
    toc1=time.perf_counter()
    vec_add(x, y)
    tic1=time.perf_counter()
        
    toc2=time.perf_counter()
    z=x+y
    tic2=time.perf_counter()
    
    cptime+=(tic1-toc1)
    nptime+=(tic2-toc2)
    #print(tic1, toc1, tic2, toc2)





print("Numpy took", nptime, "seconds.")
print("Cupy took", cptime, "seconds.")
print("Cupy is", (nptime-cptime)*100/nptime, "% better")


    
