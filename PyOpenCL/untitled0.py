#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 11:57:09 2022

@author: ug-ml
"""






##################################
#||||||||||||||||||||||||||||||||# 
################################## # input variables
# NB cubic crystals only! Everything here in the crystal reference frame
################################## # material 
# Extinction distances
# X0i is the imaginary part of the 000 extinction distance
# thickness fringes disappear at about X0i nm
X0i = 1000.0  # nm
# Xg is the (complex) extinction distance for g
# The imaginary part should be larger than X0i
Xg = 58.0+ 1j * X0i * 1.06  # nm

# lattice parameter nm
a0 = 0.4

# Poisson's ratio
nu = 0.3

################################## # sample and imaging conditions
# crystal thickness, nm
t0 = 323  # nm

# the electron beam direction (pointing into the image)
z = np.array((1, 5, 6))#Miller indices

# the foil normal (also pointing into the image)
n = np.array((5, 2, 8))#Miller indices

# g-vector
g = np.array((1,1,-1))#Miller indices

# deviation parameter (typically between -0.1 and 0.1)
s = 0.0046

################################## # the dislocation
# Burgers vector 
#b0 = np.array((-0.5, 0.0, -0.5))
#b0 = np.array((0.0, 0.5, -0.5))
b0 = np.array((-0.5, -0.5, 0.0))
#b0 = np.array((eps, 0.0, 0.0))

# line direction
u = np.array((5, 2, 3))#Miller indices

################################## # calculation and image variables
# integration step (decrease this until the image stops changing)
dt = 0.2  # fraction of a slice

# pixel scale is 1 nm per pixel, by default
# We change the effective magnification of the image (more or less pixels)
#by the scale factor pix2nm
# with an according increase (<1) or decrease (>1) in calculation time
pix2nm = 0.5# nm per pixel

# default number of pixels arounnd the dislocation
pad = 60  # nm

#Gaussian blur sigma, nm
blursigma = 2.0
##################################
#||||||||||||||||||||||||||||||||# 
################################## # end of input



# setup calculations
################################## Normalise, etc.
# convert the inputs into unit vectors
# line direction
u = u / (np.dot(u, u) ** 0.5)
# beam direction
z = z / (np.dot(z, z) ** 0.5)
# foil normal
n = n / (np.dot(n, n) ** 0.5)
# we want n pointing to the same side of the foil as z
if np.dot(n, z) < 0:  # they're antiparallel, reverse n
    n = -n
# also we want u pointing to the same side of the foil as z
if np.dot(u, z) < 0:  # they're antiparallel, reverse u and b
    u = -u
    b = -b
    

# scale dimensions
blursigma = blursigma / pix2nm
#scale thickness & pad too to make later eqns neater
t = t0 / pix2nm
pad = pad / pix2nm
a = a0 / pix2nm
X0i = X0i / pix2nm
Xg = Xg / pix2nm
# number of wave propagation steps
zlen = int(t/dt + 0.5)
# g-vector magnitude, nm^-1
g = g / a
# Burgers vector in nm
b= a * b0






























################################## Crystal<->Simulation frames
# x, y and z are the defining unit vectors of the simulation volume
# written in the crystal frame
# x is defined by the cross product of u and z
# check if they're parallel and use an alternative if so
if abs(np.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    #Think will not work, needs a different approach to the calculation
    x = b1[:] #WHAT IS THIS b1?????
    x = x / (np.dot(x, x) ** 0.5)
    if abs(np.dot(x, z) - 1) < eps:  # they're parallel too, set x parallel to g
        x = g[:]  # this will fail for u=z=b=g but it would be stupid
    phi=0.0# angle between dislocation and z-axis
else:
    x = np.cross(u, z)
    # angle between dislocation and z-axis
    phi = np.arccos(abs(np.dot(u, z)))
x = x / (np.dot(x, x) ** 0.5)
# y is the cross product of z & x
y = np.cross(z, x)
# transformation matrices between simulation frame & crystal frame
c2s = np.array((x, y, z))
s2c = np.transpose(c2s)
# foil normal in the simulation frame gives direction cosines
nS = c2s @ n
# dislocation line direction in the simulation frame
uS = c2s @ u
# some useful vectors    
n1 = np.array((0.0, n[1], n[2]))
n2 = np.array((n[0], 0.0, n[2]))
#they are unit vectors
n1 = n1 / (np.dot(n1, n1) ** 0.5)
n2 = n2 / (np.dot(n2, n2) ** 0.5)
#angle between n1 and z; foil tilt along the dislocation
psi = np.arccos(np.dot(n1,z))*np.sign(n[1])
#angle between n2 and z; foil tilt perpendicular to the dislocation
theta = np.arccos(np.dot(n2,z))*np.sign(n[0])



################################## Crystal<->Dislocation frames
# dislocation frame has zD parallel to u & xD parallel to x
# yD is given by their cross product
xD = x
yD = np.cross(u, x)
zD = u
# transformation matrix between crystal frame & dislocation frame
c2d = np.array((xD, yD, zD))
d2c = np.transpose(c2d)




















