eps = 0.000001

#extinction distances
X0i = 1000.  # nm
XgR = 3.26
Xg = XgR + 1j * X0i * 1.065  # nm



big=40000000
small=0.00000000004
med=0.4

# lattice parameter nm
a0 = med

# Poisson's ratio
nu = 0.3

# crystal thickness, nm
t0 = 3 * XgR # nm


# electron beam direction (pointing into the image) DEFINES Z DIRECTION
z = [5, 1, 4]#Miller indices 

# the foil normal (also pointing into the image)
n = [5, 1, 4]#Miller indices
#nunit = n / (np.dot(n, n) ** 0.5)

# g-vector
g = [1, -1, -1]#Miller indices

# deviation parameter (typically between -0.1 and 0.1)
s = -0.15/XgR


# the dislocation Burgers vector (Miller Indices)

#b0 = np.array((-0.5, 0.0, -0.5))
#b0 = np.array((0.0, 0.5, -0.5))
b0 = [0.5, 0.5, 0.0]
#b0 = np.array((eps, 0.0, 0.0))


# defect line direction
u = [1, 0, -1] #Miller indices

# integration step (decrease this until the image stops changing)
dt = 0.3  # fraction of a slice

# pixel scale is 1 nm per pixel, by default
pix2nm = 0.5 #nm per pixel

# pixels arounnd the dislocation 
pad = 5 #nm

#Gaussian blur sigma
blursigma = 2.0 #nm