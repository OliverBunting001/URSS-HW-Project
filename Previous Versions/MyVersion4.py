import cupy as np #lol this is going to annoy people
import matplotlib.pyplot as plt
import time
import cv2 as cv
#import inputs as inn
eps=0.000001



#%%
#INPUTS

# NB cubic crystals only! Initially everything in the crystal frame



# Extinction distances


# X0i is the imaginary part of the 000 extinction distance
# thickness fringes disappear at about X0i nm

X0i = 1000.0  # nm


# Xg is the (complex) extinction distance for g
# The imaginary part should be larger than X0i
Xg = 58.0+ 1j * X0i * 1.06  # nm









# sample and imaging conditions



# lattice parameter nm
a0 = 0.4

# Poisson's ratio
nu = 0.3

# crystal thickness, nm
t0 = 323  # nm

# the foil normal (also pointing into the image)
n = np.array((5, 2, 8))#Miller indices
#nunit = n / (np.dot(n, n) ** 0.5)

# g-vector
g = np.array((1, 1, -1))#Miller indices

# deviation parameter (typically between -0.1 and 0.1)
s = 0.0046


# electron beam direction (pointing into the image) DEFINES Z DIRECTION
z = np.array((1, 5, 6))#Miller indices


#THIS HAS IMPLICITLY DECLARED CRYSTAL ORIENTATION
# z = (r, theta, phi)
r = np.dot(z, z)**0.5
zunit = z/r
# z_crys_basis = r cos(theta)
# x_crys_basis = r cos(phi)  *  r sin(theta)
# y_crys_basis = r sin(phi)  *  r sin(theta)

theta = np.arccos(z[2]/r) #THIS IS WITH RESPECT TO CRYSTAL ORIENTATION, NOT FOIL PLANE
phi = np.arctan(z[1]/z[0])
if z[1]<0:
    phi = 2*np.pi - phi


#theta_foil = np.dot(zunit, nunit)
#print(180*theta_foil/np.pi)
'''









# the dislocation Burgers vector (Miller Indices)


#b0 = np.array((-0.5, 0.0, -0.5))
#b0 = np.array((0.0, 0.5, -0.5))
b0 = np.array((-0.5, -0.5, 0.0))
#b0 = np.array((eps, 0.0, 0.0))



# defect line direction
u = np.array((5, 2, 3)) #Miller indices










# calculation and image variables



# integration step (decrease this until the image stops changing)
dt = 10  # fraction of a slice

# pixel scale is 1 nm per pixel, by default
# We change the effective magnification of the image (more or less pixels)
#by the scale factor pix2nm
# with an according increase (<1) or decrease (>1) in calculation time
pix2nm = 0.5 #nm per pixel

#i.e. 0.5 -> 2pixels along 1nm -> 4pix/1nmsqr
#     x -> 1/x pixels along 1nm -> (1/xsqr) pix/1nmsqr




# pixels arounnd the dislocation - characteristic scale of image
pad = 10 #nm

#Gaussian blur sigma
blursigma = 2.0 #nm



# end of input



'''
X0i=inn.X0i
Xg=inn.Xg
a0 = inn.a0
nu = inn.nu
t0 = inn.t0
n = inn.n
g = inn.g
s = inn.s
z = inn.z
b0 = inn.b0
u = inn.u
dt = inn.dt
pix2nm = inn.pix2nm
pad = inn.pad
blursigma = inn.blursigma
'''


#%%




'''
# SETUP



# Normalise
# line direction
u = u / (np.dot(u, u) ** 0.5)
# beam direction
z = z / (np.dot(z, z) ** 0.5)
# foil normal
n = n / (np.dot(n, n) ** 0.5)

# we want n pointing to the same side of the foil as z
if np.dot(n, z) < 0:  # they're antiparallel(ish), reverse n
    n = -n
# also we want u pointing to the same side of the foil as z
if np.dot(u, z) < 0:  # they're antiparallel(ish), reverse u and b
    u = -u
    b0 = -b0
    

# scale dimensions - change units to pixels
blursigma = blursigma / pix2nm    
t = t0 / pix2nm
pad = pad / pix2nm
a = a0 / pix2nm
X0i = X0i / pix2nm
Xg = Xg / pix2nm



# number of wave propagation steps
zlen = int(t/dt + 0.5)
# g-vector magnitude, nm^-1
g = g / a
# Burgers vector
b = a * b0 #nm







# Crystal<->Simulation co-ordinate frames



# x, y and z are the unit vectors of the simulation frame
# written in the crystal frame


################################################
# x is defined by the cross product of u and z #
################################################




#  if u is parallel to z use an alternative


if abs(np.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    #Think will not work, needs a different approach to the calculation
    x = b0[:]
    x = x / (np.dot(x, x) ** 0.5)
    if abs(np.dot(x, z) - 1) < eps:  # they're parallel too, set x parallel to g instead
        x = g[:]  # this will fail for u=z=b=g but that would be stupid
        if abs(np.dot(x, z) - 1) < eps:  # they're parallel too, set x arbitrarily
            x = np.array(1, 0, 0)  
    phi=0.0 # angle between dislocation and z-axis
else:
    x = np.cross(u, z) #not automatically normalised for some reason
    #this reason is that axb=ab sin(theta) hence u and z are not orthogonal
    
    phi = np.arccos(abs(np.dot(u, z)))
    # angle between dislocation and z-axis






x = x / (np.dot(x, x) ** 0.5)
y = np.cross(z, x) # y is the cross product of z & x






# transformation matrices between simulation frame & crystal frame (co-ordinates system)
c2s = np.array((x, y, z))
s2c = np.transpose(c2s)










# normal and dislocation vectors transform to simulated co-ordinate frame

nS = c2s @ n #@ is matrix multiplication

uS = c2s @ u







# some useful vectors

n1, n2 = n, n
n1[0], n2[1] = 0, 0

#they are unit vectors
n1 = n1 / (np.dot(n1, n1) ** 0.5)
n2 = n2 / (np.dot(n2, n2) ** 0.5)

#angle between n1 and z; foil tilt along the dislocation
psi = np.arccos(np.dot(n1,z))*np.sign(n[1])
#angle between n2 and z; foil tilt perpendicular to the dislocation
theta = np.arccos(np.dot(n2,z))*np.sign(n[0])









# Crystal<->Dislocation frames
# dislocation frame has zD parallel to u & xD parallel to x
# yD is given by their cross product
xD = x
yD = np.cross(u, x)
zD = u
# transformation matrix between crystal frame & dislocation frame
c2d = np.array((xD, yD, zD))
d2c = np.transpose(c2d)


# Set up simulation frame (see Fig.A)

# x=((1,0,0)) is vertical up
# y=((0,1,0)) is horizontal left, origin bottom right
# z=((0,0,1)) is into the image



#############
#EXTRA STUFF#
#############

bscrew = np.dot(b,u)#NB a scalar
bedge = c2d @ (b - bscrew*u)#NB a vector
beUnit = bedge/(np.dot(bedge,bedge)**0.5)#a unit vector
bxu = c2d @ np.cross(b,u)
gD = c2d @ g

dz = 0.01
deltaz = np.array((0, dt*dz, 0)) / pix2nm
bxu = c2d @ np.cross(b,u)
##############
#/EXTRA STUFF#
##############




#####################
#CHECK THIS SECTION #
#####################






# FINAL IMAGE dimensions:

# along x: (note this is an even number; the dislocation line is between pixels)
xsiz = 2*int(pad + 0.5) # in pixels
# along y: ysiz = xsiz + dislocation image length (to be calculated below)
# along z: zsiz = t/dt + vertical padding (to be calculated below)



# extra height to account for the parts of the image to the left and right of the dislocation
hpad = xsiz/np.tan(phi)
# y dimension calculation
if abs(np.dot(u, z)) < eps:  # dislocation is in the plane of the foil
    
    #in this scenario the dislocation projection is infinite
    #so choose to make grid square
    ysiz = 1 * xsiz # in pixels
    zsiz = zlen # in slices
    print("Dislocation is in the plane of the foil")
    #the plane normal to z isn't necessarily the foil plane is it??
    #perpendicular to beam?


elif abs(np.dot(u, z)-1) < eps:  # dislocation along z
    #needs work?
    
    #likewise here, projection is zero os choose a square
    ysiz = 1 * xsiz
    zsiz = zlen # in slices
    print("Dislocation is parallel to the beam")


else:  # dislocation is at an angle to the beam direction
    # dislocation image length
    w = int(t*nS[2]*nS[2]*uS[1]/abs(np.dot(u,n1)) + 0.5)
    ysiz = w + xsiz # in pixels
    zsiz = int( (2*t*nS[2] + hpad + xsiz*np.tan(abs(psi)) )/dt + 0.5) # in slices



print("(xsiz,ysiz,zsiz)=", xsiz,ysiz,zsiz) #x value seems bigger than it should be?
#xsiz, ysiz, zsiz = 10, 20, 10




######################
#/CHECK THIS SECTION #
######################

######################
#    Functions       #
######################



def gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD):
    
    rmag = np.sum(rD*rD, axis=2)**0.5 #matrix
    
    
    ct = np.dot(rD,beUnit)/rmag #matrix
    sbt = np.cross(beUnit,rD)/rmag[...,None] #vector matrix
    st = np.dsplit(sbt, (0,1,2))[2].reshape(xsiz, zsiz+1) #matrix
    #this is fishy - needs clearing up
    
    
    Rscrew_2 = bscrew*(np.arctan(rD_1/rX)-np.pi*(rX<0))/(2*np.pi) #component 3 of vector matrix
    Rscrew = np.dstack((np.zeros((xsiz, zsiz+1, 2)), Rscrew_2)) #vector matrix
    Redge0 = (ct*st)[...,None] * bedge/(2*np.pi*(1-nu)) #vector matrix
    Redge1 = ( ((2-4*nu)*np.log(rmag)+(ct**2-st**2)) /(8*np.pi*(1-nu)))[...,None]*bxu #vector matrix
    R = (Rscrew + Redge0 + Redge1) #vector matrix
    gR = np.dot(R, gD) #matrix
    return gR


def howieWhelan(F_in,Xg,X0i,s,alpha,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real #const
    Xgi = Xg.imag #const

    s = s + eps #xy matrix

    gamma = np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2]) #2vector of gamma xy matrices
        
    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))]) #2vector of q xy matrices

    beta = np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5)) #xy matrix
    #alpha=const
    #scattering matrix
    C=np.array([[np.cos(beta/2), np.sin(beta/2)], #2matrix of xy matrices
                 [-np.sin(beta/2)*np.exp(complex(0,alpha)),
                  np.cos(beta/2)*np.exp(complex(0,alpha))]])
    
    #inverse of C is likewise a 2matrix of xy matrices
    Ci= np.array([[np.cos(beta/2), -np.sin(beta/2)*np.exp(complex(0,-alpha))],
                 [np.sin(beta/2),  np.cos(beta/2)*np.exp(complex(0,-alpha))]])

    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]], dtype=np.float_)
    #gamma/q[0/1] are all xy matrices
    #thus this is a 2matrix of xy matrices
    #0*gamma[0] give these zeroes the right dimensionality
    #np.zeros probably better
    
    
    
    gamma = np.transpose(gamma, (1,2,0)).reshape(xsiz,ysiz,2,1) #xy matrix of gamma 2vectors
    q = np.transpose(q, [1,2,0]).reshape(xsiz,ysiz,2,1) #xy matrix of q 2vectors

    C=np.transpose(C, [2,3,0,1])
    Ci=np.transpose(Ci, [2,3,0,1])
    G=np.transpose(G, [2,3,0,1])


    F_out = C  @ G  @ Ci  @ F_in
    return F_out





# Set up x-z' array for strain fields and deviation parameters
# this is a 'generalised cross section' as used by Head & co
# the dislocation lies at the mid point of this cross section and is at an angle (90-phi) to it
# a column along z in the 3D volume maps onto a line along z' in this array
# with the same x-coordinate and a start/finish point given by the position
# of the top/bottom of the foil relative to the dislocation line at the image x-y coordinate
#sxz = np.zeros((xsiz, zsiz), dtype='f')#32-bit for .tif saving
# since the dislocation lies at an angle to this plane the actual distance to the dislocation
# in the z-coordinate is z'*sin(phi)

start_time = time.perf_counter()





############################################################################
#                    CALCULATE DEVIATIONS                                  #
############################################################################






bscrew = np.dot(b,u) #scalar
bedge = c2d @ (b - bscrew*u) #vector
beUnit = bedge/(np.dot(bedge,bedge)**0.5)#unit vector
bxu = c2d @ np.cross(b,u) #vector
gD = c2d @ g #vector

dz = 0.01 #scalar
deltaz = np.array((0, dt*dz, 0)) / pix2nm #vector
bxu = c2d @ np.cross(b,u) #vector








x_vec = np.linspace(0, xsiz-1, xsiz) + 0.5 - xsiz/2
x_mat = np.tile(x_vec, (zsiz+1, 1))
rX = np.transpose(x_mat) #matrix

z_vec = (  (np.linspace(0, zsiz, zsiz+1) + 0.5 - zsiz/2)*(np.sin(phi))
        + xsiz*np.tan(psi)/(2*zsiz)  )*dt
z_mat = np.tile(z_vec, (xsiz, 1))
rD_1 = z_mat + rX*np.tan(theta)


rD = np.dstack( #vector matrix
        (np.dstack(
                (np.dstack(
                        (np.zeros((xsiz, zsiz+1, 0)),
                         rD_1)),
                rX)),
        np.zeros((xsiz, zsiz+1, 1))))
#MUST be a nicer way of doing this


gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD )
rDdz = np.add(rD, deltaz)
gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD )
sxz = (gRdz - gR)/dz #matrix


############################################################################
#                    CALCULATE IMAGE                                       #
############################################################################

Ib = np.zeros((xsiz, ysiz), dtype='f')  # Bright field image
    # 32-bit for .tif saving
Id = np.zeros((xsiz, ysiz), dtype='f') # Dark field image
    
# Complex wave amplitudes are held in F = [BF,DF]
F0 = np.array([[1], [0]])


# centre point of simulation frame is p
p = np.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
# length of wave propagation
zlen=int(t*nS[2]/dt + 0.5)#remember nS[2]=cos(tilt angle)






F = np.tile(F0, (xsiz, ysiz, 1, 1)) #matrix of bright and dark beam values evrywhr

top_vec = np.arange(ysiz) * (zsiz-zlen)/ysiz #y vector
h_vec = top_vec.astype(int) #y vector
m_vec = top_vec - h_vec #y vector


top = np.tile(top_vec, (xsiz,1)) #xy matrix
h = top.astype(int) #xy matrix
m = top - h #xy matrix



for z in range(zlen):
    slocal = s + (1-m)*sxz[:,(h_vec+z)%zsiz]+m*sxz[:,(h_vec+z+1)%zsiz] #xy matrix
    alpha = 0.0
    F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm) #xy matrix of 2vectors

F = np.transpose(F, (2,3,0,1)).reshape(2,xsiz,ysiz)

Ib, Id = abs((F*np.conjugate(F)))




end_time = time.perf_counter()
duration = end_time - start_time
print("Main loops took: " + str(duration) + " seconds")



#%%
#####################
#PRINTING MACHINERY #
#####################

ker = int(7/pix2nm+0.5)+1
#Ib2= cv.GaussianBlur(Ib,(ker,ker),blursigma)
#Id2= cv.GaussianBlur(np.float(Id),(ker,ker),blursigma)
Ib2 = np.ndarray.tolist(Ib)#2) 
Id2 = np.ndarray.tolist(Id)#2)


fig = plt.figure(figsize=(8, 4))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib2)
plt.axis("off")
fig.add_subplot(2, 1, 2)
plt.imshow(Id2)
plt.axis("off")




