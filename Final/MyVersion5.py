
import cupy as cp 
import matplotlib.pyplot as plt
import time
#import cv2 as cv
import inputs as inn
from tqdm import tqdm
eps=0.000000000001




#INPUTS


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




def normalise(x):
    norm = x / (cp.dot(x, x) ** 0.5)        
    return norm





# SETUP



# Normalise the vectors
u = normalise(u)
z = normalise(z)
n = normalise(n)
#print(1000000u, 1000000*z, 1000000*n)


# we want n pointing to the same side of the foil as z
if cp.dot(n, z) < 0:  # they're antiparallel(ish), reverse n
    n = -n
# also we want u pointing to the same side of the foil as z
if cp.dot(u, z) < 0:  # they're antiparallel(ish), reverse u and b
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

if abs(cp.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    #Think will not work, needs a different approach to the calculation
    x = b0[:]
    x = x / (cp.dot(x, x) ** 0.5)
    if abs(cp.dot(x, z) - 1) < eps:  # they're parallel too, set x parallel to g instead
        x = g[:]  # this will fail for u=z=b=g but that would be stupid
        if abs(cp.dot(x, z) - 1) < eps:  # they're parallel too, set x arbitrarily
            x = cp.array([1, 0, 0])  
    phi=0.0 # angle between dislocation and z-axis
else:
    x = cp.cross(u, z)
    phi = cp.arccos(abs(cp.dot(u, z)))
    # angle between dislocation and z-axis

x = normalise(x)
y = cp.cross(z, x) # y = z cross x



# transformation matrices between simulation frame & crystal frame (co-ordinates system)
c2s = cp.array((x, y, z))
s2c = cp.transpose(c2s)












# normal and dislocation vectors transform to simulated co-ordinate frame

nS = c2s @ n

uS = c2s @ u

# some useful vectors
n1, n2 = cp.copy(n), cp.copy(n)
n1[0], n2[1] = 0, 0

#they are unit vectors
n1 = normalise(n1)
n2 = normalise(n2)

#angle between n1 and z; foil tilt along the dislocation
psi = cp.arccos(cp.dot(n1,z))*cp.sign(n[1])
#angle between n2 and z; foil tilt perpendicular to the dislocation
theta = cp.arccos(cp.dot(n2,z))*cp.sign(n[0])










# Crystal<->Dislocation frames
# dislocation frame has zD parallel to u & xD parallel to x
# yD is given by their cross product
xD = cp.copy(x)
yD = cp.cross(u, x)
zD = cp.copy(u)


# transformation matrix between crystal frame & dislocation frame
c2d = cp.array((xD, yD, zD))
d2c = cp.transpose(c2d)


#%%


# Set up simulation frame (see Fig.A)

# x=((1,0,0)) is vertical up
# y=((0,1,0)) is horizontal left, origin bottom right
# z=((0,0,1)) is into the image








#####################
#CHECK THIS SECTION #
#####################






# FINAL IMAGE dimensions:

# along x: (note this is an even number; the dislocation line is between pixels)
xsiz = 2*int(pad + 0.5) # in pixels
# along y: ysiz = xsiz + dislocation image length (to be calculated below)
# along z: zsiz = t/dt + vertical padding (to be calculated below)



# extra height to account for the parts of the image to the left and right of the dislocation
hpad = xsiz/cp.tan(phi)
# y dimension calculation
if abs(cp.dot(u, z)) < eps:  # dislocation is in the plane of the foil
    
    #in this scenario the dislocation projection is infinite
    #so choose to make grid square
    ysiz = 1 * xsiz # in pixels
    zsiz = zlen # in slices
    print("Dislocation is in the plane of the foil")
    #the plane normal to z isn't necessarily the foil plane is it??
    #perpendicular to beam?


elif abs(cp.dot(u, z)-1) < eps:  # dislocation along z
    #needs work?
    
    #likewise here, projection is zero os choose a square
    ysiz = 1 * xsiz
    zsiz = zlen # in slices
    print("Dislocation is parallel to the beam")


else:  # dislocation is at an angle to the beam direction
    # dislocation image length
    w = int(t*nS[2]*nS[2]*uS[1]/abs(cp.dot(u,n1)) + 0.5)
    ysiz = w + xsiz # in pixels
    zsiz = int( (2*t*nS[2] + hpad + xsiz*cp.tan(abs(psi)) )/dt + 0.5) # in slices



print("(xsiz,ysiz,zsiz)=", xsiz,ysiz,zsiz) #x value seems bigger than it should be?
#xsiz, ysiz, zsiz = 10, 20, 10




######################
#/CHECK THIS SECTION #
######################






# Set up x-z' array for strain fields and deviation parameters
# this is a 'generalised cross section' as used by Head & co
# the dislocation lies at the mid point of this cross section and is at an angle (90-phi) to it
# a column along z in the 3D volume maps onto a line along z' in this array
# with the same x-coordinate and a start/finish point given by the position
# of the top/bottom of the foil relative to the dislocation line at the image x-y coordinate
#sxz = cp.zeros((xsiz, zsiz), dtype='f')#32-bit for .tif saving
# since the dislocation lies at an angle to this plane the actual distance to the dislocation
# in the z-coordinate is z'*sin(phi)














######################
#    Functions       #
######################



def gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD):
    '''
    rD = 3vector xz-matrix
    bscrew = scalar
    bedge = 3vector
    beUnit = unit 3vector
    bxu = 3vector
    d2c = 3matrix
    nu = scalar
    gD = 3vector
    
    dz = scalar
    deltaz = 3vector
    bxu = 3vector
    
    '''
    
    rmag = cp.sum(rD*rD, axis=2)**0.5 #xz-matrix
    
    ct = cp.dot(rD,beUnit)/rmag #xz-matrix
    sbt = cp.cross(beUnit,rD) #3vector xz-matrix
    #st = cp.dsplit(sbt, (0,1,2))[3].reshape(xsiz, zsiz+1) 
    #st = cp.transpose(sbt, (2,0,1))[2]/rmag #xz-matrix
    st = sbt[...,2]/rmag #xz-matrix
    #both valid. One's much more comprehesible though
    #now with even more comprehensible version
    
    Rscrew_2 = bscrew*(cp.arctan(rD[...,1].reshape(cp.shape(rX))/rX)-cp.pi*(rX<0))/(2*cp.pi)
               #component 3 (z) of 3vector xz-matrix
    #Rscrew = cp.dstack((cp.zeros((xsiz, zsiz+1, 2)), Rscrew_2)) 
    Rscrew = cp.concatenate((cp.zeros((xsiz, zsiz+1, 2)), Rscrew_2), axis=2)
             #3vector xz-matrix
    Redge0 = (ct*st)[...,None] * bedge/(2*cp.pi*(1-nu)) #3vector xz-matrix
    #[...,None] pushes everything up a dimension
    
    Redge1 = ( ((2-4*nu)*cp.log(rmag)+(ct**2-st**2)) /(8*cp.pi*(1-nu)))[...,None]*bxu
             #3vector xz-matrix
    R = (Rscrew + Redge0 + Redge1) #3vector xz-matrix
    gR = cp.dot(R, gD) #xz-matrix
    return gR


def howieWhelan(F_in,Xg,X0i,slocal,alpha,t):
    '''
    X0i  scalar
    Xg = scalar
    s =  xy matrix
    alpha = scalar
    t = scalar
    '''
    
    
    
    #for integration over each slice
    # All dimensions in nm
    Xgr = Xg.real #const
    Xgi = Xg.imag #const

    slocal = slocal + eps #xy matrix
    #print(slocal)
    gamma = cp.array([(slocal-(slocal**2+(1/Xgr)**2)**0.5)/2, (slocal+(slocal**2+(1/Xgr)**2)**0.5)/2]) #2vector of gamma xy matrices
    #print(gamma)    
    q = cp.array([(0.5/X0i)-0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5))]) #2vector of q xy matrices
    #print(q) 
    beta = cp.arccos((slocal*Xgr)/((1+slocal**2*Xgr**2)**0.5)) #xy matrix
    #print(beta)
    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)], #2matrix of xy matrices
                 [-cp.sin(beta/2)*cp.exp(complex(0,alpha)),
                  cp.cos(beta/2)*cp.exp(complex(0,alpha))]])
    #C = cp.transpose(C, (2,3,0,1))
    print(C)
    #inverse of C is likewise a 2matrix of xy matrices
    Ci= cp.array([[cp.cos(beta/2), -cp.sin(beta/2)*cp.exp(complex(0,-alpha))],
                 [cp.sin(beta/2),  cp.cos(beta/2)*cp.exp(complex(0,-alpha))]])


    G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]])
    #gamma/q[0/1] are all xy matrices
    #thus this is a 2matrix of xy matrices
    #0*gamma[0] give these zeroes the right dimensionality
    #cp.zeros probably better
    
    
    
    gamma = cp.transpose(gamma, (1,2,0)).reshape(xsiz,ysiz,2,1) #xy matrix of gamma 2vectors
    q = cp.transpose(q, (1,2,0)).reshape(xsiz,ysiz,2,1) #xy matrix of q 2vectors

    C=cp.transpose(C, [2,3,0,1])
    Ci=cp.transpose(Ci, [2,3,0,1])
    G=cp.transpose(G, [2,3,0,1])


    F_out = C  @ G  @ Ci  @ F_in
    return F_out













############################################################################
#                    CALCULATE DEVIATIONS                                  #
############################################################################
start_time = time.perf_counter()





bscrew = cp.dot(b,u) #scalar
bedge = c2d @ (b - bscrew*u) #vector

beUnit = bedge/(cp.dot(bedge,bedge)**0.5)#unit vector
bxu = c2d @ cp.cross(b,u) #vector
gD = c2d @ g #vector

dz = 0.01 #scalar
deltaz = cp.array((0, dt*dz, 0)) / pix2nm #vector
bxu = c2d @ cp.cross(b,u) #vector








x_vec = cp.linspace(0, xsiz-1, xsiz) + 0.5 - xsiz/2 #x-vector
x_mat = cp.tile(x_vec, (zsiz+1, 1)) #zx-matrix
rX = (cp.transpose(x_mat)).reshape(xsiz,zsiz+1,1) #xz-matrix

z_vec = (  (cp.linspace(0, zsiz, zsiz+1) + 0.5 - zsiz/2)*(cp.sin(phi)
        + xsiz*cp.tan(psi)/(2*zsiz))  )*dt #z-vector
z_mat = cp.tile(z_vec, (xsiz, 1)) #xz-matrix
rD_1 = z_mat.reshape(xsiz,zsiz+1,1) + rX*cp.tan(theta) #xz matrix (scalar)

rD = cp.concatenate((rX, rD_1, cp.zeros((xsiz, zsiz+1,1))), axis=2) /pix2nm
     #3vector xz-matrix


gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD) #xz-matrix
rDdz = cp.add(rD, deltaz) #3vector xz-matrix
gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD ) #xz-matrix
sxz = (gRdz - gR)/dz #xz-matrix

#shouldn't this step be iterated???


############################################################################
#                    CALCULATE IMAGE                                       #
############################################################################

Ib = cp.zeros((xsiz, ysiz), dtype='f')  # Bright field image
    # 32-bit for .tif saving
Id = cp.zeros((xsiz, ysiz), dtype='f') # Dark field image
    
# Complex wave amplitudes are held in F = [BF,DF]
F0 = cp.array([[1], [0]]) #faster than cp.array([1,0])[...,None]


# centre point of simulation frame is p
p = cp.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
# length of wave propagation
zlen=int(t*nS[2]/dt + 0.5)#remember nS[2]=cos(tilt angle)






F = cp.tile(F0, (xsiz, ysiz, 1, 1)) #matrix of bright and dark beam values evrywhr

top_vec = cp.arange(ysiz) * (zsiz-zlen)/ysiz #y vector
h_vec = top_vec.astype(int) #y vector
m_vec = top_vec - h_vec #y vector


top = cp.tile(top_vec, (xsiz,1)) #xy matrix
h = top.astype(int) #xy matrix
m = top - h #xy matrix



for z in tqdm(range(zlen)):
    slocal = s + (1-m)*sxz[:,(h_vec+z)%zsiz]+m*sxz[:,(h_vec+z+1)%zsiz] #xy matrix
    alpha = 0.0
    F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm) #xy matrix of 2vectors

F = cp.transpose(F, (2,3,0,1)).reshape(2,xsiz,ysiz)

Ib, Id = abs((F*cp.conjugate(F)))




end_time = time.perf_counter()
duration = end_time - start_time
print("Main loops took: " + str(duration) + " seconds")



#%%
#####################
#PRINTING MACHINERY #
#####################

ker = int(7/pix2nm+0.5)+1
#Ib2= cv.GaussianBlur(Ib,(ker,ker),blursigma)
#Id2= cv.GaussianBlur(cp.float(Id),(ker,ker),blursigma)
Ib2 = cp.ndarray.tolist(Ib)#2) 
Id2 = cp.ndarray.tolist(Id)#2)


fig = plt.figure(figsize=(16, 8))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib2)
plt.axis("off")
fig.add_subplot(2, 1, 2)
plt.imshow(Id2)
plt.axis("off")




