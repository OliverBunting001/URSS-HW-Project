import cupy as cp 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import inputs4 as inn
import sys
eps=0.000000000001

X0i=inn.X0i #nm
Xg=inn.Xg #nm
a0 = inn.a0 #nm
nu = inn.nu
t0 = inn.t0 #nm
dt = inn.dt #pixels
s = inn.s


n = cp.array(inn.n) #miller indices - units of a0
z = cp.array(inn.z) #miller indices
u = cp.array(inn.u) #miller indices

g = cp.array(inn.g) #miller indices
b0 = cp.array(inn.b0) #unitless

g = g / a #unitless
b = b0 * a #miller idices


pix2nm = inn.pix2nm #nm/pix
nm2pix = 1/pix2nm #pix/nm
pad = inn.pad #pix
blursigma = inn.blursigma

# scale dimensions - change units to pixels
blursigma = blursigma * nm2pix
t = t0 * nm2pix
pad = pad * nm2pix
a = a0 * nm2pix
X0i = X0i * nm2pix
Xg = Xg * nm2pix



no_slices = int(t/dt + 0.5)

def normalise(x):
    #function for normalising vectors
    norm = x / (cp.dot(x, x) ** 0.5)        
    return norm


#immediately normalise the input vectors
#so we can work with only unit vectors
    
u = normalise(u)
z = normalise(z)
n = normalise(n)

#also want all pointing 'down'
if cp.dot(n, z) < 0:  # they're antiparallel, reverse n
    n = -n
if cp.dot(u, z) < 0:  # they're antiparallel, reverse u and b
    u = -u
    b0 = -b0

'''
#############################################################################
#run some tests to clarify our scenario                                     #
                                                                            #
#n-z                                                                        #
if abs(cp.dot(n, z)-1) < eps: #parallel                                     #
    foilnormal=True                                                         #
    print("Incident beam is normal to foil")                                #
elif cp.dot(n, z) < eps: #perpendicular                                     #
    print("Beam and foil are parallel - this won't work")                   #
    sys.exit()                                                              #
                                                                            #
#u-z                                                                        #
if abs(cp.dot(u, z)-1) < eps: #parallel                                     #
    beam_defect_aligned=True                                                #
    print("Beam and defect are aligned")                                    #
elif cp.dot(u, z) < eps: #perpendicular                                     #
    beam_defect_normal=True                                                 #
    print("Defect is perpendicular to a beam")                              #
                                                                            #
#u-n                                                                        #
if abs(cp.dot(u, n)-1) < eps: #parallel                                     #
    defect_into_foil=True                                                   #
    print("Defect perpendicularly into the foil")                           #
elif cp.dot(u, n) < eps: #perpendicular                                     #
    defect_in_foil_plane=True                                               #
    print("Defect is in the plane of the foil")                             #
#############################################################################
'''


#SET UP SIMULATION FRAME

#beam direction (z) is the z-direction
#take the z-component out of u and this will define the y-direction
#x-component is the cross product of u and z


if abs(cp.dot(u, z)-1) < eps:
    #doesn't work
    print("x cannot be cross-product of z and u")
    #SET UP AN ALTERNATIVE
    phi=0.
else:
    x = cp.cross(z,u)
    x=normalise(x)
    phi = cp.arccos(cp.dot(z,u))

y=cp.cross(z,x)
y=normalise(u-z*cp.dot(u,z))
#the same thing


c2s = cp.array((x, y, z))
s2c = cp.transpose(c2s)

n = c2s @ n
u = c2s @ u



xsiz = int(2*pad + 0.5)
#ysiz = xsiz + w
#zsiz = (t+hpad)/dt


#foil defined by planes r.n=0 and r.n=t (in simulation frame)
#defect meets bottom plane at the origin
#therefor at meets the top plane at:
r = (t/cp.dot(u,n)) * u
l = t/cp.dot(u,n) #length of defect within the foil
#definitionally r has no x component, want only the y-component as w
w=int( r[1] + 0.5 )
ysiz = xsiz + w






#########################################


#different amounts of vertical padding are required if the foil is tilted

n1, n2 = cp.copy(n), cp.copy(n)
n1[0], n2[1] = 0, 0

ytilt = cp.arccos(cp.dot(n1,z))
xtilt = cp.arccos(cp.dot(n2,z))
#each of these tiilt angles will raise the height of the at pad
#along  the relevant axis.
#to get the total height, add in quadrature
yhpad = pad * cp.tan(ytilt)
xhpad = pad * cp.tan(xtilt)






#############################################
#COULD USE A DIFFERENT APPROACH OF SOLVING THE FOIL PLANE EQUATIONS
#AT THE DEFECT INTERCEPTS +=PAD IN X/Y DIRECTIONS
















zsiz = int( (t+hpad)/dt + 0.5)



print("xsiz, ysiz, zsiz =", xsiz, ysiz, zsiz)














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
    #st = cp.dsplit(sbt, (0,1,2))[3].reshape(xsiz, zsiz+1)/rmag
    #st = cp.transpose(sbt, (2,0,1))[2]/rmag #xz-matrix
    st = sbt[...,2]/rmag #xz-matrix
    #All valid - think this one's best
    
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
    gamma = cp.array([(slocal-(slocal**2+(1/Xgr)**2)**0.5)/2, (slocal+(slocal**2+(1/Xgr)**2)**0.5)/2]) #2vector of gamma xy matrices
    q = cp.array([(0.5/X0i)-0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5))]) #2vector of q xy matrices
    beta = cp.arccos((slocal*Xgr)/((1+slocal**2*Xgr**2)**0.5)) #xy matrix
    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)], #2matrix of xy matrices
                 [-cp.sin(beta/2)*cp.exp(complex(0,alpha)),
                  cp.cos(beta/2)*cp.exp(complex(0,alpha))]])
    #inverse of C is likewise a 2matrix of xy matrices
    Ci= cp.array([[cp.cos(beta/2), -cp.sin(beta/2)*cp.exp(complex(0,-alpha))],
                 [cp.sin(beta/2),  cp.cos(beta/2)*cp.exp(complex(0,-alpha))]])
    #Ci = cp.transpose(C, (1,0,2,3)) #wrong but works for alpha=0

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
rX = (cp.transpose(x_mat)).reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)

z_vec = (  (cp.linspace(0, zsiz, zsiz+1) + 0.5 - zsiz/2)*(cp.sin(phi)
        + xsiz*cp.tan(psi)/(2*zsiz))  )*dt #z-vector
z_mat = cp.tile(z_vec, (xsiz, 1)) #xz-matrix
rD_1 = z_mat.reshape(xsiz,zsiz+1,1) + rX*cp.tan(theta) #xz matrix (scalar)
rD_2 = cp.zeros((xsiz, zsiz+1,1))
rD = cp.concatenate((rX, rD_1, rD_2), axis=2) /pix2nm
     #3vector xz-matrix


gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD) #xz-matrix
rDdz = cp.add(rD, deltaz) #3vector xz-matrix
gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD ) #xz-matrix
sxz = (gRdz - gR)/dz #xz-matrix




############################################################################
#                    CALCULATE IMAGE                                       #
############################################################################

Ib = cp.zeros((xsiz, ysiz))  # Bright field image
    # 32-bit for .tif saving
Id = cp.zeros((xsiz, ysiz)) # Dark field image
    
# Complex wave amplitudes are held in F = [BF,DF]
F0 = cp.array([[1], [0]])
#F0 = cp.array([1,0])[...,None]


# centre point of simulation frame is p
p = cp.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
# length of wave propagation
no_slices=int(t*nS[2]/dt + 0.5)#remember nS[2]=cos(tilt angle)






F = cp.tile(F0, (xsiz, ysiz, 1, 1)) #matrix of bright and dark beam values evrywhr

top_vec = cp.arange(ysiz) * (zsiz-no_slices)/ysiz #y vector
h_vec = top_vec.astype(int) #y vector
m_vec = top_vec - h_vec #y vector


top = cp.tile(top_vec, (xsiz,1)) #xy matrix
h = top.astype(int) #xy matrix
m = top - h #xy matrix



for z in tqdm(range(no_slices)):
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


fig = plt.figure(figsize=(16, 4))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib2)
plt.axis("off")
fig.add_subplot(2, 1, 2)
plt.imshow(Id2)
plt.axis("off")














'''

#Dislocation frame
xD = cp.copy(x)
yD = cp.cross(u, x)
zD = cp.copy(u)



c2s = cp.array((x, y, z))
s2c = cp.transpose(c2s)
c2d = cp.array((xD, yD, zD))
d2c = cp.transpose(c2d)






#set up some vectors

n1, n2 = cp.copy(n), cp.copy(n)
n1[0], n2[1] = 0, 0
print(n1,n2)

#instead remove simulation x and y components
n1 = n -  cp.dot(n, x)*x
n2 = n - cp.dot(n, y)*y
#Don't know which version is right?



#they are unit vectors
n1 = normalise(n1)
n2 = normalise(n2)







#angle between n1 and z; foil tilt along the dislocation
psi = cp.arccos(cp.dot(n1,z))*cp.sign(n[1])
print("psi=", psi)
#angle between n2 and z; foil tilt perpendicular to the dislocation
theta = cp.arccos(cp.dot(n2,z))*cp.sign(n[0])
print("theta=", theta)





#Simulation Frame Size

xsiz = 2*int(pad + 0.5) # in pixels



hpad = xsiz/cp.tan(phi) #padding on top AND bottom hence using xsiz rather than pad


if abs(cp.dot(u, z)) < eps or abs(cp.dot(u, z)-1) < eps :
    #if u,z parallel/perpendicular
    w=0
    zsiz = no_slices
else:
    #w = int(t*nS[2]*nS[2]*uS[1]/abs(cp.dot(u,n1)) + 0.5)
    #w = int((t/abs(cp.dot(u,n1)) * nS[2]*nS[2]*uS[1]) + 0.5)
    un = cp.dot(u,n)
    uz = cp.dot(u,z)
    #w = int( (t/un)*u - (t/un)*uz*z + 0.5)
    #w = int(t*cp.tan(phi) + 0.5)
    #zsiz = int( (2*t*nS[2] + hpad + xsiz*cp.tan(abs(psi)) )/dt + 0.5) # in slices
ysiz = xsiz+w
print("(xsiz,ysiz,zsiz)=", xsiz,ysiz,zsiz)

'''


















