import cupy as cp
import numpy as np
#import numpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import time


from tqdm import tqdm
eps=0.000000000001


def normalise(x):
    if abs(x[0])<eps and abs(x[1])<eps and abs(x[2])<eps:
        return cp.zeros(3)
    norm = x / (cp.dot(x, x) ** 0.5)        
    return norm




def gdotR(r, bscrew, bedge, beUnit, bxu, nu, gD, xsiz, zsiz):
    rmag = (cp.sum(r*r, axis=2)**0.5) #xz-matrix
    
    rmag = (r[...,0]**2 + r[...,1]**2)**0.5
    
    ct = cp.dot(r,beUnit)/rmag #xz-matrix
    sbt = cp.cross(beUnit,r) #3vector xz-matrix
    st = sbt[...,2]/rmag #xz-matrix
    
    
    local_theta = cp.arctan(r[...,1]/r[...,0]) #xz-matrix
    Rscrew_2 = (bscrew/(2*cp.pi)) * (local_theta-cp.pi*(r[...,0]<0)).reshape(xsiz,zsiz+1,1)
    Rscrew = cp.concatenate((cp.zeros([xsiz, zsiz+1, 2]), Rscrew_2), axis=2) #3-vector xz-matrix
    Redge0 = (ct*st)[...,None] * bedge/(4*cp.pi*(1-nu)) #3vector xz-matrix
    Redge1 = ( ((2-4*nu)*cp.log(rmag)+(ct**2-st**2)) /(8*cp.pi*(1-nu)))[...,None]*bxu
    R = Rscrew + Redge0 + Redge1
    return cp.dot(R, gD)




























#CONSTANTS
    
#extinction distances
X0i = 1000.  # nm
XgR = 58


A=0.06
X0i1=X0i*(1.+A)
Xg = XgR + 1j * X0i1  # nm
Xgr = Xg.real #const
Xgi = Xg.imag #const


#lattice parameters
a = 0.4#nm
nu = 0.3
s = 0.0046
b = cp.array([-0.5, -0.5, 0])*a/6
g = cp.array([1,1,-1])/a


#lets say that nm are pixels for the moment


scale=50

t = 23 #nm
dt = 0.4 #effectively it's own unit
pad = 20 #nm
foil_slices = int( t/dt + 0.5)



z = cp.array([1,5,6])
z = normalise(z)
n = cp.copy(z)
u = cp.array([5,2,3])
u = normalise(u)

phi = cp.arccos( cp.dot(z,u) )


#simulation frame
x = normalise( cp.cross(u,z) )
y = cp.cross(z, x)
#z = z




c2s = cp.array([x, y, z])
s2c = cp.transpose(c2s)


nS = c2s @ n #[0,0,1] for normal foil
uS = c2s @ u


bS = c2s @ b #these two aren't normalised
gS = c2s @ g



#dislocation frame (in simulation frame basis)

xD = cp.array([1,0,0])
zD = cp.copy(uS)
yD = cp.cross(zD, xD)

s2d = cp.array([xD, yD, zD])
d2s = cp.transpose(s2d)

xd = s2d @ cp.array([1,0,0])
yd = s2d @ cp.array([0,1,0])
zd = s2d @ cp.array([0,0,1])
nD = s2d @ nS
uD = s2d @ uS #[0,0,1] by definition



bD = s2d @ bS #these two aren't normalised
gD = s2d @ gS







pad = pad
ypad = abs(t*cp.tan(phi))
zpad = abs(pad/cp.tan(phi))


xsiz = int(2*pad + 0.5)
ysiz = int( xsiz + ypad)
zsiz = 2* int((t + zpad)/dt + 0.5) #total number of vertical pixels in the simulation


print("xsiz,ysiz,zsiz=", xsiz, ysiz, zsiz)
print("foil_slices =", foil_slices)
print("zpad_slices =", int(zpad/dt))
print("dt =", dt)






start_time = time.perf_counter()

#ALL IN DISLOCATION FRAME

bscrew = cp.dot(bD,uD) #scalar COMPONENT OF b||u
bedge = (bD - bscrew*uD) #vector COMPONENT OF bTu
if cp.linalg.norm(bedge)<eps:
    beUnit = cp.array([1,0,0])
else:
    beUnit = normalise(bedge) #unit vector
bxu = cp.cross(bD,uD) #vector







dz = 0.01 #scalar
deltaz = cp.array((0, dt*dz, 0))


x_vec = cp.linspace(-xsiz/2+0.5, xsiz/2-0.5, xsiz) #+ 0.5 - xsiz/2
z_vec = cp.linspace(- zsiz/2 , zsiz/2, zsiz+1)


x_mat = cp.tile(x_vec, (zsiz+1, 1)) #zx-matrix
z_mat = cp.tile(z_vec, (xsiz, 1)) #xz-matrix



rD_x = (cp.transpose(x_mat)).reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)
rD_y = -cp.sin(phi) * z_mat.reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)
rD_z = cp.cos(phi) * z_mat.reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)




rD = cp.concatenate((rD_x, rD_y, rD_z), axis=2) #THIS IS IN DISLOCATION FRAME!!









gR = gdotR(rD, bscrew, bedge, beUnit, bxu, nu, gD, xsiz, zsiz)

rDdz = cp.add(rD, deltaz)

gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, nu, gD, xsiz, zsiz)
sxz = (gRdz - gR)/dz

plt.imshow(cp.asnumpy(sxz))



















def howieWhelan(F_in,s,Xg,X0I,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real
    Xgi = Xg.imag
    #print(X0I)
    #s = s + eps

    gamma = cp.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])

    q = cp.array([(0.5/X0I)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0I)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])
    q = cp.array([(0.5/X0I),  (0.5/X0I)])
    beta = cp.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))

    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)],
                [-cp.sin(beta/2),
                 cp.cos(beta/2)]])

    #inverse of C is just its transpose
    #Ci=cp.transpose(C)

    G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]], dtype=complex)
    
    C = cp.transpose(C, [2,3,0,1])
    Ci = cp.transpose(C, [0,1,2,3])
    G = cp.transpose(G, [2,3,0,1])
    
    F_out = C @ G @ Ci @ F_in
    

    Ib1 = cp.squeeze(F_out[...,0,:])
    Id1 = cp.squeeze(F_out[...,1,:])
    
    
    Ib = abs(Ib1*cp.conjugate(Ib1))
    Id = abs(Id1*cp.conjugate(Id1))
    return Ib, Id, F_out











F0 = cp.array([[1], [0]])
F = cp.tile(F0, (xsiz, ysiz, 1, 1)) # bright and dark field at the top of each collumn





top = cp.arange(ysiz) * (zsiz-t)/(ysiz-1)
top_int = top.astype(int)
diff = top - top_int



'''
sxzbulk = cp.zeros((xsiz,ysiz, foil_slices))

for yy in range(ysiz):
    relevant_section = sxz[:,top_int[yy]:top_int[yy]+foil_slices]
    sxzbulk[:,yy,:] = relevant_section
'''


for zz in tqdm(range(foil_slices)):
    slocal = s + (1-diff)*sxz[:,top_int+zz] + diff*sxz[:, top_int+zz+1] #the reason for zsiz+1
    Ib, Id, F = howieWhelan(F, slocal, Xg, X0i, dt)





Ib[0,0], Id[0,0], Ib[0,1], Id[0,1] = 1,1,0,0

norm_factor = cp.amax(Ib)

#Ib = Ib/norm_factor
#Id = Id/norm_factor


fig = plt.figure(figsize=(16, 10))
fig.add_subplot(2, 1, 1)
plt.imshow(cp.asnumpy(Ib))
fig.add_subplot(2, 1, 2)
plt.imshow(cp.asnumpy(Id))


plt.savefig("hhh.png")



'''
#sxzbulk = cp.transpose(sxzbulk/cp.amax(sxzbulk), (1,2,0))
sxzbulk = cp.asnumpy(sxzbulk/cp.amax(sxzbulk))




fig = plt.figure(figsize=(4,4))

space = cp.asnumpy(cp.array([*product(range(xsiz), range(ysiz), range(foil_slices))])) # all possible triplets of
ax = fig.add_subplot(111, projection='3d')
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel("z")
ax.scatter(space[:,0], space[:,1], space[:,2], s=0.00000000000001*sxzbulk**-3)



ax.view_init(0, 0)
#yz (0,0)
#xz (0,90)
#xy (90,90)

#plt.savefig("help.png")
'''




