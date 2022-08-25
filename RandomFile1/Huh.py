import cupy as cp
import numpy as np
import numpy as cp
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
    

    #ct = r[...,0]/rmag #xz-matrix
    #st = r[...,1]/rmag

    ct = cp.dot(r,beUnit)/rmag #xz-matrix
    sbt = cp.cross(beUnit,r) #3vector xz-matrix

    st = sbt[...,2]/rmag #xz-matrix
    
    
    local_theta = cp.arctan(r[...,1]/r[...,0]) #xz-matrix
    Rscrew_2 = (bscrew/(2*cp.pi)) * (local_theta-cp.pi*(r[...,0]<0)).reshape(xsiz,zsiz+1,1)
    Rscrew = cp.concatenate((cp.zeros((xsiz, zsiz+1, 2)), Rscrew_2), axis=2) #3-vector xz-matrix
    Redge0 = (ct*st)[...,None] * bedge/(4*cp.pi*(1-nu)) #3vector xz-matrix
    Redge1 = ( ((2-4*nu)*cp.log(rmag)+(ct**2-st**2)) /(8*cp.pi*(1-nu)))[...,None]*bxu
    R = Rscrew + Redge0 + Redge1
    return cp.dot(R, gD)


#CONSTANTS
    
#extinction distances
X0i = 1000.  # nm
XgR = 10


A=0.06
Xg = XgR + 1j * X0i * (1.+A)  # nm
Xgr = Xg.real #const
Xgi = Xg.imag #const


#lattice parameters
a = 0.3#nm
nu = 0.3
s = 0.1
b = cp.array([0, 0.5, 0.5])*a
g = cp.array([1,1,1])/a


#lets say that nm are pixels for the moment


scale=50

t = 50 #nm
dt = 1 #effectively it's own unit
pad = scale #nm
foil_slices = int( t/dt + 0.5)



z = cp.array([3,4,1])
z = normalise(z)
n = cp.copy(z)
u = cp.array([3,5,4])
u = normalise(u)

phi = cp.arccos( cp.dot(z,u) )


#simulation frame
x = normalise( cp.cross(u,z) )
y = cp.cross(z, x)
#z = z




c2s = cp.array((x, y, z))
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
ypad = t*cp.tan(phi)
zpad = pad/cp.tan(phi)


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
rD_z = cp.zeros((xsiz,zsiz+1,1)) #just want radial distance to the dislocation



rD = cp.concatenate((rD_x, rD_y, rD_z), axis=2) #THIS IS IN DISLOCATION FRAME!!
                                                #or meant to be




'''
xco = cp.random.randint(0, xsiz-1, 2)
zco = cp.random.randint(0, zsiz, 2)
point1 = rD[xco[0],zco[0]]
point2 = rD[xco[1],zco[1]]
#vector in xz (in dislocation frame)
vector = point2 - point1
print(point1,point2)
vector=normalise(vector)
print("FINAL!!!", cp.dot(vector,yd))
'''







gR = gdotR(rD, bscrew, bedge, beUnit, bxu, nu, gD, xsiz, zsiz)

rDdz = cp.add(rD, deltaz)

gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, nu, gD, xsiz, zsiz)
sxz = (gRdz - gR)/dz
plt.imshow(sxz)

#plt.imshow(cp.asnumpy(cp.log(abs(sxz))))
#print(sxz[:,0])





def HowieWhelan(F_in, ss, XGr, XGi, X0i, t,z): #ss takes the role of slocal for the avoidance of ambiguity
    
    sXsqr = ss**2 * XGr**2
    
    beta = cp.arccos( cp.sqrt( sXsqr/(1+sXsqr) ) )
    
    #gamma = 0.5* cp.array( (ss - 1/XGr * cp.sqrt(1+sXsqr)) , (ss + 1/XGr * cp.sqrt(1+sXsqr)) )
    
    if z==0:
        print(cp.shape(ss))

    ss = ss + eps #xy matrix
    gamma = cp.array([(ss-(ss**2+(1/XGr)**2)**0.5)/2, (ss+(ss**2+(1/XGr)**2)**0.5)/2]) #2vector of gamma xy matrices
    q = cp.array([(0.5/X0i)-0.5/(XGi*((1+(ss*XGr)**2)**0.5)),  (0.5/X0i)+0.5/(XGi*((1+(ss*XGr)**2)**0.5))]) #2vector of q xy matrices
    beta = cp.arccos((ss*XGr)/((1+ss**2*XGr**2)**0.5)) #xy matrix
    
    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)],
                [-cp.sin(beta/2),cp.cos(beta/2)]])
    Ci= cp.array([[cp.cos(beta/2), -cp.sin(beta/2)], [cp.sin(beta/2),  cp.cos(beta/2)]])
    #Ci = cp.transpose(C, [1,0,2,3])
    G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]])

    C=cp.transpose(C, [2,3,0,1])
    Ci=cp.transpose(Ci, [2,3,0,1])
    G=cp.transpose(G, [2,3,0,1])


    F_out = C  @ G  @ Ci  @ F_in
    return F_out




F0 = cp.array([[1], [0]])
F = cp.tile(F0, (xsiz, ysiz, 1, 1)) # bright and dark field at the top of each collumn





top = cp.arange(ysiz) * (zsiz-t)/(ysiz-1)
top_int = top.astype(int)
diff = top - top_int




sxzbulk = cp.zeros((foil_slices, xsiz,ysiz))

bigIarr = np.zeros((2,xsiz,ysiz, foil_slices))



for zx in tqdm(range(foil_slices)):
    slocal = s + (1-diff)*sxz[:,top_int+zx] + diff*sxz[:, top_int+zx+1] #the reason for zsiz+1
    F = HowieWhelan(F, slocal, Xgr, Xgi, X0i, t,zx)
    I = abs((F*cp.conjugate(F)))
    #bigIarr[...,zx] =cp.asnumpy(cp.transpose(I, (3,2,0,1))).reshape(2,xsiz,ysiz)
    
    sxzbulk[zx] = slocal

    
F = cp.transpose(F, (2,3,0,1)).reshape(2,xsiz,ysiz)
Ib, Id = abs((F*cp.conjugate(F)))

fig = plt.figure(figsize=(16,10))

plt.imshow(Ib)


sxzbulk = cp.transpose(sxzbulk/cp.amax(sxzbulk), (1,2,0))
#sxzbulk = cp.asnumpy(sxzbulk)


fig = plt.figure(figsize=(4,4))

'''
space = cp.asnumpy(cp.array([*product(range(xsiz), range(ysiz), range(foil_slices))])) # all possible triplets of
ax = fig.add_subplot(111, projection='3d')
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel("z")
ax.scatter(space[:,0], space[:,1], space[:,2], c=sxzbulk, s=1*sxzbulk**-1)



ax.view_init(90, 90)
#yz (0,0)
#xz (0,90)
#xy (90,90)

plt.savefig("help.png")
'''




