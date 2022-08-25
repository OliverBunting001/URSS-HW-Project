import cupy as cp 
import matplotlib.pyplot as plt
import time


from tqdm import tqdm
eps=0.000000000001


def normalise(x):
    norm = x / (cp.dot(x, x) ** 0.5)        
    return norm


#CONSTANTS
    
#extinction distances
X0i = 1000.  # nm
XgR = 10


A=40
Xg = XgR + 1j * X0i * (1.+A)  # nm
Xgr = Xg.real #const
Xgi = Xg.imag #const


#lattice parameters
a = 0.3#nm
nu = 0.3
s = 0.3
b = cp.array([0.5, 0.5, 0.])*a
g = cp.array([1,1,1])/a


#lets say that nm are pixels for the moment




t = 100 #nm
dt = 1 #effectively it's own unit
pad = 100 #nm




z = cp.array([3,4,1])
z = normalise(z)
n = cp.copy(z)
u = cp.array([3,5,9])
u = normalise(u)

phi = cp.arccos( cp.dot(z,u) )
print(phi)

#simulation frame
x = normalise( cp.cross(u,z) )
y = cp.cross(z, x)
z = z

c2s = cp.array((x, y, z))
s2c = cp.transpose(c2s)


nS = c2s @ n #[0,0,1] for normal foil
uS = c2s @ u


bS = c2s @ b #these two aren't normalised
gS = c2s @ g



#dislocation frame (in simulation frame basis)

xD = cp.array([1,0,0])
yD = cp.cross(uS, xD)
zD = cp.copy(uS)


s2d = cp.array([xD, yD, zD])
d2s = cp.transpose(s2d)
#this gives phi
#These are rotation matrices about the x axis by +/- phi

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
zsiz = 2* int((t + zpad)/dt + 0.5) #number of slices in z domain
foil_slices = int(t/dt)

print("xsiz,ysiz,zsiz=", xsiz, ysiz, zsiz)
print("tot_slices =", tot_slices)
print("foil_slices =", foil_slices)
print("zpad_slices =", int(zpad/dt))







start_time = time.perf_counter()

#ALL IN DISLOCATION FRAME

bscrew = cp.dot(bD,uD) #scalar COMPONENT OF b||u
bedge = (bD - bscrew*uD) #vector COMPONENT OF bTu
beUnit = normalise(bedge) #unit vector
bxu = cp.cross(bD,uD) #vector

dz = 0.01 #scalar
deltaz = cp.array((0, dt*dz, 0))








x_vec = cp.linspace(0, xsiz-1, xsiz) + 0.5 - xsiz/2 
z_vec = cp.linspace(0, zsiz, zsiz+1) - zsiz/2 

x_mat = cp.tile(x_vec, (zsiz+1, 1)) #zx-matrix
z_mat = cp.tile(z_vec, (xsiz, 1)) #xz-matrix


rD_x = (cp.transpose(x_mat)).reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)
rD_y = cp.sin(phi) * z_mat.reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)
rD_z = cp.cos(phi) * z_mat.reshape(xsiz,zsiz+1,1) #xz-matrix (ready for stacking)

rD = cp.concatenate((rD_x, rD_y, rD_z), axis=2) #THIS IS IN DISLOCATION FRAME!!
print(rD)


# new strategy for rD:
#construct desired simulation x-z plane and multiply by the basis changing matrix
'''
rD_x = cp.transpose(cp.tile(cp.arange(xsiz), (zsiz+1,1))).reshape(xsiz,zsiz+1,1)
rD_y = cp.ones((xsiz,zsiz+1,1)) * zsiz/2
rD_z = cp.tile(cp.linspace(0, zsiz, zsiz+1)-zsiz/2, (xsiz,1)).reshape(xsiz,zsiz+1,1)
rD = cp.concatenate((rD_x, rD_y, rD_z), axis=2)
'''




def gdotR(r):
    rmag = (cp.sum(r*r, axis=2)**0.5) #xz-matrix
    ct = cp.dot(r,beUnit)/rmag #xz-matrix
    sbt = cp.cross(beUnit,r) #3vector xz-matrix
    st = sbt[...,2]/rmag #xz-matrix
    local_theta = cp.arctan( rD[...,0]/rD[...,1] ) #xz-matrix
    Rscrew_2 = (bscrew/(2*cp.pi)) * (local_theta-cp.pi*(rD[...,0]<0)).reshape(xsiz,zsiz+1,1)
    Rscrew = cp.concatenate((cp.zeros((xsiz, zsiz+1, 2)), Rscrew_2), axis=2) #3-vector xz-matrix
    Redge0 = (ct*st)[...,None] * bedge/(2*cp.pi*(1-nu)) #3vector xz-matrix
    Redge1 = ( ((2-4*nu)*cp.log(rmag)+(ct**2-st**2)) /(8*cp.pi*(1-nu)))[...,None]*bxu
    R = Rscrew + Redge0 + Redge1
    return cp.dot(R, gD)

gR = gdotR(rD)
rDdz = cp.add(rD, deltaz)
gRdz = gdotR(rDdz)
sxz = (gRdz -gR)/dz
#plt.imshow(cp.asnumpy(sxz))

#plt.imshow(cp.asnumpy(cp.log(abs(sxz))))







def HowieWhelan(F_in, ss): #ss takes the role of slocal for the avoidance of ambiguity



    ss = ss + eps #xy matrix
    gamma = cp.array([(ss-(ss**2+(1/Xgr)**2)**0.5)/2, (ss+(ss**2+(1/Xgr)**2)**0.5)/2]) #2vector of gamma xy matrices
    q = cp.array([(0.5/X0i)-0.5/(Xgi*((1+(ss*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(ss*Xgr)**2)**0.5))]) #2vector of q xy matrices
    beta = cp.arccos((ss*Xgr)/((1+ss**2*Xgr**2)**0.5)) #xy matrix
    
    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)],
                [-cp.sin(beta/2),cp.cos(beta/2)]])
    #Ci= cp.array([[cp.cos(beta/2), -cp.sin(beta/2)],[cp.sin(beta/2),  cp.cos(beta/2)]])
    Ci = cp.transpose(C, [1,0,2,3])
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



for z in tqdm(range(foil_slices)):
    pixel_depth = z*dt
    nearest_pixel = int(pixel_depth + 0.5)


    slocal = s + (1-diff)*sxz[:,top_int+nearest_pixel] + diff*sxz[:, top_int+ nearest_pixel +1]

    F = HowieWhelan(F, slocal)
F = cp.transpose(F, (2,3,0,1)).reshape(2,xsiz,ysiz)
Ib, Id = abs((F*cp.conjugate(F)))

#fig = plt.figure()





fig = plt.figure(figsize=(20, 10))
fig.add_subplot(2, 1, 1)
plt.imshow(cp.asnumpy(Ib))
plt.axis("off")
fig.add_subplot(2, 1, 2)
plt.imshow(cp.asnumpy(Id))
plt.axis("off")















