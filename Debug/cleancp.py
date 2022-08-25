import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import time
import inputs as inn
#from tqdm import tqdm
eps=0.00000000000001



#%%

X0i=inn.X0i
Xg=inn.Xg
a0 = inn.a0
nu = inn.nu
t0 = inn.t0
n = cp.array(inn.n)
g = cp.array(inn.g)
s = inn.s
z = cp.array(inn.z)
b0 = cp.array(inn.b0)
u = cp.array(inn.u)
dt = inn.dt
pix2nm = inn.pix2nm
pad = inn.pad
blursigma = inn.blursigma
inputtest1 = cp.asnumpy(u)
#%%

blursigma = blursigma / pix2nm    
t = t0 / pix2nm
pad = pad / pix2nm
a = a0 / pix2nm
X0i = X0i / pix2nm
Xg = Xg / pix2nm
#%%

def normalise(x):
    norm = x / (cp.dot(x, x) ** 0.5)
    return norm
#%%

u = normalise(u)
z = normalise(z)
n = normalise(n)
if cp.dot(n, z) < 0:  # they're antiparallel(ish), reverse n
    n = -n
if cp.dot(u, z) < 0:  # they're antiparallel(ish), reverse u and b
    u = -u
    b0 = -b0
#%%
zlen = int(t/dt + 0.5)
g = g / a
b = a * b0 #nm
n1, n2 = cp.copy(n), cp.copy(n)
n1[0], n2[1] = 0, 0
n1 = normalise(n1)
n2 = normalise(n2)
psi = cp.arccos(cp.dot(n1,z))*cp.sign(n[1])
theta = cp.arccos(cp.dot(n2,z))*cp.sign(n[0])
#%%

if abs(cp.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    x = b0[:]
    x = x / (cp.dot(x, x) ** 0.5)
    if abs(cp.dot(x, z) - 1) < eps:  # they're parallel too, set x parallel to g instead
        x = g[:]
        if abs(cp.dot(x, z) - 1) < eps:  # they're parallel too, set x arbitrarily
            x = cp.array(1, 0, 0)  
    phi=0.0 # angle between dislocation and z-axis
else:
    x = cp.cross(u, z)
    phi = cp.arccos(abs(cp.dot(u, z)))
#%%
x = normalise(x)
y = cp.cross(z, x)
#%%
c2s = cp.array((x, y, z))
s2c = cp.transpose(c2s)
nS = c2s @ n
uS = c2s @ u
#%%
xD = cp.copy(x)
yD = cp.cross(u, x)
zD = cp.copy(u)
c2d = cp.array((xD, yD, zD))
d2c = cp.transpose(c2d)

#%%
bscrew = cp.dot(b,u) #scalar
bedge = c2d @ (b - bscrew*u) #vector
beUnit = normalise(bedge)#unit vector
bxu = c2d @ cp.cross(b,u) #vector
gD = c2d @ g #vector
dz = 0.01 #scalar
deltaz = cp.array((0, dt*dz, 0)) / pix2nm #vector
#%%

xsiz = 2*int(pad + 0.5)
hpad = xsiz/cp.tan(phi, dtype=cp.float64)
if abs(cp.dot(u, z)) < eps:  # dislocation is in the plane of the foil
    ysiz = 1 * xsiz
    zsiz = zlen
elif abs(cp.dot(u, z)-1) < eps:  # dislocation along z
    ysiz = 1 * xsiz
    zsiz = zlen
else:
    w = int(t*nS[2]*nS[2]*uS[1]/abs(cp.dot(u,n1)) + 0.5)
    ysiz = w + xsiz
    zsiz = int( (2*t*nS[2] + hpad + xsiz*cp.tan(abs(psi)) )/dt + 0.5)
print(xsiz,ysiz,zsiz)

#%%

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
    rDnorm = rD/rmag[...,None]
    ct = cp.dot(rDnorm,beUnit) #xz-matrix
    sbt = cp.cross(beUnit,rDnorm) #3vector xz-matrix 
    st = sbt[...,2] #xz-matrix
    Rscrew_2 = (bscrew*(cp.arctan(rD[...,1]/rD[...,0])-cp.pi*(rD[...,0]<0))/(2*cp.pi)).reshape(xsiz, zsiz+1,1) #component 3 (z) of 3vector xz-matrix
    Rscrew = cp.concatenate((cp.zeros((xsiz, zsiz+1, 2)), Rscrew_2), axis=2) #3vector xz-matrix
    Redge0 = (ct*st)[...,None] * bedge/(4*cp.pi*(1-nu)) #3vector xz-matrix
    Redge1 = ( ((2-4*nu)*cp.log(rmag)+(ct**2-st**2)) /(8*cp.pi*(1-nu)))[...,None]*bxu #3vector xz-matrix
    R = (Rscrew + Redge0 + Redge1) #3vector xz-matrix
    gR = cp.dot(R, gD) #xz-matrix
    return gR
#%%

testx = np.zeros((2, xsiz, ysiz))

def howieWhelan(F_in,Xg,X0i,slocal,alpha,t):
    '''
    X0i  scalar
    Xg = scalar
    s =  xy matrix
    alpha = scalar
    t = scalar
    '''
    
    
    #ISSUE NOW HERE: S NOT PROPAGATING AS A MATRIX!!!!
    
    
    
    
    
    Xgr = Xg.real #const
    Xgi = Xg.imag #const
    slocal = slocal + eps #xy matrix
    #testx[0] = cp.asnumpy(slocal)
    #print("slocal", slocal)
    sX = (slocal * Xgr)**2
    #print(sX)
    gamma = cp.zeros((2, xsiz, ysiz))
    gamma[0] = (slocal-(slocal**2+(1/Xgr)**2)**0.5)/2
    gamma[1] = (slocal+(slocal**2+(1/Xgr)**2)**0.5)/2
    #print("gamma:", gamma)
    #testx[0] = cp.asnumpy(gamma[0])
    #gamma = cp.array([0.5*slocal*(1-(1+(1/sX))**0.5), 0.5*slocal*(1+(1+(1/sX))**0.5)]) #2vector of gamma xy matrices
    #print(cp.transpose(gamma, (1,2,0))) #WRONG
    #q = cp.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))]) #2vector of q xy matrices
    #q = cp.array([(0.5/X0i) - (0.5/Xgi)*(1+sX)**0.5, (0.5/X0i) + (0.5/Xgi)*(1+sX)**0.5])
    q = cp.zeros((2, xsiz, ysiz))
    q[0] = (0.5/X0i) - (0.5/Xgi)*(1+sX)**-0.5
    q[1] = (0.5/X0i) + (0.5/Xgi)*(1+sX)**-0.5
    #print("q:", cp.transpose(q, (1,2,0)))
    
    #print(cp.transpose(q, (1,2,0))) #WRONG
    #beta = cp.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5)) #xy matrix
    #beta = cp.arccos((sX/(1+sX))**0.5) #not sure why this doesn't work
    beta = np.arccos((slocal*Xgr)/((1+slocal**2*Xgr**2)**0.5))
    #print("beta=", beta)
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)], #2matrix of xy matrices
                 [-cp.sin(beta/2)*cp.exp(complex(0,alpha)),
                  cp.cos(beta/2)*cp.exp(complex(0,alpha))]])
    Ci= cp.array([[cp.cos(beta/2), -cp.sin(beta/2)*cp.exp(complex(0,-alpha))],
                 [cp.sin(beta/2),  cp.cos(beta/2)*cp.exp(complex(0,-alpha))]])
    G=cp.zeros((2,2,xsiz, ysiz))
    G[0,0,...] = cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t)
    G[1,1,...] = cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)
    #G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
    #            [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]], dtype=cp.float_)
    

    
    
    
    gamma = cp.transpose(gamma, (1,2,0)).reshape(xsiz,ysiz,2,1) #xy matrix of gamma 2vectors
    q = cp.transpose(q, [1,2,0]).reshape(xsiz,ysiz,2,1) #xy matrix of q 2vectors
    C=cp.transpose(C, [2,3,0,1])
    Ci=cp.transpose(Ci, [2,3,0,1])
    G=cp.transpose(G, [2,3,0,1])
    F_out = C  @ G  @ Ci  @ F_in
    return F_out
#%%
start_time = time.perf_counter()
#%%
x_vec = cp.arange(xsiz) + 0.5 - xsiz/2 #x-vector
rX = (cp.transpose(cp.tile(x_vec, (zsiz+1, 1)))).reshape(xsiz,zsiz+1,1) #xz-matrix
z_vec = dt*(cp.arange(zsiz+1) +0.5 -zsiz/2)*(cp.sin(phi) + xsiz*cp.tan(psi)/(2*zsiz))
z_mat = cp.tile(z_vec, (xsiz, 1)) #xz-matrix
rD_1 = z_mat.reshape(xsiz,zsiz+1,1) + rX*cp.tan(theta) #xz matrix (scalar)
rD = cp.concatenate((rX, rD_1, cp.zeros((xsiz, zsiz+1,1))), axis=2)/pix2nm #3vector xz-matrix
gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD) #xz-matrix
rDdz = cp.add(rD, deltaz) #3vector xz-matrix
gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD) #xz-matrix
sxz = (gRdz - gR)/dz #xz-matrix

#%%
F0 = cp.array([[1], [0]])
F = cp.tile(F0, (xsiz, ysiz, 1, 1))
p = cp.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
zlen=int(t*nS[2]/dt + 0.5)

top_vec = cp.arange(ysiz) * (zsiz-zlen)/ysiz #y vector
h_vec = top_vec.astype(int) #y vector
m_vec = top_vec - h_vec #y vector
top = cp.tile(top_vec, (xsiz,1)) #xy matrix
h = top.astype(int) #xy matrix
m = top - h #xy matrix

#%%
testarr = np.zeros((2, zlen))
for z in (range(zlen)):
    slocal = s + (1-m)*sxz[:,(h_vec+z)]+m*sxz[:,(h_vec+z+1)] #xy matrix
    
    alpha = 0.0
    F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm) #xy matrix of 2vectors
    
    testarr[0,z] = cp.asnumpy(F[0,0,0,0].real)
    



F = cp.transpose(F, (2,3,0,1)).reshape(2,xsiz,ysiz)
Ib, Id = cp.real(F*cp.conj(F))


#%%
end_time = time.perf_counter()
duration = end_time - start_time
#print("Main loops took: " + str(duration) + " seconds")
#print(dt)
#%%
'''
Ib = cp.ndarray.tolist(Ib) 
Id = cp.ndarray.tolist(Id)
fig = plt.figure(figsize=(16, 8))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib)
plt.axis("off")
fig.add_subplot(2, 1, 2)
plt.imshow(Id)
plt.axis("off")
'''
































#%%
X0i=inn.X0i
Xg=inn.Xg
a0 = inn.a0
nu = inn.nu
t0 = inn.t0
n = np.array(inn.n)
g = np.array(inn.g)
s = inn.s
z = np.array(inn.z)
b0 = np.array(inn.b0)
u = np.array(inn.u)
dt = inn.dt
pix2nm = inn.pix2nm
pad = inn.pad
blursigma = inn.blursigma

t = t0 / pix2nm
pad = pad / pix2nm
a = a0 / pix2nm
X0i = X0i / pix2nm
Xg = Xg / pix2nm
zlen = int(t/dt + 0.5)
g = g / a
b = a * b0

#%%
u = u / (np.dot(u, u) ** 0.5)
z = z / (np.dot(z, z) ** 0.5)
n = n / (np.dot(n, n) ** 0.5)

if np.dot(n, z) < 0:  # they're antiparallel, reverse n
    n = -n
if np.dot(u, z) < 0:  # they're antiparallel, reverse u and b
    u = -u
    b = -b
n1, n2 = np.copy(n), np.copy(n)
n1[0], n2[1] = 0, 0
n1 = n1 / (np.dot(n1, n1) ** 0.5)
n2 = n2 / (np.dot(n2, n2) ** 0.5)
psi = np.arccos(np.dot(n1,z))*np.sign(n[1])
theta = np.arccos(np.dot(n2,z))*np.sign(n[0])
dz = 0.01
deltaz = np.array((0, dt*dz, 0)) / pix2nm
#%%

if abs(np.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    x = b[:]
    x = x / (np.dot(x, x) ** 0.5)
    if abs(np.dot(x, z) - 1) < eps:  # they're parallel too, set x parallel to g
        x = g[:]
    phi=0.0
else:
    x = np.cross(u, z)
    phi = np.arccos(abs(np.dot(u, z)))
#%%
x = x / (np.dot(x, x) ** 0.5)
y = np.cross(z, x)
#%%
c2s = np.array((x, y, z))
s2c = np.transpose(c2s)
nS = c2s @ n
uS = c2s @ u
#%%
xD = x
yD = np.cross(u, x)
zD = u
c2d = np.array((xD, yD, zD))

d2c = np.transpose(c2d)
#%%
bscrew = np.dot(b,u)#NB a scalar
bedge = c2d @ (b - bscrew*u)#NB a vector
beUnit = bedge/(np.dot(bedge,bedge)**0.5)#a unit vector
bxu = c2d @ np.cross(b,u)
gD = c2d @ g

#%%
xsiz = 2*int(pad + 0.5)
hpad = xsiz/np.tan(phi)
if abs(np.dot(u, z)) < eps:  # dislocation is in the plane of the foil
    ysiz = 1 * xsiz
    zsiz = zlen
    print("Dislocation is in the plane of the foil")
elif abs(np.dot(u, z)-1) < eps:  # dislocation along z
    ysiz = 1 * xsiz
    zsiz = zlen
    print("Dislocation is parallel to the beam")
else:
    w = int(t*nS[2]*nS[2]*uS[1]/abs(np.dot(u,n1)) + 0.5)
    ysiz = w + xsiz
    zsiz = int( (2*t*nS[2] + hpad + xsiz*np.tan(abs(psi)) )/dt + 0.5)
print(xsiz,ysiz,zsiz)
#%%
def howieWhelan(F_in,Xg,X0i,slocal,alpha,t):
    Xgr = Xg.real
    Xgi = Xg.imag
    slocal = slocal + eps
    #print(slocal)
    gamma = np.array([(slocal-(slocal**2+(1/Xgr)**2)**0.5)/2, (slocal+(slocal**2+(1/Xgr)**2)**0.5)/2])
    #print(gamma)
    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5))])
    #q[0] = (0.5/X0i) - (0.5/Xgi)*(1+sX)**-0.5
    #print("q2:", q)
    beta = np.arccos((slocal*Xgr)/((1+slocal**2*Xgr**2)**0.5))
    C=np.array([[np.cos(beta/2), np.sin(beta/2)],
                [-np.sin(beta/2)*np.exp(complex(0,alpha)),
                 np.cos(beta/2)*np.exp(complex(0,alpha))]])
    Ci=np.array([[np.cos(beta/2), -np.sin(beta/2)*np.exp(complex(0,-alpha))],
                 [np.sin(beta/2),  np.cos(beta/2)*np.exp(complex(0,-alpha))]])
    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0],
                [0, np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]])
    F_out = C @ G @ Ci @ F_in
    return F_out
#%%
def gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD):
    r2 = np.dot(rD,rD)
    rmag = r2**0.5
    ct = np.dot(rD,beUnit)/rmag
    sbt = np.cross(beUnit,rD)/rmag
    st = sbt[2]
    Rscrew = np.array((0,0,bscrew*(np.arctan(rD[1]/rD[0])-np.pi*(rD[0]<0))/(2*np.pi)))
    Redge0 = bedge*ct*st/(4*np.pi*(1-nu))
    Redge1 = bxu*( ( (2-4*nu)*np.log(rmag)+(ct**2-st**2) )/(8*np.pi*(1-nu)))
    R = (Rscrew + Redge0 + Redge1)
    gR = np.dot(gD,R)
    return gR
#%%
def calculate_deviations(xsiz, zsiz, pix2nm, t, dt, u, g, b, c2d, nu, phi, psi, theta):
    sxz = np.zeros((xsiz, zsiz+1), dtype='f')
    for x in (range(xsiz)):
        for z in range(zsiz+1):
            rX = 0.5+x-xsiz/2
            rD = np.array((rX,
                           dt*( (0.5+z-zsiz/2)*(np.sin(phi) + xsiz*np.tan(psi)/(2*zsiz)) )
                               + rX*np.tan(theta),
                           0)) / pix2nm#in nm
            gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD)
            rDdz = rD + deltaz
            gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD)
            sxz[x,z] = (gRdz - gR)/dz
    return sxz
#%%
def calculate_image(sxz, xsiz, ysiz, zsiz, pix2nm, t, dt, s,
                    Xg, X0i, g, b, nS, psi, theta, phi):
    Ib = np.zeros((xsiz, ysiz))
    Id = np.zeros((xsiz, ysiz))
    F0 = np.array([[1], [0]])
    p = np.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
    zlen=int(t*nS[2]/dt + 0.5)
    for x in (range(xsiz)):
        for y in range(ysiz):
            F = F0[:]
            top = (zsiz-zlen)*y/ysiz
            h=int(top)
            m = top-h
            for z in range(zlen):
                slocal = s + (1-m)*sxz[x,h+z]+m*sxz[x,h+z+1]
                alpha = 0.0
                F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm)
                testarr[1,z] = F[0,0]
            Ib[xsiz-x-1,y] = (F[0]*np.conj(F[0])).real
            Id[xsiz-x-1,y] = (F[1]*np.conj(F[1])).real
    return Ib, Id
#%%
start_time = time.perf_counter()
#%% 

sxz = calculate_deviations(xsiz, zsiz, pix2nm, t, dt,
                                   u, g, b, c2d, nu, phi, psi, theta)

Ib, Id= calculate_image(sxz, xsiz, ysiz, zsiz, pix2nm, 
                                 t, dt, s, Xg, X0i, g, b, nS, psi, theta, phi)


#%%

end_time = time.perf_counter()
duration = end_time - start_time
#print("Main loops took: " + str(duration) + " seconds")

#%%
'''
fig = plt.figure(figsize=(16, 8))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib)
plt.axis("off")
fig.add_subplot(2, 1, 2)
plt.imshow(Id)
plt.axis("off")
'''


#print(testarr[0]-testarr[1]) #slocal ok. what's changing it?????
instability = testarr[0]-testarr[1]
print(instability)
plt.plot(instability)
plt.savefig('Instability.png')
plt.show()
#THIS SHOWS THE DIFFERENCE BETWEEN THE VERSIONS AS A FUNCTION OF SLICES (ITERATIONS)






