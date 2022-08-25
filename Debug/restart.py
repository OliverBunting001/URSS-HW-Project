import numpy as np
import matplotlib.pyplot as plt
import time
import inputs as inn
eps=0.000000000001
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
b= a * b0

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
#print(bedge)
#print(bedge-cp.asnumpy(bedge_dup))
beUnit = bedge/(np.dot(bedge,bedge)**0.5)#a unit vector
bxu = c2d @ np.cross(b,u)
gD = c2d @ g
d2c = np.transpose(c2d)

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
def howieWhelan(F_in,Xg,X0i,s,alpha,t):
    Xgr = Xg.real
    Xgi = Xg.imag
    s = s + eps
    gamma = np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])
    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])
    beta = np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))
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
    Redge0 = bedge*ct*st/(2*np.pi*(1-nu))
    Redge1 = bxu*( ( (2-4*nu)*np.log(rmag)+(ct**2-st**2) )/(8*np.pi*(1-nu)))
    R = (Rscrew + Redge0 + Redge1)
    gR = np.dot(gD,R)
    return gR, st
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
            gR, test = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD)
            test1[1,x,z] = test
            #test2[1,x,z] = test 
            rDdz = rD + deltaz
            gRdz, _ = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD)
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
            Ib[xsiz-x-1,y] = (F[0]*np.conj(F[0])).real
            Id[xsiz-x-1,y] = (F[1]*np.conj(F[1])).real
    return Ib, Id
#%%
start_time = time.perf_counter()
#%% 

sxz = calculate_deviations(xsiz, zsiz, pix2nm, t, dt,
                                   u, g, b, c2d, nu, phi, psi, theta)

Ib, Id = calculate_image(sxz, xsiz, ysiz, zsiz, pix2nm, 
                                 t, dt, s, Xg, X0i, g, b, nS, psi, theta, phi)