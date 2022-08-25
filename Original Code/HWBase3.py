import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import PySupport as funcs_1
import inputs_ez as inn

eps = 0.000000000001


################################## Choose method
   


toc = time.perf_counter()


X0i=inn.X0i #scalar
Xg=inn.Xg #complex number
a0 = inn.a0 #scalar
nu = inn.nu
t0 = inn.t0
n = np.array(inn.n)
g = np.array(inn.g)
s = np.array(inn.s)
z = np.array(inn.z)
b0 = np.array(inn.b0)
u = np.array(inn.u)
dt = inn.dt
pix2nm = inn.pix2nm
pad = inn.pad
blursigma = inn.blursigma
print(100000000000*s)



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
    b0 = -b0


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
b = a * b0




if abs(np.dot(u, z) - 1) < eps:  # they're parallel, set x parallel to b
    x = b[:]
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




xD = x
yD = np.cross(u, x)
zD = u
c2d = np.array((xD, yD, zD))
d2c = np.transpose(c2d)



xsiz = 2*int(pad + 0.5)
hpad = xsiz/np.tan(phi)


if abs(np.dot(u, z)) < eps:  # dislocation is in the plane of the foil
    ysiz = 1 * xsiz # in pixels
    zsiz = zlen # in slices
    print("Dislocation is in the plane of the foil")
elif abs(np.dot(u, z)-1) < eps:  # dislocation along z
    #needs work?
    ysiz = 1 * xsiz
    zsiz = zlen # in slices
    print("Dislocation is parallel to the beam")
else:  # dislocation is at an angle to the beam direction
    # dislocation image length
    w = int(t*nS[2]*nS[2]*uS[1]/abs(np.dot(u,n1)) + 0.5)
    ysiz = w + xsiz # in pixels
    zsiz = int( (2*t*nS[2] + hpad + xsiz*np.tan(abs(psi)) )/dt + 0.5) # in slices
# print(xsiz,ysiz,zsiz)















def howieWhelan(F_in,Xg,X0i,slocal,alpha,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real
    Xgi = Xg.imag

    slocal = slocal + eps

    gamma = np.array([(slocal-(slocal**2+(1/Xgr)**2)**0.5)/2, (slocal+(slocal**2+(1/Xgr)**2)**0.5)/2])

    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(slocal*Xgr)**2)**0.5))])

    beta = np.arccos((slocal*Xgr)/((1+slocal**2*Xgr**2)**0.5))

    #scattering matrix
    C=np.array([[np.cos(beta/2), np.sin(beta/2)],
                [-np.sin(beta/2)*np.exp(complex(0,alpha)),
                 np.cos(beta/2)*np.exp(complex(0,alpha))]])

    #inverse of C is just its transpose
    Ci=np.transpose(C)

    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0],
                [0, np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]])

    F_out = C @ G @ Ci @ F_in

    return F_out

def gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD):
    # returns displacement vector R at coordinate xyz
    r2 = np.dot(rD,rD)
    rmag = r2**0.5
    #cos(theta) & sin(theta) relative to Burgers vector
    ct = np.dot(rD,beUnit)/rmag
    sbt = np.cross(beUnit,rD)/rmag
    st = sbt[2]
    # From Head et al. Eq. 2.31, p31
    # infinite screw dislocation displacement is b.phi/(2pi):
    # using x=r.sin(phi) & y=r.cos(phi)
    # have to take away pi to avoid double-valued tan
    Rscrew = np.array((0,0,bscrew*(np.arctan(rD[1]/rD[0])-np.pi*(rD[0]<0))/(2*np.pi)))
    # infinite edge dislocation displacement field part 1: b.sin2theta/4pi(1-nu)
    # using sin(2theta)=2sin(theta)cos(theta)
    Redge0 = bedge*ct*st/(2*np.pi*(1-nu))
    # part 2: (b x u)*( (1-2v)ln(r)/2(1-v) + cos(2theta)/4(1-v) )/2pi
    # using cos(2theta)= cos^2(theta) - sin^2(theta)
    Redge1 = bxu*( ( (2-4*nu)*np.log(rmag)+(ct**2-st**2) )/(8*np.pi*(1-nu)))
    # total displacement 
    R = (Rscrew + Redge0 + Redge1)
    # dot product with g-vector
    gR = np.dot(gD,R)
#    gR = np.dot(gD,Redge1)
    return gR

def calculate_deviations(xsiz, zsiz, pix2nm, t, dt, u, g, b, c2d, nu, phi, psi, theta):
    # calculates the local change in deviation parameter s as the z-gradient of the displacement field
    
    #dislocation components & g: in the dislocation reference frame
    bscrew = np.dot(b,u)#NB a scalar
    bedge = c2d @ (b - bscrew*u)#NB a vector
    print(bedge)
    bedge[2]=0
    print(bedge)
    beUnit = bedge/(np.dot(bedge,bedge)**0.5)#a unit vector
    bxu = c2d @ np.cross(b,u)
    gD = c2d @ g
    #tranformation matrix from dislocation to crystal frame
    d2c = np.transpose(c2d)
    #the x-z array array of deviation parameters
    sxz = np.zeros((xsiz, zsiz+1), dtype='f')#32-bit for .tif saving, +1 is for interpolation

    # small z value used to get derivative
    dz = 0.01
    deltaz = np.array((0, dt*dz, 0)) / pix2nm

    # calculation of displacements R and the gradient of g.R
    for x in tqdm(range(xsiz)):
        for z in range(zsiz+1):
            # working in the dislocation frame here
            # vector to dislocation is rD, NB half-pixels puts the dislocation between pixels
            rX = 0.5+x-xsiz/2
            rD = np.array((rX,
                           dt*( (0.5+z-zsiz/2)*(np.sin(phi) + xsiz*np.tan(psi)/(2*zsiz)) )
                               + rX*np.tan(theta),
                           0)) / pix2nm#in nm
            #Displacement R is calculated in the crystal frame
            gR = gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD )
            rDdz = rD + deltaz
            gRdz = gdotR(rDdz, bscrew, bedge, beUnit, bxu, d2c, nu, gD )
            sxz[x,z] = (gRdz - gR)/dz
    return sxz


def calculate_image(sxz, xsiz, ysiz, zsiz, pix2nm, t, dt, s,
                    Xg, X0i, g, b, nS, psi, theta, phi):
    Ib = np.zeros((xsiz, ysiz), dtype='f')  # 32-bit for .tif saving
    # Dark field image
    Id = np.zeros((xsiz, ysiz), dtype='f')
    # Complex wave amplitudes are held in F = [BF,DF]
    F0 = np.array([[1], [0]])  # input wave, top surface
    # centre point of simulation frame is p
    p = np.array((0.5+xsiz/2,0.5+ysiz/2,0.5+zsiz/2))
    # length of wave propagation
    zlen=int(t*nS[2]/dt + 0.5)#remember nS[2]=cos(tilt angle)

    for x in tqdm(range(xsiz)):
        for y in range(ysiz):
            F = F0[:]
            # z loop propagates the wave over a z-line of sxz of length zlen 
            # corresponding to the thickness of the foil (t/dt slices)
            # For this row the top of the foil is
#            top = (zsiz-zlen)*(1 - y/ysiz)
            top = (zsiz-zlen)*y/ysiz 
            # if (nS[1] > 0):
            #     top = (zsiz-zlen)*(1 - y/ysiz + x*xsiz*np.tan(theta)) # in pixels
            # else:
            #     top = (zsiz-zlen)*(y/ysiz + x*xsiz*np.tan(theta))
            h=int(top)
            # linear interpolation of slocal between calculated points
            m = top-h
            for z in range(zlen):
                slocal = s + (1-m)*sxz[x,h+z]+m*sxz[x,h+z+1]

                # stacking fault shift is present between the two dislocations
                # if firs < x < las and h+z-int(zrange / 2) == 0:
                #     alpha = 2*np.pi*np.dot(g,b1)
                # else:
                alpha = 0.0
                F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm)

            # bright field is the first element times its complex conjugate
            Ib[xsiz-x-1,y] = (F[0]*np.conj(F[0])).real
            # dark field is the second element times its complex conjugate
            Id[xsiz-x-1,y] = (F[1]*np.conj(F[1])).real

    return Ib, Id





































start_time = time.perf_counter()


sxz = funcs_1.calculate_deviations(xsiz, zsiz, pix2nm, t, dt,
                                   u, g, b, c2d, nu, phi, psi, theta)
Ib, Id = funcs_1.calculate_image(sxz, xsiz, ysiz, zsiz, pix2nm, 
                                  t, dt, s, Xg, X0i, g, b, nS, psi, theta, phi)


end_time = time.perf_counter()
duration = end_time - start_time
print("Main loops took: " + str(duration) + " seconds")

ker = int(7/pix2nm+0.5)+1


Ib2=Ib
Id2=Id


################################## Output image notation stuff
# g-vector on image is leng pixels long
leng = pad / 4
gDisp = c2s @ g
gDisp = leng * gDisp / (np.dot(gDisp, gDisp) ** 0.5)
bDisp1 = c2s @ b
bDisp1 = leng * bDisp1 / (np.dot(bDisp1, bDisp1) ** 0.5)


################################## Output image display
fig = plt.figure(figsize=(16, 8))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib2)
plt.axis("off")


fig.add_subplot(2, 1, 2)
plt.imshow(Id2)
plt.axis("off")




tic = time.perf_counter()
print("Full function took: " + str(tic - toc) + " seconds")