import cupy as cp
#import numpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
eps = 0.000000000000001



def normalise(x):
    if abs(x[0])<eps and abs(x[1])<eps and abs(x[2])<eps:
        return cp.zeros(3)
    norm = x / (cp.dot(x, x) ** 0.5)        
    return norm

X0I = 1000
a = 0
XGR = 20
XGI = X0I * (1+a)

XG = XGR + 1j*XGI






s=0.05





b = cp.array([0.5, 0.5, 0])
g = cp.array([2, 0, 0])




z = cp.array([9, -4, 1])
u = cp.array([2, 3, -1])
n = cp.array([1, 1, -8])

z = normalise(z)
u = normalise(u)
n = normalise(n)

un = cp.dot(u,n)
uz = cp.dot(u,z)
nz = cp.dot(n,z)

t=100 #nm
dt =1 #nm
pad = 20 #nm
a = 0.3 #nm
nu = 0.4 #dimensionless

pixpernm = 1 #pix/nm

t *=pixpernm #pix
dt *=pixpernm #pix
pad *=pixpernm #pix
a *= pixpernm #pix


g = g/a
b = a*b

bu = cp.dot(b,u)
bxu = cp.cross(b,u)
bscrew = bu * u #parallel
bedge = b - bscrew #perpendicular
beUnit = normalise(bedge)



'''
foil planes are:

r.n=0 , r.n=t   (1)

defect intercepts (respectively) at:

r=0 , r=u * t/u.n   (2)

simulation (xy) frame defined by:
    
r.z=0 (also intercepts defect at the origin)    (3)


Projection of the defect onto the xy plane defines the y direction.

(u * t/u.n) - hz = wy   (4)

h is such that it lands in the xy simulation plane (hence dot product with z is 0)

((u * t/u.n) - hz).z=0      (5)

h = t* (u.z)/(u.n)      (6)

This gives the y direction and w by (4)
'''

r = t/un #from (2)

h = t * uz/un #from (5/6)

wy = r*u - h*z #from (4)

w = (cp.dot(wy,wy)**0.5) #magnitude
w_int = int(w)
y = wy/w # unit vector


#now have y and z which together give the x direction
x = cp.cross(y,z)


ux = cp.dot(u,x) #these will be useful
uy = cp.dot(u,y)
nx = cp.dot(n,x)
ny = cp.dot(n,y)

'''
4 corners of the simulated rectanagle are, in the simulation frame:
    
    
(-pad,-pad); (pad,-pad); (-pad, w+pad); (pad, w+pad)        (7)


To each of these, add some amount of z so that they intersect, both the r.n=0
and the r.n=t planes:


(-pad*x - pad*y + a*z).n=0
(+pad*x - pad*y + b*z).n=0
(-pad*x + (w+pad)*y + c*z).n=0
(+pad*x + (w+pad)*y + d*z).n=0

(-pad*x - pad*y + e*z).n=t
(+pad*x - pad*y + f*z).n=t
(-pad*x + (w+pad)*y + g*z).n=t
(+pad*x + (w+pad)*y + h*z).n=t                      (8)

where bracketed quantity is the r we want to find for each
In practice, only need to find the z components given by letters a-h
'''

co_ords = cp.array([ [-pad, -pad], [pad, -pad], [-pad, w+pad], [pad, w+pad] ], dtype=float)






def z_to_meet_foil(coord, zero_or_t):
    letter = (zero_or_t - coord[0]*nx - coord[1]*ny) / nz
    return letter

z0_array = cp.zeros(4)
zt_array = cp.zeros(4)

for i in range(4):
    z0_array[i] = z_to_meet_foil(co_ords[i], 0)
    zt_array[i] = z_to_meet_foil(co_ords[i], t)



'''
Would also like the points where the defect intercepts the simulation x-z planes:

y = -pad , y = pad + w      (9)


-pad*y + H1*z || u
(pad+w)*y + H2*z || u       (10)


u = u.y y + u.z z           (11)

u.z/u.y = H1/-pad = H2/(pad+w)      (12)
'''


H1 = -pad * uz/uy       #from (12)
H2 = (pad+w) * uz/uy





'''
Now need some idea of the orientation of the planes with respect to the defect

We already have the simulation z coordinates of where the foil intercepts
the xz planes y=-pad; y=w+pad 

And have just obtained the same information for the defect

Problem is that defect's z height is at x=0 and plane corners are at x=+/-pad


Let's linearly interpolate between the plane intercepts to find
the x=0 simulation foil top and bottom z heights
'''

top_foil_midheight = (z0_array[0] + z0_array[1])*0.5
bottom_foil_midheight = (zt_array[0] + zt_array[1])*0.5

#H1 must be either bigger or smaller than BOTH of these






'''

if H1>top_foil_midheight and H1>bottom_foil_midheight:
    forward_defect=False
    print("Backwards defect") #as in \
elif H1<top_foil_midheight and H1<bottom_foil_midheight:
    forward_defect=True
    print("Forwards defect") #as in /
else:
    print("Broken")


if z0_array[0]>z0_array[1]:
    front_low = True
    print("Front is lower than back (in simulation frame)")
elif z0_array[0]<z0_array[1]:
    front_low = False
    print("Front is higher than back (in simulation frame)")
else:
    print("No x tilt")
    


if left_low:
    if front_low:
        low_z = z0_array[1]
        high_z = zt_array[2]
    else:
        low_z = z0_array[0]
        high_z = zt_array[3]
else:
    if front_low:
        low_z = z0_array[3]
        high_z = zt_array[0]
    else:
        low_z = z0_array[2]
        high_z = zt_array[1]

'''





'''
in y=-pad plane, the defect will be either above or below the two planes
We need the vertical distance to the lowest or highest point of the foil
'''

z_arr = cp.append(z0_array, zt_array).reshape(2,4)

z_pad = z_arr[:,:2] #foil intercepts in y=-pad plane
z_wpad = z_arr[:,2:] # foil intercepts in y=pad+w plane

zsiz_pad = cp.amax(abs(z_pad-H1)) + 0.5 #pix
zsiz_wpad =  cp.amax(abs(z_wpad-H2)) + 0.5 #pix - These should always be identical? 
zsiz = int((zsiz_pad + zsiz_wpad)/dt + 0.5) + 1 #slices

xsiz = int(2*pad + 0.5)
ysiz = 2*pad + w_int

print("xsiz,ysiz,zsiz =", xsiz, ysiz, zsiz)


'''
high_z = cp.round(high_z).astype(int) #slices required above the origin 
low_z = cp.round(low_z).astype(int) #slices required below the origin

print(high_z,low_z)
'''




'''

u = u.y y + u.z z           (12)

Would like to define a coordinate system where u is the z direction
x is the x direction and u cross x is the y direction
'''


y_alt = cp.cross(u,x)


'''
Let's map the y=0 x-z plane to a set of vectors (rD) which connect each point
perpendicularly to the defect


X x + Z z  +  rD = factor * u       (13)

Can solve this by finding 'factor'

Usefully, rD.u=0 (they are perpendicular by definition)

X x.u + Z z.u + (rD.u=0) = factor (14)
'''


def rD(X, Z):
    factor = X * ux + Z * uz        #from (14)
    return factor*u - X*x - Z*z     #from (13)




xarr = cp.linspace(-xsiz/2 + 0.5, xsiz/2 - 0.5, xsiz+1)
xmat = cp.transpose( cp.tile(xarr, (zsiz+1,1)) ).reshape(xsiz+1,zsiz+1,1)

zarr = cp.linspace(-zsiz/2, zsiz/2, zsiz+1)
zmat = cp.tile(zarr, (xsiz+1,1)).reshape(xsiz+1,zsiz+1,1)


RD = rD(xmat,zmat) #in these arrays, martix values correspond exactly to grid points
                   #i.e. the origin is given by index [xsiz/2, zsiz/2]


'''
now need to find the displacement field - d/dz (g.R)

R = 1/2pi (bscrew*theta + bedge*Redge + bxu*Rbxu)     (15)
Redge = sin(2theta)/4(1-nu)                           (16)
Rbxu = (1-2nu)ln(r)/2(1-nu) + cos(2theta)/4(1-nu)     (17)

theta is the angle between the dislocation-plane vector and beUnit

ct = RD.beUnit
st = 
'''


def gdotR(r):
    rmag = (r[...,0]**2 + r[...,1]**2)**0.5
    cth = cp.dot(r, beUnit)/rmag
    sth = cp.cross(r, beUnit).dot(u)/rmag
    th = cp.arctan(r[...,1]/r[...,1])
    
    Redge = 2 * sth * cth / (4*(1-nu))    #(16)
    Rbxu = ((1-2*nu)/(1-nu)) * cp.log(rmag)/2 + (cth**2-sth**2)/(4*(1-nu))  #(17)
    
    R = 1/(2*cp.pi) * (bscrew*th[...,None] + bedge*Redge[...,None] + bxu*Rbxu[...,None])       #(15)

    return cp.dot(R, g)




gR = gdotR(RD)

dz=0.1
RDdz = rD(xmat, zmat-dz) #vectors pointing to very slightly below the original points
gRdz = gdotR(RDdz)

sxz = (gRdz-gR)/dz

plt.imshow(cp.asnumpy(cp.log(abs(sxz))))
#plt.imshow((sxz))




def howieWhelan(F_in,Xg,X0i,s,t,zcount):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real
    Xgi = Xg.imag
    if zcount==0:
        print(Xgr, Xgi)
    s = s + eps

    gamma = cp.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])

    q = cp.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])

    beta = cp.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))

    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)],
                [-cp.sin(beta/2),
                 cp.cos(beta/2)]])

    #inverse of C is just its transpose
    #Ci=cp.transpose(C)

    G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]])
    
    
    C = cp.transpose(C, [2,3,0,1])
    Ci = cp.transpose(C, [0,1,3,2])
    G = cp.transpose(G, [2,3,0,1])

    

    F_out = C @ G @ Ci @ F_in
    
    return F_out


F0 = cp.array([[1], [0]])
F = cp.tile(F0, (xsiz+1, ysiz+1, 1, 1))


'''
Make a function that can generate the simulation z coordinate of a given plane
being given the x and y values in the simulation plane.

Obviously in practice, this given plane will be one of the foil planes
    

(X x + Y y + Z z).n = 0 or t            (18)

Z = ((0 or t) - X x.n - Y y.n)/z.n      (19)
'''

def zvalue(X, Y, zero_or_t):
    Z = (zero_or_t - X*nx - Y*ny)/nz
    return Z



yarr = cp.linspace(-pad, w_int+pad, ysiz+1)
ymat = cp.tile(yarr, [xsiz+1,1])

xarr = cp.linspace(-pad, pad, xsiz+1)
xmat = cp.transpose(cp.tile(xarr, [ysiz+1,1]))


z_bottom = zvalue(xmat, ymat, 0)
z_top = zvalue(xmat, ymat, t)


'''
Defect runs from H1 at y=-pad to H2 at y=w+pad through 0 at y=0
'''

defect_height = cp.linspace(H1, H2, ysiz+1)




'''
Thickness in simulation frame is z_top - z_bottom
'''

tS = abs(cp.mean(z_top-z_bottom))
no_slices = int(tS/dt + 0.5)



for xx in tqdm(range(xsiz+1)):
    for yy in range(ysiz+1):
         vert = z_top[xx,yy]-defect_height[yy]
         vert_int = vert.astype(int)
         vertdiff = vert-vert_int
         hor = xarr[xx].astype(int)
         
         slocal = s + (1-vertdiff)*sxz[hor,vert_int] + vertdiff*sxz[hor,vert_int+1]
         





vert = z_top - defect_height

















































'''

ytop = cp.arange(ysiz) * (zsiz-t)/(ysiz-1)
ytop_int = ytop.astype(int)
ydiff = ytop - ytop_int

xtop = cp.arange(xsiz+1) * (zsiz-t)/(xsiz)
xtop_int = xtop.astype(int)
xdiff = xtop - xtop_int


sxzbulk = cp.zeros((xsiz,ysiz, foil_slices))

for yy in range(ysiz):
    relevant_section = sxz[:,top_int[yy]:top_int[yy]+foil_slices]
    sxzbulk[:,yy,:] = relevant_section



for zz in tqdm(range(foil_slices)):
    slocal = s + (1-diff)*sxz[:,top_int+zz] + diff*sxz[:, top_int+zz+1] #the reason for zsiz+1

    F = howieWhelan(F, XG, X0I, slocal, dt, zz)
#print(F)
F = cp.transpose(cp.squeeze(F), [2,0,1])
print(cp.shape(F))
Ib = abs(F[0]*cp.conjugate(F[0]))
Id = abs(F[1]*cp.conjugate(F[1]))




fig = plt.figure(figsize=(16, 10))
fig.add_subplot(2, 1, 1)
plt.imshow(cp.asnumpy(Ib))
fig.add_subplot(2, 1, 2)
plt.imshow(cp.asnumpy(Id))

'''


