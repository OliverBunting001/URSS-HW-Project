import cupy as np #lol this is going to annoy people

import time
eps=0.000001

#INPUTS

# NB cubic crystals only! Everything here in the crystal co-ordinate system



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
g = np.array((1,1,-1))#Miller indices

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










# the dislocation Burgers vector (Miller Indices)


#b0 = np.array((-0.5, 0.0, -0.5))
#b0 = np.array((0.0, 0.5, -0.5))
b0 = np.array((-0.5, -0.5, 0.0))
#b0 = np.array((eps, 0.0, 0.0))



# defect line direction
u = np.array((5, 2, 3)) #Miller indices










# calculation and image variables



# integration step (decrease this until the image stops changing)
dt = 0.2  # fraction of a slice

# pixel scale is 1 nm per pixel, by default
# We change the effective magnification of the image (more or less pixels)
#by the scale factor pix2nm
# with an according increase (<1) or decrease (>1) in calculation time
pix2nm = 0.5 #nm per pixel

#i.e. 0.5 -> 2pixels along 1nm -> 4pix/1nmsqr
#     x -> 1/x pixels along 1nm -> (1/xsqr) pix/1nmsqr




# pixels arounnd the dislocation - characteristic scale of image
pad = 60 #nm

#Gaussian blur sigma
blursigma = 2.0 #nm



# end of input
















# SETUP



# Normalise
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
    x = np.cross(u, z)
    # angle between dislocation and z-axis
    phi = np.arccos(abs(np.dot(u, z)))







x = x / (np.dot(x, x) ** 0.5)
y = np.cross(z, x) # y is the cross product of z & x






# transformation matrices between simulation frame & crystal frame (co-ordinates system)
c2s = np.array((x, y, z))
s2c = np.transpose(c2s)










# normal and dislocation vectors transform to simulated co-ordinate frame

nS = c2s @ n #@ is matrix multiplication

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
    ysiz = 1 * xsiz # in pixels
    zsiz = zlen # in slices
    print("Dislocation is in the plane of the foil")
    #the plane normal to z isn't necessarily the foil plane is it??


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



print(xsiz,ysiz,zsiz) #x value seems bigger than it should be?

sxz = np.array((xsiz, zsiz+1))



######################
#/CHECK THIS SECTION #
######################

########################
#ALL THE CUPY STUFF HERE #
########################





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

for x in range (xsiz):
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
#            sxz[x,z] = gR#np.sqrt(np.dot(rD,rD))




rD_evrwhr = np.zeros((xsiz, zsiz+1, 3))
#rDmag_evrwhr = np.zeros((xsiz, zsiz+1))
#Rscrew_evrwhr = np.zeros((xsiz, zsiz+1, 3))
#Redge0_evrwhr = np.zeros((xsiz, zsiz+1))
#Redge1_evrwhr = np.zeros((xsiz, zsiz+1))
#R_evrwhr = np.zeros((xsiz, zsiz+1, 3))
gR_evrwhr = np.zeros((xsiz, zsiz+1))


for x in range (xsiz):
    for z in range(zsiz+1):
        rD_evrwhr[x][z][0] = (0.5+x-xsiz/2)/pix2nm
        rD_evrwhr[x][z][1] = (dt*( (0.5+z-zsiz/2)*(np.sin(phi) + xsiz*np.tan(psi)/(2*zsiz)) )
                           + rD_evrwhr[x][z][0]*np.tan(theta))/pix2nm
        rD_evrwhr[x][z][2] = 0
        
        
        
        rDmag_local = (np.dot(rD_evrwhr[x][z], rD_evrwhr[x][z]))**0.5
        ct = np.dot(rD_evrwhr[x][z],beUnit)/rDmag_local
        sbt = np.cross(beUnit,rD_evrwhr[x][z])/rDmag_local
        st = sbt[2]
        
        
        #rDmag_evrwhr[x][z] = (np.dot(rD_evrwhr[x][z], rD_evrwhr[x][z]))**0.5
        
        #Rscrew_evrwhr[x][z][2] = bscrew*(np.arctan(rD_evrwhr[x][z][1]/rD_evrwhr[x][z][0])-np.pi*(rD_evrwhr[x][z][0]<0))/(2*np.pi)
        Rscrew_local = bscrew*(np.arctan(rD_evrwhr[x][z][1]/rD_evrwhr[x][z][0])-np.pi*(rD_evrwhr[x][z][0]<0))/(2*np.pi)
        Redge0_local = bedge*ct*st/(2*np.pi*(1-nu))
        Redge1_local = bxu*( ( (2-4*nu)*np.log(rDmag_local)+(ct**2-st**2) )/(8*np.pi*(1-nu)))
        #R_evrwhr[x][z] = Rscrew_local + Redge0_local + Redge1_local
        R = Rscrew_local + Redge0_local + Redge1_local
        gR = np.dot(gD, R)
        
        
        #I think I haven't done anything. Time for version3







































cl_hw = funcs_0.ClHowieWhelan()
cl_hw.calculate_deviations(xsiz, zsiz, pix2nm, dt, u, g, b, c2d, d2c, nu, phi, psi, theta)
# intermediate output, for debugging
# sxz=cl_hw.get_sxz_buffer(xsiz, (zsiz+1))
# plt.imshow(sxz)
# plt.axis("off")    
Ib, Id = cl_hw.calculate_image(xsiz, ysiz, zsiz, pix2nm, t, dt, s, 
                               Xg, X0i, g, b, nS, psi, theta, phi)


end_time = time.perf_counter()
duration = end_time - start_time
print("Main loops took: " + str(duration) + " seconds")














#####################
#PRINTING MACHINERY #
#####################




ker = int(7/pix2nm+0.5)+1
# Gaussian blur (input, kernel size, sigma)
Ib2= cv.GaussianBlur(Ib,(ker,ker),blursigma)
Id2= cv.GaussianBlur(Id,(ker,ker),blursigma)

if save_images:
    Ib2= cv.GaussianBlur(Ib,(7,7),blursigma)
    Id2= cv.GaussianBlur(Id,(7,7),blursigma)
else:
    Ib2=Ib
    Id2=Id
#pixels at 0 and 1 to allow contrast comparisons
#Ib2[0,0]=0
#Ib2[0,1]=1

################################## Output image notation stuff
# g-vector on image is leng pixels long
leng = pad / 4
gDisp = c2s @ g
gDisp = leng * gDisp / (np.dot(gDisp, gDisp) ** 0.5)
bDisp1 = c2s @ b
bDisp1 = leng * bDisp1 / (np.dot(bDisp1, bDisp1) ** 0.5)


################################## Output image display
fig = plt.figure(figsize=(8, 4))
fig.add_subplot(2, 1, 1)
plt.imshow(Ib2)
plt.axis("off")
pt = int(pad / 2)
heady=6/pix2nm
plt.arrow(pt, pt, gDisp[1], -gDisp[0],
      shape='full', head_width=heady, head_length=2*heady)
plt.annotate("g", xy=(pt + 2, pt + 2))
fig.add_subplot(2, 1, 2)
plt.imshow(Id2)
plt.axis("off")
if (abs(bDisp1[0]) + abs(bDisp1[1])) < eps:  # Burgers vector is along z
    plt.annotate(".", xy=(pt, pt))
else:
    plt.arrow(pt, pt, bDisp1[1], -bDisp1[0],
              shape='full', head_width=heady, head_length=2*heady)
plt.annotate("b", xy=(pt + 2, pt + 2))
bbox_inches = 0

################################## Image saving
if save_images:

    # save & show the result
    t = t * pix2nm
    imgname = "BF_t=" + str(int(t)) + "_s" + str(s) + suffix + ".tif"
    Image.fromarray(Ib2).save(imgname)
    imgname = "DF_t=" + str(int(t)) + "_s" + str(s) + suffix + ".tif"
    Image.fromarray(Id2).save(imgname)

    plotnameP = "t=" + str(int(t)) + "_s" + str(s) + suffix + ".png"
    # print(plotnameP)
    plt.savefig(plotnameP)  # , format = "tif")

tic = time.perf_counter()
print("Full function took: " + str(tic - toc) + " seconds")




