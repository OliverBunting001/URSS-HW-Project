import cupy as cp
import numpy as np
from tqdm import tqdm


xsiz, ysiz, zsiz = 9, 3, 5
pix2nm = 0.5
phi,psi,theta=1.4614697276429125, 0.8812217608851565, 0.15492231987081392
dt = 0.3
tanpsi, sinphi,tantheta = 1,1,0.1

test = np.zeros((2,xsiz,zsiz+1))



x_vec = cp.arange(xsiz) + 0.5 - xsiz/2 #x-vector
x_mat = cp.tile(x_vec, (zsiz+1, 1)) #zx-matrix
rX = (cp.transpose(x_mat)).reshape(xsiz,zsiz+1,1) #xz-matrix
#this is all fine

pre_z_vec = (cp.sin(phi) + xsiz*cp.tan(psi)/(2*zsiz))
pre_z_vec = (cp.linspace(0, zsiz, zsiz+1) +0.5 - zsiz/2)* pre_z_vec
z_vec = (  (cp.arange(zsiz+1) + 0.5 - zsiz/2)*(cp.sin(phi)
        + xsiz*cp.tan(psi))/(2*zsiz)  )*dt #z-vector
z_mat = cp.tile(z_vec, (xsiz, 1)) #xz-matrix
rD_1 = z_mat.reshape(xsiz,zsiz+1,1) + rX*cp.tan(theta) #xz matrix (scalar)
rD = cp.concatenate((rX, rD_1, cp.zeros((xsiz, zsiz+1,1))), axis=2) #3vector xz-matrix





#test[0] = cp.asnumpy(rX[:,0]).reshape(np.shape(test[0]))
#test[0,:] = cp.asnumpy(z_vec).reshape(np.shape(test[0,0]))
test[0] = cp.asnumpy(rD_1).reshape(np.shape(test[0]))


rX=np.zeros((xsiz, zsiz+1))
rD=np.zeros((xsiz, zsiz+1, 3))
rD2=np.zeros((xsiz, zsiz+1))
rQ=np.zeros((xsiz, zsiz+1))
for x in range(xsiz):
        for z in range(zsiz+1):
            rX[x][z] = 0.5+x-xsiz/2
            rD2[x][z] = dt*( (0.5+z-zsiz/2)*(sinphi + xsiz*tanpsi/(2*zsiz)) ) + rX[x][z]*tantheta
            rD2[x][z] = dt*(0.5+z-zsiz/2)*(sinphi + xsiz*tanpsi/(2*zsiz))+ rX[x][z]*tantheta

print(rD2)


#test[1] = rX[:,0]
#test[1] = rD[...,1]
#print(test)
#print(test[1]-test[0])


A = sinphi+xsiz*tanpsi/(2*zsiz)
x_vec = cp.arange(xsiz) + 0.5 - xsiz/2 #x-vector
x_mat = cp.tile(x_vec, (zsiz+1, 1)) #zx-matrix
rX = (cp.transpose(x_mat)).reshape(xsiz,zsiz+1)




z_vec = cp.arange(zsiz+1) +0.5 -zsiz/2
z_vec = A*z_vec*dt 
z_mat = cp.tile(z_vec, (xsiz, 1)) + rX*tantheta
print(z_mat)