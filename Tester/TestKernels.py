#CUPY DOCUMENTATION
import cupy as cp
import numpy as np



squared_diff = cp.ElementwiseKernel( #///////ELEMENTWISE KERNEL
   'float32 x, float32 y',           #///////INPUTS
   'float32 z',                      #///////OUTPUTS   
   'z = (x - y) * (x - y)',          #///////BODY CODE
   'squared_diff')                   #///////NAME







howieWhelan = cp.ElementwiseKernel(
        '''
            float32 F_in,
            float32 Xg,
            float32 X0i,
            float32 s,
            float32 alpha,
            float32 t
        ''',
        
        'float32 F_out',
        
        '''
            Xgr = Xg.real
            Xgi = Xg.imag
            s = s + eps
            gamma = np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])
            q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])
            beta = np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))
            C=np.array([[np.cos(beta/2), np.sin(beta/2)],
                        [-np.sin(beta/2)*np.exp(complex(0,alpha)), np.cos(beta/2)*np.exp(complex(0,alpha))]])
            Ci= np.array([[np.cos(beta/2), -np.sin(beta/2)*np.exp(complex(0,-alpha))],
                 [np.sin(beta/2),  np.cos(beta/2)*np.exp(complex(0,-alpha))]])
            G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0],
                [0, np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]])
            F_out = C @ G @ Ci @ F_in
        ''',
        
        'howieWhelan' )





gdotR = cp.ElementwiseKernel(
        '''
            float32 rD,
            float32 bscrew,
            float32 bedge,
            float32 beUnit,
            float32 bxu,
            float32 d2c,
            float32 nu,
            float32 gD
        ''',
        
        'float32 gR',
        
        '''
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
            
        ''',
        
        'gdotR' )





"""
def calculate_deviations(xsiz, zsiz, pix2nm, t, dt, u, g, b, c2d, nu, phi, psi, theta):
    # calculates the local change in deviation parameter s as the z-gradient of the displacement field
    
    #dislocation components & g: in the dislocation reference frame
    bscrew = np.dot(b,u)#NB a scalar
    bedge = c2d @ (b - bscrew*u)#NB a vector
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
                
    plt.imshow(sxz)
    plt.axis("off")    
    return sxz
"""













calculate_deviations = cp.ElementwiseKernel(
        '''
            S xsiz,
            S zsiz,
            S pix2nm,
            S t,
            S dt,
            S u,
            S g,
            S b,
            S c2d,
            S nu,
            S phi,
            S psi,
            S theta
        ''',
        
        'S sxz',
        
        '''
            bscrew = np.dot(b,u)#NB a scalar
            bedge = c2d @ (b - bscrew*u)#NB a vector
            beUnit = bedge/(np.dot(bedge,bedge)**0.5)#a unit vector
            bxu = c2d @ np.cross(b,u)
            gD = c2d @ g
            d2c = np.transpose(c2d)
            dz = 0.01
            deltaz = np.array((0, dt*dz, 0)) / pix2nm
            rX = np.arange(xsiz) + 0.5 + xsiz/2
            rD_1 = (np.arange(zsiz) + 0.5 + zsiz/2)
            
            
            
            
            
        ''',
        
        'gdotR' )


###########################################################
#THE KERNELS ARE THE TASKS WHICH ARE EXECUTED IN PARALLEL #
#       ONLY ONE CHANNEL SHOULD BE CALCULATED BY IT       #
###########################################################














































