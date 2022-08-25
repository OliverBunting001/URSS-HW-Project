import numpy as np
import MyVersion4 as mv4

eps = 0.000001



def gdotR(rD, bscrew, bedge, beUnit, bxu, d2c, nu, gD):
    
    rmag = np.sum(rD*rD, axis=2)**0.5 #matrix
    
    
    ct = np.dot(rD,beUnit)/rmag #matrix
    sbt = np.cross(beUnit,rD)/rmag[...,None] #vector matrix
    st = np.dsplit(sbt, (0,1,2))[2].reshape(xsiz, zsiz+1) #matrix
    #this is fishy - needs clearing up
    
    
    Rscrew_2 = bscrew*(np.arctan(rD_1/rX)-np.pi*(rX<0))/(2*np.pi) #component 3 of vector matrix
    Rscrew = np.dstack((np.zeros((xsiz, zsiz+1, 2)), Rscrew_2)) #vector matrix
    Redge0 = (ct*st)[...,None] * bedge/(2*np.pi*(1-nu)) #vector matrix
    Redge1 = ( ((2-4*nu)*np.log(rmag)+(ct**2-st**2)) /(8*np.pi*(1-nu)))[...,None]*bxu #vector matrix
    R = (Rscrew + Redge0 + Redge1) #vector matrix
    gR = np.dot(R, gD) #matrix
    return gR


def howieWhelan(F_in,Xg,X0i,s,alpha,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real #const
    Xgi = Xg.imag #const

    s = s + eps #xy matrix

    gamma = np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2]) #2vector of gamma xy matrices
        
    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))]) #2vector of q xy matrices

    beta = np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5)) #xy matrix
    #alpha=const
    #scattering matrix
    C=np.array([[np.cos(beta/2), np.sin(beta/2)], #2matrix of xy matrices
                 [-np.sin(beta/2)*np.exp(complex(0,alpha)),
                  np.cos(beta/2)*np.exp(complex(0,alpha))]])
    
    #inverse of C is likewise a 2matrix of xy matrices
    Ci= np.array([[np.cos(beta/2), -np.sin(beta/2)*np.exp(complex(0,-alpha))],
                 [np.sin(beta/2),  np.cos(beta/2)*np.exp(complex(0,-alpha))]])

    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]], dtype=object)
    #gamma/q[0/1] are all xy matrices
    #thus this is a 2matrix of xy matrices
    #0*gamma[0] give these zeroes the right dimensionality
    #np.zeros probably better
    
    
    
    gamma = np.transpose(gamma, (1,2,0)).reshape(xsiz,ysiz,2,1) #xy matrix of gamma 2vectors
    q = np.transpose(q, [1,2,0]).reshape(xsiz,ysiz,2,1) #xy matrix of q 2vectors

    C=np.transpose(C, [2,3,0,1])
    Ci=np.transpose(Ci, [2,3,0,1])
    G=np.transpose(G, [2,3,0,1])


    F_out = C  @ G  @ Ci  @ F_in
    return F_out