import cupy as cp
import numpy as np

xsiz=12
ysiz=10
zsiz=6
zlen=1
X0i = 1000.  # nm
Xg = 3 + 2000j  # nm

F = cp.zeros((xsiz, ysiz, 2, 1))

slocal=cp.arange(xsiz*(zsiz+1)).reshape(xsiz,zsiz+1)

def howieWhelanCP(F_in,Xg,X0i,s,alpha,t):
    '''
    X0i  scalar
    Xg = scalar
    s =  xy matrix
    alpha = scalar
    t = scalar
    '''
    Xgr = Xg.real #const
    Xgi = Xg.imag #const
    s = s + eps #xy matrix
    gamma = cp.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2], dtype=cp.float64) #2vector of gamma xy matrices
    q = cp.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))], dtype=cp.float64) #2vector of q xy matrices
    beta = cp.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5), dtype=cp.float64) #xy matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)], #2matrix of xy matrices
                 [-cp.sin(beta/2)*cp.exp(complex(0,alpha)),
                  cp.cos(beta/2)*cp.exp(complex(0,alpha))]], dtype=cp.float64)
    Ci= cp.array([[cp.cos(beta/2), -cp.sin(beta/2)*cp.exp(complex(0,-alpha))],
                 [cp.sin(beta/2),  cp.cos(beta/2)*cp.exp(complex(0,-alpha))]], dtype=cp.float64)
    G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]], dtype=cp.float64)
    gamma = cp.transpose(gamma, (1,2,0)).reshape(xsiz,ysiz,2,1) #xy matrix of gamma 2vectors
    q = cp.transpose(q, [1,2,0]).reshape(xsiz,ysiz,2,1) #xy matrix of q 2vectors
    C=cp.transpose(C, [2,3,0,1])
    Ci=cp.transpose(Ci, [2,3,0,1])
    G=cp.transpose(G, [2,3,0,1])
    F_out = C  @ G  @ Ci  @ F_in
    return F_out




def howieWhelanPY(F_in,Xg,X0i,s,alpha,t):
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




F=cp.asnumpy(F)

for z in (range(zlen)):

    
    alpha = 0.0
    F = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm) #xy matrix of 2vectors

print(F)



for x in (range(xsiz)):
    for y in range(ysiz):
        F = np.array(xsiz, ysiz, 2, 1)
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


            # stacking fault shift is present between the two dislocations
            # if firs < x < las and h+z-int(zrange / 2) == 0:
            #     alpha = 2*np.pi*np.dot(g,b1)
            # else:
            alpha = 0.0
            F[x,y] = howieWhelan(F,Xg,X0i,slocal,alpha,dt*pix2nm)

print(F)


