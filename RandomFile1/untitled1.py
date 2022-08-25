import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
eps = 0.00000000001

'''
def HowieWhelan(F_in, ss, XGr, XGi, X0i, t): #ss takes the role of slocal for the avoidance of ambiguity
    
    sXsqr = ss**2 * XGr**2
    beta = cp.arccos( cp.sqrt( sXsqr/(1+sXsqr) ) )
    gamma = 0.5* cp.array([ (ss - 1/XGr * cp.sqrt(1+sXsqr)) , (ss + 1/XGr * cp.sqrt(1+sXsqr)) ])
    q = 0.5* cp.array(( 1/X0i - 1/XGi * 1/(1+sXsqr)**0.5, 1/X0i + 1/XGi * 1/(1+sXsqr)**0.5 ))


    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)],
                [-cp.sin(beta/2),cp.cos(beta/2)]])
    Ci = cp.transpose(C, [1,0,2,3])
    G=cp.array([[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]])

    C=cp.transpose(C, [2,3,0,1])
    Ci=cp.transpose(Ci, [2,3,0,1])
    G=cp.transpose(G, [2,3,0,1])


    F_out = C  @ G  @ Ci  @ F_in
    return F_out
'''



def howieWhelan(F_in,ss, Xgr, Xgi,X0i,t):
    #for integration over n slices
    # All dimensions in nm


    ss = ss + eps
    w = ss*Xgr

    N = Xgr/X0i
    A = Xgr/Xgi



    gamma = 1/(2*Xgr) * cp.array([(w-(w**2+1)**0.5), (w+(w**2+1)**0.5)])

    q =  1/(2*Xgr) * cp.array([N - A*(1+w**2)**-0.5 , N + A*(1+w**2)**-0.5 ])
    
    beta = cp.arccos( w * (1+w**2)**-0.5 )

    #scattering matrix
    C=cp.array([[cp.cos(beta/2), cp.sin(beta/2)],
                [-cp.sin(beta/2), cp.cos(beta/2)]]).reshape(2,2)

    #inverse of C is just its transpose
    Ci=cp.transpose(C)

    G=cp.array(   [[cp.exp(2*cp.pi*1j*(gamma[0]+1j*q[0])*t), 0],
                [0, cp.exp(2*cp.pi*1j*(gamma[1]+1j*q[1])*t)]]    , dtype=complex)

    mat = C @ G @ Ci
    #dett = cp.linalg.det(mat)
    
    F_out = mat @ F_in
    
    return F_out
'''

def HowieWhelan(F_in, ss, XGr, XGi, XOi, dz):
    A = XGr/XGi
    N = XGr/XOi
    w = ss*XGr
    
    gamma_plus = cp.pi * (-N + 1j*(w + cp.sqrt(w**2 + 1 - A**2 + 2*A*1j)) )
    gamma_minus = cp.pi * (-N + 1j*(w - cp.sqrt(w**2 + 1 - A**2 + 2*A*1j)) )
    
    gamma_plus_red = cp.pi * 1j*(w + cp.sqrt(w**2 + 1 - A**2 + 2*A*1j))
    gamma_minus_red = cp.pi * 1j*(w - cp.sqrt(w**2 + 1 - A**2 + 2*A*1j))  
    
    
    delta_gamma = gamma_plus - gamma_minus

    coeffs = cp.array([gamma_minus_red/delta_gamma,
                       gamma_plus_red/delta_gamma  ])
    
    exps = cp.array([cp.exp(gamma_plus*dz), -cp.exp(gamma_minus*dz)])
    
    m11 = -cp.dot(coeffs , exps)
    m22 = cp.dot(coeffs , cp.flip(exps))
    m12 = cp.pi * (1j-A)/delta_gamma * cp.sum(exps)
    
    M = cp.array([[m11,m12], [m12, m22]])
    
    
    F_out = M @ F_in
    return F_out

'''







s=0.1
X0i = 10  # nm
XgR = 30

A=0 #Xgi>=X0i -> A>=0
Xgi = X0i*(1+A)

Xg = XgR + 1j*Xgi  # nm







'''
width=100
height=2000
length=100
'''

#F0 = cp.concatenate( (cp.ones((width,length,1)), cp.zeros((width,length,1))) , axis=2)
#print(F0)

#F=cp.copy(F0)[...,None]




tlen=120
t=cp.arange(tlen)


F0=cp.array([1,0])
F = cp.copy(F0)
slocal = s

I = cp.zeros((2,tlen))
#Ib = I[0]
#Id = I[1]









for i in tqdm(t):
    F_out = howieWhelan(F, slocal, XgR, Xgi, X0i, i)
    x,y=F_out
    I[0,i], I[1,i] = abs(x*cp.conjugate(x)), abs(y*cp.conjugate(y))


plt.plot(cp.asnumpy((I[0])))
plt.plot(cp.asnumpy((I[1])))
plt.show()








































'''
#x=cp.arange(width)
#y=cp.arange(length)
#z=cp.arange(height)



def plane(z):
    yside = length* (height-z)/height
    yside_int = int(yside)
    diff = yside-yside_int
    
    sxy = cp.concatenate( (cp.ones((width, yside_int)), cp.zeros((width,length-yside_int))), axis=1)
    return sxy



for z in tqdm(range(height)):
    slocal = s*plane(z)
    F = HowieWhelan(F, slocal, Xgr, Xgi, X0i, 1)



Ib = cp.squeeze(F[...,0,:])
Id = cp.squeeze(F[...,1,:])

Ib = abs((Ib*cp.conjugate(Ib)))
Id = abs((Id*cp.conjugate(Id)))


plt.plot(cp.asnumpy(Ib[0]))

plt.show()

#plt.imshow(cp.asnumpy(Ib))
'''
