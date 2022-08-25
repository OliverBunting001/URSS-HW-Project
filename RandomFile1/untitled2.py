import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



def howieWhelan(F_in,Xg,X0i,s,alpha,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real
    Xgi = Xg.imag

    s = s + eps

    gamma = np.array([(s-(s**2+(1/Xgr)**2)**0.5)/2, (s+(s**2+(1/Xgr)**2)**0.5)/2])

    q = np.array([(0.5/X0i)-0.5/(Xgi*((1+(s*Xgr)**2)**0.5)),  (0.5/X0i)+0.5/(Xgi*((1+(s*Xgr)**2)**0.5))])

    beta = np.arccos((s*Xgr)/((1+s**2*Xgr**2)**0.5))

    #scattering matrix
    C=np.array([[np.cos(beta/2), np.sin(beta/2)],
                [-np.sin(beta/2),
                 np.cos(beta/2)]])

    #inverse of C is just its transpose
    Ci=np.transpose(C)

    G=np.array([[np.exp(2*np.pi*1j*(gamma[0]+1j*q[0])*t), 0*gamma[0]],
                [0*gamma[0], np.exp(2*np.pi*1j*(gamma[1]+1j*q[1])*t)]], dtype=complex)
    
    '''
    C = np.transpose(C, [2,3,0,1])
    Ci = np.transpose(Ci, [2,3,0,1])
    G = np.transpose(G, [2,3,0,1])
    '''
    

    F_out = C @ G @ Ci @ F_in
    

    Ib1 = np.copy(F_out[0])
    Id1 = np.copy(F_out[1])
    
    Ib = abs(Ib1*np.conjugate(Ib1))
    Id = abs(Id1*np.conjugate(Id1))
    return Ib, Id




a = 9999999999999999999999999
X0I = 999999999999999999999
XGr = 20
XG = XGr + 1j * X0I*(1+a)

ss=0.0
'''
slen=50
xsiz=30


sss = np.tile(np.linspace(-0.1, 0.1, slen), [xsiz,1])


F0 = np.array([1,0])
F = np.tile(F0, [xsiz,slen,1,1])


I2D = np.zeros((2,xsiz,slen))


I2D[0], I2D[1] = howieWhelan(F,XG,X0I,sss,0,10)

'''




























F0 = np.array([1,0])
F = np.copy(F0)

tlen = 100
t=np.arange(tlen)




I = np.zeros((2,tlen))

for i in tqdm(range(tlen)):
    I[0,i], I[1,i] = howieWhelan(F, XG,X0I,ss,0,i)


plt.plot((I[0]))
plt.plot((I[1]))
plt.show()



print("A =", XGr/(X0I*(1+a)) )
print("N =", XGr/X0I)
































'''
slen=50

sI = np.zeros((2,slen,tlen))


sarr = np.linspace(-0.1, 0.1, slen)

for i in tqdm(range(slen)):
    for j in range(tlen):
        sI[0,i,j], sI[1,i,j] = howieWhelan(F, XG,X0I,sarr[i],0,t[j])


plt.imshow(sI[0])
'''

















