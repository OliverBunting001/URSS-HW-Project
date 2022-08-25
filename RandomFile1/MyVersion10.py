import cupy as cp
from tqdm import tqdm
import matplotlib.pyplot as plt










def howieWhelan(F_in,Xg,X0i,s,alpha,t):
    #for integration over n slices
    # All dimensions in nm
    Xgr = Xg.real
    Xgi = Xg.imag

    #s = s + eps

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
    Ci = cp.transpose(C, [0,1,2,3])
    G = cp.transpose(G, [2,3,0,1])

    F_out = C @ G @ Ci @ F_in


    Ib1 = cp.squeeze(F_out[...,0,:])
    Id1 = cp.squeeze(F_out[...,1,:])
    print(cp.shape(Ib1))
    
    Ib = abs(Ib1*cp.conjugate(Ib1))
    Id = abs(Id1*cp.conjugate(Id1))
    return Ib, Id




a = 0#999999999999999999999999999
X0I = 400
XGr = 10
XG = XGr + 1j * X0I*(1+a)

ss=0.08



tlen = 200
xsiz=10
t= cp.tile(cp.arange(tlen), (xsiz,1))



ss = ss * cp.ones((xsiz,tlen))



F0=cp.array([1,0])
F= cp.transpose(cp.tile(F0, [xsiz,tlen,1]), [2,0,1]).reshape(xsiz,tlen, 2,1)

I = cp.zeros((2,xsiz,tlen))


I[0], I[1] = howieWhelan(F, XG,X0I,ss,0,0.5)


plt.imshow(cp.asnumpy(I[0]))
#plt.plot((I[1]))
#plt.show()



print("A =", XGr/(X0I*(1+a)) )
print("N =", XGr/X0I)

'''
plt.plot(cp.cos(cp.cos(t/10)))
plt.show()'''