import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1001)
y = np.linspace(0, 10, 1001)
z = np.empty((2, 1001))

a = np.sin(x)



xy = np.zeros((1001, 1001))







aim = 1
err = 10 #percent




for i in range(1000):
    for j in range(1000):
        I=x[i]**y[j]
        if I>(aim*(1-err/100)) and I<(aim*(1+err/100)):
            xy[i][j]=1





















plt.plot(xy)






eta = 0.1
A = 0.1
w = 0.2
pi = 3.141592653589
beta = 1


        

def dT(S, T):
    return -eta * T  +  (i - A) * S


def dS(S, T):
    return (i - A) * T  +  (- eta + 2*i*w + 2*pi*i*beta) * S





























