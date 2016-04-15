import numpy as np
import matplotlib.pyplot as plt


def Weibull(x):
    return -2 * x * np.exp(-x * x) * (x<0)

def Frechet(x):
    return 1 / (pow(x,3)) * np.exp(-1 / pow(x,2)) * (x>0)

def Gumbel(x):
    return exp(-x - np.exp(-x))


abs = np.arange(-5, 10, 0.01)
plt.plot(abs,Weibull(abs), ':', label='Weibull')
plt.plot(abs,Gumbel(abs), '--', label='Gumbel')
plt.plot(abs,Frechet(abs), label='Frechet')

plt.legend()
plt.show()
