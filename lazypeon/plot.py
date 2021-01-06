import numpy as np
from matplotlib import pyplot as plt


mean1, best1=  np.loadtxt("out1", delimiter=' ', unpack=True)
mean2, best2=  np.loadtxt("out2", delimiter=' ', unpack=True)
x = np.arange(mean1.shape[0])

plt.plot(x, best2)
plt.plot(x, best1)
plt.show()