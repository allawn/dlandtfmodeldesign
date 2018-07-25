import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
        #a.append(math.log(1+math.exp(item)))
        #a.append((math.exp(item)-math.exp(-item))/(math.exp(item)+math.exp(-item)))
        # if item>=0:
        #     a.append(item)
        # else:
        #     a.append(0)
    return a

x = np.arange(-10., 10., 0.2)
sig = sigmoid(x)

plt.plot(x,sig)
plt.show()
