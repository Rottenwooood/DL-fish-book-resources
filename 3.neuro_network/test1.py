import numpy as np
import matplotlib.pylab as plt

# 实数
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
    
# numpy array
def _step_function(x):
    # y = x > 0 # bool array
    # return y.astype(np.int) # bool to int
    return np.array(x > 0,dtype = np.int32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maxinum(0,x)

x = np.arange(-5.0,5.0,0.1)
y = sigmoid(x)#_step_function(x)
plt.plot(x,y) # draw
plt.ylim(-0.1,1.1)
plt.show()