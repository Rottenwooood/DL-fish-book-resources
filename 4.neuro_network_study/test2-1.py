import numpy as np
# mean squared error
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

# cross entropy error
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
ans = cross_entropy_error(np.array(y),np.array(t))
print()