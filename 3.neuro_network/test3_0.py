import numpy as np
# neuro network
# 内积

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def identify_function(x):
    return x
# I -> 1
X = np.array([1.0,0.5])# input1
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

A1 = np.dot(X,W1) + B1
Z1 = sigmoid(A1)
print(A1)
print(Z1)

# 1 -> 2
#Z1
W2 = np.array([0.1,0.4],[0.2,0.5],[0.3,0.6])
B2 = np.array([0.1,0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1,W2) + B2
Z2 = sigmoid(A2)
print(A2)
print(Z2)

# 2 -> O
#Z2
W3 = np.array([0.1,0.3],[0.2,0.4])
B3 = np.array([0.1,0.2])

A3 = np.dot(Z2,W3) + B3
Y = identify_function(A3)