import numpy as np
import matplotlib.pylab as plt

# 中心差分
def numerical_diff(f,x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def func_1(x):
    return 0.01*x**2 + 0.1*x

def func_2(x):
    x1,x2 = x
    return x1**2 + x2**2

# 梯度
def numerical_grediant(f,x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_grediant(f,x)
        x -= lr * grad
    
    return x
# x = np.arange(0.0,20.0,0.1)
# y = func_1(x)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.plot(x,y)
# plt.show()

a = numerical_grediant(func_2,np.array([3.0,4.0]))
b = numerical_grediant(func_2,np.array([0.0,2.0]))
c = numerical_grediant(func_2,np.array([3.0,0.0]))
print(a,b,c)

init_x = np.array([-3.0,4.0])
d = gradient_descent(func_2,init_x=init_x,lr=0.1,step_num=100)
print(d)