# softmax
import numpy as np

# unoptimized
def _softmax(a):
    exp_a = np.exp(a)
    sum_exp_a =  np.sum(exp_a)
    y =  exp_a / sum_exp_a

    return y

# optimized
def softmax(a):
    c = np.max(a) # 取最大值
    exp_a = np.exp(a - c) # 并在之后的运算中将其减去，防止溢出
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)
print(sum(y))
