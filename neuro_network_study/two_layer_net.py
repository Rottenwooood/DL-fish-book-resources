import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,
                 weight_init_std=0.01):
        ## 存储W1,W2和b1,b2
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    # x->z1->y
    #  W1 W2
    #  b1 b2
    # a1,a2-sigmoid->z1,y
    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = sigmoid(a2)
        
        return y
    
    # 训练数据x，预测值y，正确解集合t
    def loss(self,x,t):
        y = self.predict(x)

        return cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        # y为各个训练数据是正确解的概率，下面取最大值为1
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        # 预测y与正确解t相符的个数/总数
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    # 计算损失函数的梯度
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads
    
    