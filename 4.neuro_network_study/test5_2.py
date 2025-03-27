import numpy as np
from mnist import load_mnist
from two_layer_net import TwoLayerNet

# 1.获取训练数据和测试（监督）数据，后者用于测试模型的泛化能力
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_laobel = True)

# 2.记录损失函数值的list，其是表示拟合程度的连续的指标
    # 损失函数包括均方误差和交叉熵误差
    # mean_squared_error
    # cross_entropy_error
    
train_loss_list = []
train_acc_list = []
test_acc_list = []

# 超参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max( train_size / batch_size,1)

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    grad = network.numerical_gradient(x_batch,t_batch)

    # 更新参数
    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 记录学习参数
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))