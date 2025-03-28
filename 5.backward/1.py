# 导入必要的库和模块
import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist
from BACKWARD_TwoLayerNet import TwoLayerNet

import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt


# 加载MNIST数据集
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# 初始化神经网络
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 设置训练参数
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
train_loss_list = []
train_acc_list = []
test_acc_list = []
iter_per_epoch = max(train_size / batch_size, 1)

# 开始训练循环
for i in range(iters_num):
    # 随机选择一个批量的训练数据
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 通过误差反向传播法求梯度
    grad = network.gradient(x_batch, t_batch)

    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 计算损失值
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每完成一个epoch时计算准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Train Acc: {train_acc}, Test Acc: {test_acc}")

# 绘制准确率随训练次数变化的图像
# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制训练集准确率曲线
plt.plot(range(len(train_acc_list)), train_acc_list, label="Train Accuracy", linestyle='-', marker='o')

# 绘制测试集准确率曲线
plt.plot(range(len(test_acc_list)), test_acc_list, label="Test Accuracy", linestyle='--', marker='x')

# 添加标题和标签
plt.title("Accuracy vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

# 添加图例
plt.legend()

# 显示网格
plt.grid(True)

# 保存图像为文件
plt.savefig("accuracy_plot.png")  # 保存为PNG文件
print("图像已保存为 accuracy_plot.png")