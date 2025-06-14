# 尝试自己写一遍线性回归
import torch
import matplotlib.pyplot as plt
import random

# 生成数据集
def generate_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mv(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y

# 模拟批量读取数据集
def data_iter(batch_size, features, labels):
    total = len(features)
    indexes = list(range(total))
    random.shuffle(indexes)
    for i in range(0, total, batch_size):
        j = torch.tensor(indexes[i: min(total, i+batch_size)])

        yield features[j], labels[j]

# 线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 损失函数
def squared_loss(pred, label):
    return (pred - label.reshape(pred.shape)) ** 2 / 2

# 优化函数
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == "__main__":

    # 生成数据集
    w_ = torch.tensor([1, 2.3])
    b_ = -4.5
    features, labels = generate_data(w_, b_, 1000)

    # 画出数据集
    plt.scatter(features[:,1], labels, 1)
    plt.show()
   
    # 模拟取出数据
    
    # 初始化权重
    w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练模型
    num_epochs = 3
    lr = 0.03
    net = linreg
    loss = squared_loss
    batch_size = 10

    for i in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            pred = net(X, w, b)
            l = loss(pred, y)
            l.sum().backward()

            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_loss = loss(net(features, w, b),labels)
            print(f"epochs:{i}\t loss:{float(train_loss.mean()):f}")
    
    # 打印结果
    print(f"w:{w}\nb:{b}")

    # 画出拟合的直线
    plt.scatter(features[:,1], labels, 1)
    x1 = torch.arange(-3, 3, 0.1).reshape(-1, 1) # 变成列向量
    x0 = torch.ones(x1.shape[0], 1)

    print(x0.shape,x1.shape)

    x = torch.cat((x0, x1), 1)
    y = net(x, w, b)
    plt.plot(x1, y.detach(), c='red')
    plt.show()

