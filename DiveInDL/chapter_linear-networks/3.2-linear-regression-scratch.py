# 尝试自己写一遍线性回归
import torch
import matplotlib.pyplot as plt
import random

# 生成数据集
def generate_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w))) # 正态分布
    y = torch.mv(X, w)
    y += torch.normal(0, 0.01, y.shape) # 噪声

    return X, y #.reshape((-1,1))

# 模拟批量读取数据集
def data_iter(batch_size, features, labels):
    total = len(features)
    indexes = list(range(total))
    random.shuffle(indexes) # 打乱数据集

    for i in range(0, total, batch_size):
        j = torch.tensor(indexes[i:min(i + batch_size, total)])

        yield features[j], labels[j]

# 线性回归模型
def linreg(X, w, b):
    # X.shape = (batch_size, 2)
    # w.shape = (2, 1)
    # 要进行广播
    return torch.matmul(X, w) + b

# 损失函数
def squared_loss(pred, label):
    # pred.shape = (batch_size, 1) 形如列向量
    # label.shape = (batch_size, ) 形如行向量
    # 直接相减会触发广播，即使得结果变为（batch_size, batch_size）
    # 所以要先reshape成（batch_size, 1），实现逐元素相减
    return (pred - label.reshape(-1, 1)) ** 2 / 2

# 优化函数
def sgd(params, lr, batch_size):

    # 禁止梯度计算
    with torch.no_grad():
        for param in params:
            # 此时梯度是对于整个batch的，所以要除以batch_size
            param -= lr * param.grad / batch_size 
            param.grad.zero_() # 清空梯度

        # print(params[0].grad)
        # print(params[1].grad)
        # print(params[0])
        # print(params[1])
        # print('---')
        # break


if __name__ == "__main__":

    # 生成数据集
    w_ = torch.tensor([2, -3.4])
    b_ = 4.2
    features, labels = generate_data(w_, b_, 1000)

    # # 画出数据集
    # plt.scatter(features[:, 1], labels, 1)
    # plt.title('x1 vs y')
    # plt.xlabel('x1')
    # plt.ylabel('y')
    # plt.show()

    # # 取出数据
    # batch_size = 10
    # for X, y in data_iter(batch_size, features, labels):
    #     print(X)
    #     print(y)
    #     break

    # 线性回归模型
    w = torch.normal(0, 0.01, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练模型
    num_epochs = 3
    lr = 0.03
    net = linreg
    loss = squared_loss
    batch_size = 10

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            y_hat = net(X, w, b)
            l = loss(y_hat, y)

            l.sum().backward()

            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    print("w的估计值", w)
    print("b的估计值", b)

    # 画出拟合的直线
    plt.scatter(features[:, 1], labels, 1)
    plt.title('x1 vs y')
    plt.xlabel('x1')
    plt.ylabel('y')
    # 以上跟之前一样
    
    x = torch.arange(-3, 3, 0.1).reshape(-1, 1)
    y = net(torch.cat((torch.ones(x.shape[0], 1), x), 1), w, b)
    plt.plot(x, y.detach(), color='red')
    plt.show()
