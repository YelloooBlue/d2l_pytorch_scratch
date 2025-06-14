
import torch
from torch.utils import data
from torch import nn

# 生成数据集
def generate_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.mv(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


if __name__ == "__main__":
    
    # 生成数据集
    w_ = torch.tensor([1, 2.3])
    b_ = -4.5
    features, labels = generate_data(w_, b_, 1000)

    # torch的工具代替data_iter
    batch_size = 10
    dataset = data.TensorDataset(features, labels)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1)) # 两个输入分别代表对x1和x2的权重

    # 初始化第一层参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 损失函数
    loss = nn.MSELoss() # L2损失函数

    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.03) # 随机梯度下降优化器

    # 训练模型
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y) # 计算损失
            l.backward() # 反向传播

            optimizer.step()
            optimizer.zero_grad()

        l = loss(net(features), labels) # 计算损失
        print(f'epoch {epoch + 1}, loss {l:f}')