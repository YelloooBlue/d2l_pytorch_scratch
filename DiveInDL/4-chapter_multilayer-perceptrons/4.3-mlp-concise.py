import torch
from torch import nn
from torch.utils import data

import torchvision
from torchvision import transforms

# 超参数
batch_size = 256
num_inputs = 28 * 28    # 展平图像
num_hiddens = 256       # 隐藏层单元数
num_outputs = 10        # 输出类别数

num_epochs = 10         # 训练轮数
learning_rate = 0.1     # 学习率

# 读取数据集封装
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize)) # 优先Resize
    trans = transforms.Compose(trans) # 组合成一个转换
    
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    
    print(f"Number of training examples: {len(mnist_train)}")
    print(f"Number of test examples: {len(mnist_test)}")
    print(f"Shape of training example: {mnist_train[0][0].shape}")
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4))

network = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs)
)

loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)  # 优化器

def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)  # 预测类别
    return (y_hat == y).float().sum().item()  # 计算正确预测的数量

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)

if __name__ == "__main__":

    # 加载数据集
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    
    # 初始化权重
    network.apply(init_weights)

    # 训练模型
    for epoch in range(num_epochs):
        train_loss, train_acc, num_samples = 0.0, 0.0, 0
        
        network.train()  # 设置为训练模式
        for X, y in train_iter:
            optimizer.zero_grad()  # 梯度清零
            y_hat = network(X)  # 前向传播
            l = loss(y_hat, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            train_loss += l.item() * y.size(0)  # 累加损失（乘以每个batch的样本数后，可以做到与在整个数据集上计算损失一致）
            train_acc += accuracy(y_hat, y)  # 累加准确率
            num_samples += y.size(0)  # 累加样本数

        test_loss, test_acc, num_test_samples = 0.0, 0.0, 0
        network.eval()
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = network(X)
                l = loss(y_hat, y)
                test_loss += l.item() * y.size(0)
                test_acc += accuracy(y_hat, y)
                num_test_samples += y.size(0)

        # 打印
        print(f"Epoch {epoch + 1}, "
              f"Train Loss: {train_loss / num_samples:.4f}, "
              f"Train Accuracy: {train_acc / num_samples:.4f}, "
              f"Test Loss: {test_loss / num_test_samples:.4f}, "
              f"Test Accuracy: {test_acc / num_test_samples:.4f}")


        
