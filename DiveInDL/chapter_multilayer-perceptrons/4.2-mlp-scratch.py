import torch
from torch.utils import data
from torch import nn
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

# 激活函数
def relu(X):
    return torch.max(X, torch.tensor(0.0, device=X.device))

# 网络
def network(X, w1, b1, w2, b2):
    X = X.reshape((-1, num_inputs))  # 展平图像
    H = relu(torch.matmul(X, w1) + b1)  # 隐藏层
    return torch.matmul(H, w2) + b2  # 输出层

    # 另一种写法
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)

# 确率
def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)  # 预测类别
    return (y_hat == y).float().sum().item()  # 计算正确预测的数量




if __name__ == "__main__":

    # 加载数据集
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=28)

    # 初始化参数
    W1 = nn.Parameter(
        torch.randn((num_inputs, num_hiddens), dtype=torch.float32) * 0.01)
    b1 = nn.Parameter(torch.zeros(num_hiddens, dtype=torch.float32))
    W2 = nn.Parameter(
        torch.randn((num_hiddens, num_outputs), dtype=torch.float32) * 0.01)
    b2 = nn.Parameter(torch.zeros(num_outputs, dtype=torch.float32))
    params = [W1, b1, W2, b2]


    # ===
    loss = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(params, lr=0.1)  # 优化器


    for epoch in range(num_epochs):

        loss_sum = 0.0
        acc_train = 0.0
        acc_test = 0.0
        
        for X, y in train_iter:
            # 前向传播
            y_hat = network(X, W1, b1, W2, b2)
            l = loss(y_hat, y)
            # 反向传播
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss_sum += l.item() * y.shape[0]

        loss_sum /= len(train_iter.dataset)

        with torch.no_grad():
            # 计算训练集准确率
            for X, y in train_iter:
                y_hat = network(X, W1, b1, W2, b2)
                acc_train += accuracy(y_hat, y)
            acc_train /= len(train_iter.dataset)

            # 计算测试集准确率
            for X, y in test_iter:
                y_hat = network(X, W1, b1, W2, b2)
                acc_test += accuracy(y_hat, y)
            acc_test /= len(test_iter.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
                f"Loss: {loss_sum:.4f}, "
                f"Train Acc: {acc_train:.4f}, "
                f"Test Acc: {acc_test:.4f}")



