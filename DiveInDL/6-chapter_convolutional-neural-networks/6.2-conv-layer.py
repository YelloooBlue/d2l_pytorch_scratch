import torch
from torch import nn

# 卷积 convolution
# 互相关 correlation

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape  # 卷积核的高度和宽度
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # 卷积后会变小
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = torch.sum(X[i:i+h, j:j+w] * K)
    return Y

class myConv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


if __name__ == "__main__":
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    Y = corr2d(X, K)
    print(Y)

    # 模拟边缘检测
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    print(Y.shape)

    # 模拟真实训练
    conv2d = nn.Conv2d(1,1,kernel_size=(1,2), bias=False)
    X = X.reshape((1,1,6,8)) # 加入批次大小和通道
    Y = Y.reshape((1,1,6,7)) # 已知的边缘
    lr = 3e-2

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.3f}')

    print(conv2d.weight)


    