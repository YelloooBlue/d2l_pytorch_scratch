import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
from torchinfo import summary

# 超参数
batch_size = 256
lr = 0.1
num_epochs = 10

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


def init_weights(m):
    """初始化网络参数"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Xavier初始化

def eval_accuracy(net, data_iter, device):
    """计算模型在数据集上的准确率"""
    net.eval()  # 设置为评估模式
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
    return acc_sum / n

# 定义网络
class DenseBlock(nn.Module):
    """DenseNet中的DenseBlock"""
    def __init__(self, num_convs, in_channels, num_channels):
        super(DenseBlock, self).__init__()
        self.net = nn.ModuleList()
        for i in range(num_convs):
            sum_channels = in_channels + i * num_channels
            self.net.append(
                nn.Sequential(
                    nn.BatchNorm2d(sum_channels),
                    nn.ReLU(),
                    nn.Conv2d(sum_channels, num_channels, kernel_size=3, padding=1)
                )
            )

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
        

# 过渡层
class Transition(nn.Module):
    """DenseNet中的过渡层"""
    def __init__(self, in_channels, num_channels):
        super(Transition, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, X):
        return self.net(X)
    
# 定义DenseNet网络结构
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_convs_in_dense_block=[4, 4, 4, 4], num_channels=64):
        super(DenseNet, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(1, num_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # DenseBlock和过渡层
        self.dense_blocks = nn.ModuleList()
        in_channels = num_channels
        for i, num_convs in enumerate(num_convs_in_dense_block):
            self.dense_blocks.append(DenseBlock(num_convs, in_channels, growth_rate))
            in_channels += num_convs * growth_rate

            # 在DenseBlock之间添加过渡层
            if i != len(num_convs_in_dense_block) - 1:
                self.dense_blocks.append(Transition(in_channels, in_channels // 2))
                in_channels //= 2
        
        # 最后的卷积层
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        self.fc = nn.Linear(in_channels, 10)  # 输出10类

    def forward(self, X):
        X = self.relu(self.bn1(self.conv1(X)))
        X = self.maxpool(X)
        for blk in self.dense_blocks:
            X = blk(X)
        X = self.relu2(self.bn2(X))
        X = self.conv2(X)
        X = torch.flatten(X, 1)  # 展平
        X = self.fc(X)  # 全连接层
        return X
    
net = DenseNet()

if __name__ == "__main__":
    
    # 检查CUDA和Metal是否可用
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Metal is available: {torch.backends.mps.is_available()}")
    print(f"Metal is built: {torch.backends.mps.is_built()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 检查Dense块的输出
    blk = DenseBlock(num_convs=2, in_channels=3, num_channels=10)
    X = torch.randn(4, 3, 8, 8)
    Y = blk(X)
    print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

    # 检查过渡层的输出
    trans = Transition(23, 10)
    Y1 = trans(Y)
    print(f"Input shape: {Y.shape}, Output shape: {Y1.shape}")

    # 检查DenseNet网络结构
    summary(DenseNet(), input_size=(256, 1, 96, 96), device=device.type)

    # 初始化权重
    net.apply(init_weights)
    net.to(device)

    # 损失函数及优化器
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)

    # 读取数据
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)

    # 训练
    record_train_loss = []
    record_train_acc = []
    record_test_acc = []

    for epoch in range(num_epochs):

        loss_train_sum, n = 0.0, 0
        acc_train_sum = 0.0
        
        net.train()
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            trainer.zero_grad()
            l.backward()
            trainer.step()

            loss_train_sum += l.item() * y.shape[0]
            acc_train_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        train_loss = loss_train_sum / n
        train_acc = acc_train_sum / n

        record_train_loss.append(train_loss)
        record_train_acc.append(train_acc)
        print(f"epoch {epoch + 1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}")

        # 评估
        net.eval()
        test_acc = eval_accuracy(net, test_iter, device)
        record_test_acc.append(test_acc)
        print(f"\t test acc {test_acc:.4f}")


    # 绘制训练损失和测试准确率(画在同一张图上)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), record_train_loss, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), record_train_acc, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), record_test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Record')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("densenet_training_record.png", dpi=300, bbox_inches='tight')


"""
一种不对的写法，一股脑吧所有层都放在一个Sequential中了
class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, num_channels):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            out_channels = in_channels + i * num_channels # 前面所有卷积层的输出通道数
            layers.append(nn.Conv2d(in_channels=out_channels, out_channels=num_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(num_features=num_channels))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            X = torch.cat((X, Y), dim=1)  # 在通道维度上连接
        return X
"""