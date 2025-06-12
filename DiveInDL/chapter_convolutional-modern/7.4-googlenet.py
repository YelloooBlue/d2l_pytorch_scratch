import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

# 超参数
batch_size = 128
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


# 定义GoogleNet网络结构

class Inception(nn.Module):
    """Inception模块"""
    """
        参数：
            in_channels: 输入通道数
            c1: 第一个分支的输出通道数 [1*1卷积核输出通道数]
            c2: 第二个分支的输出通道数 [1*1卷积核 + 卷积层输入通道数, 卷积层输出通道数]
            c3: 第三个分支的输出通道数 [1*1卷积核 + 卷积层输入通道数, 卷积层输出通道数]
            c4: 第四个分支的输出通道数 [1*1卷积核输出通道数]
    """
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()

        # 1. 单个1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )

        # 2. 1x1卷积 + 3x3卷积分支
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )

        # 3. 1x1卷积 + 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )

        # 4. 3x3最大池化 + 1x1卷积分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        # 通过四个分支
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)

        # 将四个分支的输出拼接在一起
        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, dim=1)
    
# Inception前面的卷积层
conv1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
)

conv2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

inception1 = nn.Sequential(
    Inception(192, 64, [96, 128], [16, 32], 32),
    Inception(256, 128, [128, 192], [32, 96], 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

inception2 = nn.Sequential(
    Inception(480, 192, [96, 208], [16, 48], 64),
    Inception(512, 160, [112, 224], [24, 64], 64),
    Inception(512, 128, [128, 256], [24, 64], 64),
    Inception(512, 112, [144, 288], [32, 64], 64),
    Inception(528, 256, [160, 320], [32, 128], 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

inception3 = nn.Sequential(
    Inception(832, 256, [160, 320], [32, 128], 128),
    Inception(832, 384, [192, 384], [48, 128], 128)
)

final_layers = nn.Sequential(
    nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，将输出大小调整为1x1
    nn.Flatten(),  # 将四维的输出转成二维的输出，其形状为(批量大小, 1024)
    nn.Linear(1024, 10)  # 输出层，10个类别
)

net = nn.Sequential(
    conv1,
    conv2,
    inception1,
    inception2,
    inception3,
    final_layers
)

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

    # 检查网络结构
    X = torch.rand(size=(1, 1, 96, 96), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape: \t',X.shape)

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
    plt.savefig("googlenet_training_record.png")  # 保存图像
