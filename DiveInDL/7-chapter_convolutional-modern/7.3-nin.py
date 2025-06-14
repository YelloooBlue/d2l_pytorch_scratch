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

# 定义网络结构
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    """定义NIN块"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1卷积
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1卷积
        nn.ReLU()
    )

net = nn.Sequential(
    # 1 * 224 * 224
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    # 96 * 54 * 54 (224 - 11 + 2 * 0) / 4 + 1 = 54
    nn.MaxPool2d(3, stride=2),
    # 96 * 26 * 26 (54 - 3 + 2 * 0) / 2 + 1 = 26
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    # 256 * 26 * 26 (26 - 5 + 2 * 2) / 1 + 1 = 26
    nn.MaxPool2d(3, stride=2),
    # 256 * 12 * 12 (26 - 3 + 2 * 0) / 2 + 1 = 12
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    # 384 * 12 * 12 (12 - 3 + 2 * 1) / 1 + 1 = 12
    nn.MaxPool2d(3, stride=2),
    # 384 * 5 * 5 (12 - 3 + 2 * 0) / 2 + 1 = 5
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    # 10 * 5 * 5 (5 - 3 + 2 * 1) / 1 + 1 = 5
    nn.AdaptiveAvgPool2d((1, 1)), # 自适应平均池化，将输出大小调整为1x1
    nn.Flatten() # 将四维的输出转成二维的输出，其形状为(批量大小,10)
) 


if __name__ == "__main__":
    
    # 检查Metal是否可用
    print(f"Metal is available: {torch.backends.mps.is_available()}")
    print(f"Metal is built: {torch.backends.mps.is_built()}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 检查网络结构
    X = torch.rand(size=(1, 1, 224, 224), dtype=torch.float32)
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
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

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
