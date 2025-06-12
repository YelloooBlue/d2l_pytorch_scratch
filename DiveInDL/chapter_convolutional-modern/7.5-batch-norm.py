import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

# 超参数
batch_size = 256
lr = 1.0
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

# 原LeNet网络结构
# net = nn.Sequential(
#     nn.Conv2d(1, 6, kernel_size=5, padding=2),
#     nn.Sigmoid(),
#     # 6 * 28 * 28
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # 6 * 14 * 14
#     nn.Conv2d(6, 16, kernel_size=5),
#     nn.Sigmoid(),
#     # 16 * 10 * 10
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # 16 * 5 * 5
#     nn.Flatten(),
#     nn.Linear(16 * 5 * 5, 120),
#     nn.Sigmoid(),
#     nn.Linear(120, 84),
#     nn.Sigmoid(),
#     nn.Linear(84, 10)
# )

# 加入BatchNorm的LeNet网络结构
net = nn.Sequential(
    # 1 * 28 * 28

    nn.Conv2d(1, 6, kernel_size=5),
    nn.BatchNorm2d(6), 
    nn.Sigmoid(),
    # 6 * 24 * 24 (28 - 5 + 2 * 0) / 1 + 1 = 24
   
   nn.AvgPool2d(kernel_size=2, stride=2),
    # 6 * 12 * 12 (24 - 2 + 2 * 0) / 2 + 1 = 12

    nn.Conv2d(6, 16, kernel_size=5), 
    nn.BatchNorm2d(16), 
    nn.Sigmoid(),
    # 16 * 8 * 8 （12 - 5 + 2 * 0) / 1 + 1 = 8

    nn.AvgPool2d(kernel_size=2, stride=2), 
    # 16 * 4 * 4 （8 - 2 + 2 * 0）/ 2 + 1 = 4

    nn.Flatten(),
    
    nn.Linear(256, 120), 
    nn.BatchNorm1d(120), 
    nn.Sigmoid(),

    nn.Linear(120, 84), 
    nn.BatchNorm1d(84), 
    nn.Sigmoid(),

    nn.Linear(84, 10)
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
    X = torch.rand(size=(256, 1, 28, 28), dtype=torch.float32) # 注意这里测试的时候 batch_size 不能为1，因为 BatchNorm 需要至少2个样本来计算均值和方差
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
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=28)

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
    plt.savefig("batch_norm_training_record.png", dpi=300, bbox_inches='tight')
