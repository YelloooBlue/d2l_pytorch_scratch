import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

# 超参数
batch_size = 256
num_inputs = 28 * 28    # 展平图像
num_outputs = 10        # 输出类别数
num_hiddens1 = 256      # 隐藏层单元数
num_hiddens2 = 256      # 隐藏层单元数
dropout_rate1 = 0.2     # 第一层的dropout率
dropout_rate2 = 0.5     # 第二层的dropout率

num_epochs = 10         # 训练轮数
learning_rate = 0.5     # 学习率

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

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout_rate1),  # 第一层的dropout
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout_rate2),  # 第二层的dropout
    nn.Linear(num_hiddens2, num_outputs)
)

# def init_weights(m):
#     if type(m) == nn.Linear:
#         nn.init.normal_(m.weight, mean=0, std=0.01)


if __name__ == "__main__":
    
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # net.apply(init_weights)  # 初始化权重
    # 这里不初始化权重也是可以的，因为nn.Linear默认会使用kaiming初始化，如果需要改成正态（高斯）分布，可以使用nn.init.normal_方法
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    loss_record_train = []
    acc_record_train = []
    acc_record_test = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        
        net.train()  # 设置为训练模式
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_acc += (y_hat.argmax(dim=1) == y).sum().item()

        train_loss /= len(train_iter)
        train_acc /= len(train_iter.dataset)
        loss_record_train.append(train_loss)
        acc_record_train.append(train_acc)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        net.eval()
        test_acc = 0.0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                test_acc += (y_hat.argmax(dim=1) == y).sum().item()
        test_acc /= len(test_iter.dataset)
        acc_record_test.append(test_acc)
        print(f"Test Accuracy: {test_acc:.4f}")

    print("Training complete.")

    # 绘制损失曲线
    plt.plot(loss_record_train, label='Train Loss')
    plt.plot(acc_record_train, label='Train Accuracy')
    plt.plot(acc_record_test, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.show()
