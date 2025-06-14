import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt

def dropout_forward(x, p=0.5):
    assert 0 <= p < 1

    if p == 0:
        return x
    if p == 1:
        return torch.zeros_like(x)
    
    mask = torch.rand_like(x) > p
    return x * mask.float() / (1 - p) # 这里的1-p是为了保持期望值不变

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


# 模型封装
class net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(net, self).__init__()
        self.training = is_training # 必须是self.training, 因为继承nn.Module。当执行model.eval()时，is_training会被设置为False
        self.linear1 = nn.Linear(num_inputs, num_hiddens1)
        self.linear2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.linear3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape((-1, num_inputs))

        h1 = self.relu(self.linear1(x))
        if self.training:
            h1 = dropout_forward(h1, dropout_rate1)

        h2 = self.relu(self.linear2(h1))
        if self.training:
            h2 = dropout_forward(h2, dropout_rate2)

        return self.linear3(h2)
    
def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)
    return (y_hat == y).float().sum().item()


if __name__ == "__main__":
    
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # 初始化模型
    net_model = net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_model.parameters(), lr=learning_rate)

    loss_record_train = []
    acc_record_train = []
    acc_record_test = []

    # 训练模型
    for epoch in range(num_epochs):

        # ===== Note =====
        # 1. 训练损失不能和测试损失一起计算，因为「测试损失」需要在模型评估模式下计算，而「训练损失」需要在模型训练模式下计算。
        # 2. 「训练损失」和「训练准确率」都需要在模型训练模式下计算，而「测试准确率」需要在模型评估模式下计算。
        # 3. 如果在评估模式下计算「训练准确率」，会导致其准确率曲线和「测试准确率」曲线基本相同，因为在评估模式下，dropout被关闭，模型的输出是确定的。


        net_model.train()  # 设置模型为训练模式
        loss_train = 0.0
        acc_train = 0.0
        for X, y in train_iter:
            y_hat = net_model(X)
            l = loss(y_hat, y)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            loss_train += l.item()
            acc_train += accuracy(y_hat, y)

        loss_train /= len(train_iter)
        acc_train /= len(train_iter.dataset)

        loss_record_train.append(loss_train)
        acc_record_train.append(acc_train)
        print(f"Epoch {epoch + 1}, Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.4f}")

        net_model.eval()  # 设置模型为评估模式，关闭dropout
        acc_test = 0.0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net_model(X)
                acc_test += accuracy(y_hat, y)
        acc_test /= len(test_iter.dataset)
        acc_record_test.append(acc_test)
        print(f"Epoch {epoch + 1}, Test Acc: {acc_test:.4f}")

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
