import torch
import torchvision
from torch.utils import data
from torchvision import transforms

import matplotlib.pyplot as plt

# # 标签向量转文本
# def get_fashion_mnist_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]

# # 显示子图
# def show_some_img(images, num_rows, num_cols, titles=None, scale=1.5):
#     # 检查输入参数
#     if len(images) != num_rows * num_cols:
#         raise ValueError("The number of images must be equal to num_rows * num_cols.")
#     if titles is not None and len(titles) != num_rows * num_cols:
#         raise ValueError("The number of titles must be equal to num_rows * num_cols.")

#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten() 
#     for i, ax in enumerate(axes):
#         ax.imshow(images[i].reshape(28, 28).numpy(), cmap='gray')
#         ax.axis('off')
#         if titles is not None:
#             ax.set_title(titles[i])
#     plt.show()

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

# softmax函数
def softmax(X):
    X_exp = torch.exp(X) # 形状是(batch_size, num_outputs)
    partition = X_exp.sum(dim=1, keepdim=True) # 形状是(batch_size, 1)
    return X_exp / partition # 形状是(batch_size, num_outputs)，用了广播机制

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y]) # 形状是(batch_size,)

# 网络
def net(X):
    return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

# 准确率（精度）
def accuracy(y_hat, y):
    # 一个batch_size中有多少个样本预测正确
    y_hat = torch.argmax(y_hat, dim=1) # 形状是(batch_size,)
    return (y_hat == y).float().sum().item() # 形状是(batch_size,)

# 动态绘图类
class TrainPlotter:
    # 一个简单的封装，用于动态绘制训练过程中的损失和准确率

    def __init__(self, title="Training Progress", num_epochs=None):
        plt.ion()  # 开启交互模式
        self.epochs = []
        self.losses = []
        self.train_accs = []
        self.test_accs = []

        # 创建图形和坐标轴
        self.fig, self.ax = plt.subplots()

        # 初始化绘图线
        self.loss_line, = self.ax.plot([], [], 'r-', label='Loss')
        self.train_accs_line, = self.ax.plot([], [], 'b-', label='Train Accuracy')
        self.test_acc_line, = self.ax.plot([], [], 'g--', label='Test Accuracy')

        # 标题
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Value')
        self.ax.set_title(title)

        self.ax.legend()    # 添加图例
        self.ax.grid(True)  # 添加网格线

        plt.show(block=False)  # 非阻塞显示图形

    def update(self, epoch, loss, train_acc, test_acc):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.train_accs.append(train_acc)
        self.test_accs.append(test_acc)

        # 更新数据
        self.loss_line.set_data(self.epochs, self.losses)
        self.train_accs_line.set_data(self.epochs, self.train_accs)
        self.test_acc_line.set_data(self.epochs, self.test_accs)

        # 设置坐标轴范围
        self.ax.set_xlim(1, num_epochs if num_epochs else epoch + 1)
        self.ax.set_ylim(0, 1)


        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # 必须 pause 才会实时更新

    def close(self):
        plt.ioff()
        plt.show()


if __name__ == "__main__":

    batch_size = 256
    mnist_train, mnist_test = load_data_fashion_mnist(batch_size, resize=28)
   
    num_inputs = 28 * 28 # 展平图像
    num_outputs = 10

    # 初始化
    w = torch.normal(0, 0.01, (num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 训练
    num_epochs = 10
    lr = 0.1
    net = net
    loss = cross_entropy
    optimizer = torch.optim.SGD([w, b], lr=lr)

    plotter = TrainPlotter(title="Softmax Regression Training Progress (lr={})".format(lr), num_epochs=num_epochs)


    # 训练循环
    for epoch in range(num_epochs):

        loss_sum = 0.0
        train_acc = 0.0
        test_acc = 0.0

        # 训练
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 处理每一个batch
        for X, y in mnist_train:
            y_hat = net(X)
            l = loss(y_hat, y)

            # 更新梯度
            optimizer.zero_grad()
            l.mean().backward()  # 计算梯度
            optimizer.step()

            loss_sum += l.mean().item()  # 累加损失

        loss_sum /= len(mnist_train)
        
        # 评估
        with torch.no_grad():  # 不需要计算梯度
            # 计算训练集准确率
            for X, y in mnist_train:
                y_hat = net(X)
                train_acc += accuracy(y_hat, y)

            train_acc /= len(mnist_train.dataset)

            # 计算测试集准确率
            for X, y in mnist_test:
                y_hat = net(X)
                test_acc += accuracy(y_hat, y)
            test_acc /= len(mnist_test.dataset)

        print(f"Epoch {epoch + 1}, Loss: {loss_sum:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
       

        # 更新绘图
        plotter.update(epoch + 1, loss_sum, train_acc, test_acc)
    plotter.close()