import torch
from torch import nn
import matplotlib.pyplot as plt

# 超参数
num_epochs = 100    # 迭代次数
lr = 0.003          # 学习率
lambda_ = 3         # L2正则化系数
num_inputs = 200    # 输入特征数
batch_size = 5      # 批量大小

# 生成数据
def generate_data(w, b, num_samples=100):
    print(f"w.shape: {w.shape}, b.shape: {b.shape}")
    
    x = torch.normal(0, 1, (num_samples, len(w)))
    y = x @ w + b # b使用了广播机制
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")

    y += torch.normal(0, 0.01, y.shape)  # Add noise
    print(f"y.shape after noise: {y.shape}")
    return x, y

# 初始化参数
def initialize_weights(m):
    # 为什么要单独定义一个函数来初始化权重？
    # 而不是直接在模型定义时使用nn.init.normal_()？
    # 这是因为使用函数可以更灵活地应用于不同的层，例如针对nn.Linear层，我们使用正则化初始化。
    # 后续的模型可能会更加复杂，包含不同类型的层，使用函数可以更方便地进行统一管理。
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1)  # 初始化权重
        nn.init.zeros_(m.bias)  # 初始化偏置为0

if __name__ == "__main__":

    # 生成数据
    num_training_samples = 20
    num_testing_samples = 100
    w_ = torch.ones((num_inputs, 1)) * 0.01
    b_ = torch.ones(1) * 0.05
    x_train, y_train = generate_data(w_, b_, num_training_samples)
    x_test, y_test = generate_data(w_, b_, num_testing_samples)

    # 数据迭代
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 一般会将训练集打乱
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型
    net = nn.Sequential(
        nn.Linear(num_inputs, 1, bias=True)  # 使用nn.Linear定义线性层
    )

    # 初始化参数
    net.apply(initialize_weights)
    
    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化器
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 这里不能直接这么用，需要更加细致地控制L2正则化
    optimizer = torch.optim.SGD(
        [
            {'params': net[0].weight, 'weight_decay': lambda_},  # 对第0层的权重使用L2正则化
            {'params': net[0].bias}  # 偏置不使用L2正则化
        ]
    )

    # 训练模型
    loss_record_train = []
    loss_record_test = []

    for epoch in range(num_epochs):
        for x, y in train_loader:
            # 前向传播
            y_hat = net(x)
            l = loss(y_hat, y)

            # 反向传播
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:

            # 计算训练集损失
            train_loss = 0.0
            for x, y in train_loader:
                y_hat = net(x)
                train_loss += loss(y_hat, y).item()
            train_loss /= len(train_loader)
            loss_record_train.append(train_loss)

            # 计算测试集损失
            test_loss = 0.0
            for x, y in test_loader:
                y_hat = net(x)
                test_loss += loss(y_hat, y).item()
            test_loss /= len(test_loader)
            loss_record_test.append(test_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
    # 绘制损失曲线
    plt.plot(range(1, num_epochs // 5 + 1), loss_record_train, label='Train Loss')
    plt.plot(range(1, num_epochs // 5 + 1), loss_record_test, label='Test Loss')
    plt.xlabel('Epochs (every 5)')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()


