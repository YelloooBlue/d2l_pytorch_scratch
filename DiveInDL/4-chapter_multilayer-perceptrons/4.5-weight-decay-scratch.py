import torch
import matplotlib.pyplot as plt



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

# 计算L2范数惩罚项
def l2_penalty(w):
    return (w ** 2).sum() / 2

# 网络
def net(x, w, b):
    return x @ w + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

if __name__ == "__main__":

    # 超参数
    num_epochs = 100    # 迭代次数
    lr = 0.003          # 学习率
    lambda_ = 3      # L2正则化系数

    # 生成数据
    num_training_samples = 20
    num_testing_samples = 100
    num_inputs = 200
    batch_size = 5

    w_ = torch.ones((num_inputs, 1)) * 0.01
    b_ = torch.ones(1) * 0.05
    x_train, y_train = generate_data(w_, b_, num_training_samples)
    x_test, y_test = generate_data(w_, b_, num_testing_samples)

    # 数据迭代
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 一般会将训练集打乱
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化参数
    w = torch.normal(0, 1, (num_inputs, 1), requires_grad=True)  # 初始化权重
    # w = torch.zeros((num_inputs, 1), requires_grad=True)  # 注意不能初始化为0，因为我们初始化的w_是0.01，这样就没有意义了
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.SGD([w, b], lr=lr)  # 使用SGD优化器

    loss_record_train = []
    loss_record_test = []

    # 训练
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            # 前向传播
            y_hat = net(x_batch, w, b)
            loss = squared_loss(y_hat, y_batch).mean() + lambda_ * l2_penalty(w) # 一般都用mean()来计算平均损失，sum可能会受batch_size影响
            # 反向传播
            optimizer.zero_grad()
            loss.backward() # 在这里mean()也奏效，但「正则项」会被除以batch_size。但如果是sum()，就没有这个问题了
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            # 计算训练集损失
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                y_hat = net(x_batch, w, b)
                train_loss += squared_loss(y_hat, y_batch).mean().item()
            train_loss /= len(train_loader) # 如果上面是mean()，需要除以「批次数」。如果是sum()，则需要除以「样本总数」
            loss_record_train.append(train_loss)

            # 计算测试集损失
            test_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    y_hat = net(x_batch, w, b)
                    test_loss += squared_loss(y_hat, y_batch).mean().item()
            test_loss /= len(test_loader)
            loss_record_test.append(test_loss)

            print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
        
            

    # 绘制损失曲线
    plt.plot(range(1, len(loss_record_train) + 1), loss_record_train, label='Train Loss')
    plt.plot(range(1, len(loss_record_test) + 1), loss_record_test, label='Test Loss')
    plt.xlabel('Epochs (every 5 epochs)')
    plt.ylabel('Loss')
    plt.title('Loss Curve with L2 Regularization')
    plt.legend()
    plt.grid()
    plt.show()