import torch
from matplotlib import pyplot as plt
from torch import nn

# 超参数
lr = 0.01
num_epochs = 5

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight) # 原地操作初始化

net = nn.Sequential(
    nn.Linear(4, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

if __name__ == "__main__":

    # ============ 生成数据 ============
    
    # 生成1000个时间步的正弦波数据
    num_sample = 1000
    time = torch.arange(1, num_sample+1, dtype=torch.float32)
    x = torch.sin(0.01 * time)
    x += torch.normal(0, 0.2, size=x.shape) # 噪声
    # plt.plot(time, x)
    # plt.show()

    # 将数据转换为输入和目标序列
    tau = 4 # 时间延迟 = 时间步数 = 窗口大小
    feature = torch.zeros((num_sample-tau, tau)) # 特征矩阵，每行是一个时间步的输入，每列是前tau个时间步的值
    
    # 按列进行填充，最终只会有T-tau行
    for i in range(tau):
        feature[:, i] = x[i:num_sample-tau+i]

    # ============ 加载数据 ============

    batch_size = 16
    n_train = 600
    train_data = feature[:n_train]
    train_label = x[tau:n_train+tau] # 目标序列是时间步tau之后的值，也就是说标签是时间步窗口后的第一个
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_data, train_label),
        batch_size=batch_size, shuffle=True
    )

    # ============ 开始训练 ============

    loss = nn.MSELoss(reduction='none') # 取消默认对损失的mean处理
    optimizer = torch.optim.Adam(net.parameters(), lr)

    net.train() # 设置为训练模式
    net.apply(init_weight) # 初始化权重
    for epoch in range(num_epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat.squeeze(), y) # 去掉多余的维度（维度为1的维度）
            
            optimizer.zero_grad()
            l.sum().backward() # 计算平均损失的梯度，手动指定使用sum()而不是mean()
            optimizer.step()

        print(f'epoch {epoch + 1}, loss {l.mean():f}')


    # ============ 单步预测 ============
    """单步预测是指每次只预测下一个时间步的值"""
    """即 P(y_t | y_{t-1}, y_{t-2}, ..., y_{t-tau})"""
    net.eval() # 设置为评估模式
    pred = net(feature) # 预测全部数据，每一行有tau个值

    # 第一个有效预测从tau开始
    plt.plot(time[tau:], x[tau:], label='true')
    plt.plot(time[tau:], pred.detach().squeeze(), label='pred')
    plt.legend()
    plt.show()

    # ============ 多步预测 ============
    """多步预测是指每次预测多个时间步的值"""
    """在本例中，n_train = 600，所以 600 至 1000 的时间步没有标签数据"""
    """所以我们从600开始向后预测 1000 - 600 = 400 个时间步的值"""
    """即 k = 400"""
    """P(y_{t+1}, y_{t+2}, ..., y_{t+k} | y_{t-tau}, ..., y_{t-1})"""
    net.eval() # 设置为评估模式
    pred_multi = torch.zeros(num_sample)
    pred_multi[:n_train] = x[:n_train] # 前600个时间步的值是已知的

    for i in range(n_train, num_sample):
        # 每次都用前tau个时间步的值来预测下一个时间步的值，即自回归预测
        X = pred_multi[i-tau:i].view(1, -1)
        y_hat = net(torch.tensor(X, dtype=torch.float32))
        pred_multi[i] = y_hat.item()

    plt.plot(time, x, label='true')
    plt.plot(time, pred_multi, label='pred')
    plt.legend()
    plt.show()

    # ============ 多步预测 ===========
    """ 即将每个时间步的预测结果作为下一个时间步的输入 """
    """ 预测多个时间步的值 """
    """ 这里我们预测1, 4, 16, 32, 64个时间步的值 """
    """ 即 P(y_{t+1}, y_{t+4}, y_{t+16}, y_{t+32}, y_{t-64} | y_{t-tau}, ..., y_{t-1}) """

    # 我们在每步中直接把后续64个都预测出来，随后在根据需要进行切片，以便评估
    steps = [1, 4, 16, 32, 64]
    max_steps = max(steps) # 最大步数
    net.eval() # 设置为评估模式

    feature = torch.zeros((num_sample - tau - max_steps + 1, tau + max_steps)) # 特征矩阵，每行是一个时间步的输入，每列是前tau个时间步的值

    # 填充观测数据
    for i in range(tau):
        feature[:, i] = x[i:num_sample - tau - max_steps + 1 + i]
    
    # 填充64个预测值
    for i in range(tau, tau + max_steps):
        feature[:, i] = net(feature[:, i-tau:i]).squeeze()

    # 根据我们的需要的steps值进行裁剪
    plt.figure(figsize=(6, 3))
    for i in steps:
        plt.plot(time[tau + i - 1: num_sample - max_steps + i], 
                 feature[:, (tau + i - 1)].detach().numpy(), label=f'{i}-step preds')

    plt.xlabel('time')
    plt.ylabel('x')
    plt.legend()
    plt.show()