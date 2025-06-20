import torch
from torch import nn
import matplotlib.pyplot as plt

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                            sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


def show_results(x_train, y_train, x_test, y_truth, y_pred):
    """显示训练样本、测试样本和预测结果"""
    plt.scatter(x_train, y_train, label='Train samples', color='blue')
    plt.plot(x_test, y_truth, label='Test samples', color='orange')
    plt.plot(x_test, y_pred, label='Predictions', color='green')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()



class NWKernelRegression(nn.Module):
    """Nadaraya-Watson核回归模型"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # 输入的查询形状
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        # 1. 将一个查询按行重复“键－值”对个数次
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))


        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w)**2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)




if __name__ == '__main__':
    

    # 测试画图
    attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
    # show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')

    # 生成测试数据
    n_train = 50  # 训练样本数
    x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本（x轴）
    print(f'训练样本: {x_train}')

    def f(x):
        return 2 * torch.sin(x) + x**0.8

    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    x_test = torch.arange(0, 5, 0.1)  # 用于测试的输入/样本（x轴）
    y_truth = f(x_test)  # 测试样本的真实输出
    print(f'训练样本数: {n_train}, 测试样本数: {len(x_test)}')

    # 画出训练样本和测试样本
    show_results(x_train, y_train, x_test, y_truth, [None] * len(x_test))

    # 1. 简单的平均汇聚
    y_pred = torch.mean(y_train) * torch.ones_like(x_test)
    show_results(x_train, y_train, x_test, y_truth, y_pred)

    # 2. Nadaraya-Watson核回归
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))  # 重复训练样本，模拟「查询」, shape: (n_test,n_train)
    """
        此时x_test是一个从0到5的等间距点（0.1为步长），形如[0.0, 0.1, 0.2, ..., 4.9]。
        假设 经过 repeat_interleave（2）后，形如 [0.0, 0.0, 0.1, 0.1, ..., 4.9, 4.9]。
        只不过在这里我们将其reshape成了(n_test, n_train)的形状。
        X_repeat的每一列都是相同 x_test。
        相当于吧x_test转置成列后，沿着列方向重复n_train次。
        [0.0, 0.0, 0.0, ..., 0.0]
        [0.1, 0.1, 0.1, ..., 0.1]
        [0.2, 0.2, 0.2, ..., 0.2]
        ...
        [4.9, 4.9, 4.9, ..., 4.9]
    """
    print(f'X_repeat shape: {X_repeat.shape}, x_test shape: {x_test.shape}')
    attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)  # 使用「高斯核」计算注意力权重


    """
        此时x_train是一个随机生成的50个点，形如[0.0, 0.11, 0.19, ..., 4.91]。
        触发广播机制后，x_train被沿着行方向重复n_test次，形成(n_test, n_train)的形状。
        [0.0, 0.11, 0.19, ..., 4.91]
        [0.0, 0.11, 0.19, ..., 4.91]
        [0.0, 0.11, 0.19, ..., 4.91]
        ...
        [0.0, 0.11, 0.19, ..., 4.91]
        这样每一行都是相同的x_train。
        随后 -（X_repeat - x_train)**2 以后会得到类似
        [0.0, -0.01, -0.04, ..., -12.08]
        [-0.01, 0.0, -0.03, ..., -12.07]
        [-0.04, -0.03, 0.0, ..., -12.04]
        ...
        这样的矩阵。
        我们可以看到，矩阵的对角线上的元素都是0（相对来说是最大的，因为其他元素都是负数），
        经过softmax后，对角线上的元素会趋近1，其余元素趋近于0。
    """

    print(f'attention_weights shape: {attention_weights.shape}, y_train shape: {y_train.shape}')
    y_pred = attention_weights @ y_train
    show_results(x_train, y_train, x_test, y_truth, y_pred)

    """
        在这里中我们根据“相似度”来计算注意力权重，越接近的点权重越大。
        我们知道x_train是随机生成的，坐落于[0, 5]之间的50个点。
        当我们查询时，x_test是一个从0到5的等间距点（0.1为步长），我们希望得到的“值”是距离当前「查询」点最近的「键」点的「值」。
    """

    # 可视化注意力权重
    show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Keys', ylabel='Queries') # (1, 1, n_test, n_train)



    # X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入（随机产生的x轴坐标）
    X_tile = x_train.repeat((n_train, 1))
    print(f'X_tile: {X_tile}')
    """
        X_tile的每一行都是相同的x_train。
        形如
        [0.0, 0.11, 0.19, ..., 4.91]
        [0.0, 0.11, 0.19, ..., 4.91]
        [0.0, 0.11, 0.19, ..., 4.91]
        ...
        [0.0, 0.11, 0.19, ..., 4.91]
    """
    # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出（加入噪声的y轴坐标）
    Y_tile = y_train.repeat((n_train, 1))
    print(f'Y_tile: {Y_tile}')
    """
        Y_tile的每一行都是相同的y_train。
        每一行就像sin(x) + noise的结果。
    """
    
    mask = (1 - torch.eye(n_train)).type(torch.bool)  # 创建一个对角线为0，其余为1的矩阵。屏蔽自身的影响

    keys = X_tile[mask].reshape((n_train, -1))  # keys的形状:('n_train'，'n_train'-1)
    values = Y_tile[mask].reshape((n_train, -1))  # values的形状:('n_train'，'n_train'-1)
    # keys的每一行都是相同的x_train，除了对角线上的元素（自身）被屏蔽掉了。
    """
        keys的形状为(n_train, n_train-1)，每一行都是相同的x_train，除了对角线上的元素（自身）被屏蔽掉了。
        就像对角线上的元素被删除后，右边往左边补位（俄罗斯方块），所以列会少一个。
        形如
        [0.11, 0.19, ..., 4.91]
        [0.0, 0.19, ..., 4.91]
        [0.0, 0.11, ..., 4.91]
        ...
        [0.0, 0.11, ..., 4.80]
        values同理
    """

    net = NWKernelRegression()
    net.train()  # 设置模型为训练模式
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)

    record_loss = []  # 用于记录损失

    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train) 
        """
            送入模型时，虽然keys也是由x_train生成的，但它对角线上的元素已经被屏蔽掉了。
            所以，queries 会 寻找 keys 中与自身最相似的点（除了自身），并赋予更高的权重。
        """
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        record_loss.append(float(l.sum()))

    # 画出训练损失
    plt.plot(range(1, 6), record_loss, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(range(1, 6))
    plt.grid()
    plt.show()

    # 预测
    net.eval()  # 设置模型为评估模式
    keys = x_train.repeat((len(x_test), 1))  # 重复训练样本，模拟「键」
    values = y_train.repeat((len(x_test), 1))  # 重复训练
    y_pred = net(x_test, keys, values).unsqueeze(1).detach()  # 预测结果
    print(f'y_pred shape: {y_pred.shape}, y_truth shape: {y_truth.shape}')
    show_results(x_train, y_train, x_test, y_truth, y_pred)

    # 可视化注意力权重
    show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0), xlabel='Keys', ylabel='Queries',
                  titles=['Attention Weights'], figsize=(3, 3), cmap='Blues')
    
    """
        在这个例子中，我们有两种x轴的坐标取值方式：
        1. x_train：随机生成的50个点，作为训练样本
        2. x_test：从0到5的等间距点（0.1为步长），作为测试样本
        在本例中，我们的“注意力”是基于 key 和 query 在x轴上的距离来计算的。
        也就是说我们在培养模型关注与当前查询点（query）最接近的键点（key）的值（y轴坐标）。
        这样不管我们怎么对x_train进行采样，模型都能根据当前查询点（x_test）找到最接近的键点（x_train）并返回对应的值。
    """
