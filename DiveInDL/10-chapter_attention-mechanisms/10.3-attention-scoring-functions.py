import math
import torch
from torch import nn
import matplotlib.pyplot as plt

# 来自10.1
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

# 来自9.7
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # 此时X的形状为(batch_size,num_steps)
    # valid_len的形状为(batch_size,)，表示每个序列的有效长度
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # torch.arange(maxlen)生成一个从0到maxlen-1的张量
    # [None, :]将其转换为行向量(1, maxlen)
    # valid_len[:, None]将valid_len转换为列向量(batch_size, 1)

    # 当(1, maxlen)的行向量与(batch_size, 1)的列向量进行比较时，
    # 会广播成(batch_size, maxlen)的布尔矩阵

    # mask的形状为(batch_size, maxlen)，表示每个位置是否有效
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    # X:形状为(batch_size, num_queries, num_kvs)，
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """
    加性注意力
        简单来说，当Query和Key的维度不同时，我们通过一个线性变换将它们映射到相同的维度，
        然后将它们相加，经过tanh激活函数后，再通过一个线性变换得到注意力分数。
    
    """
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries的形状：(batch_size，查询的个数，dim_query)
        # keys的形状：(batch_size，“键－值”对的个数，dim_key)

        queries, keys = self.W_q(queries), self.W_k(keys)
        # queries的形状：(batch_size，查询的个数，num_hidden)
        # keys的形状：(batch_size，“键－值”对的个数，num_hidden)
        
        
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # features的形状：(batch_size，查询的个数，“键－值”对的个数，num_hidden)
        features = torch.tanh(features)
        
        
        # w_v(features)的形状：(batch_size，查询的个数，“键－值”对的个数，1)
        scores = self.w_v(features).squeeze(-1)

        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        self.attention_weights = masked_softmax(scores, valid_lens)

        # attention_weights的形状：(batch_size，查询的个数，“键－值”对的个数)
        # values的形状：(batch_size，“键－值”对的个数，dim_value)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # 执行批量矩阵乘法，得到的形状为(batch_size, 查询的个数, dim_value)


class DotProductAttention(nn.Module):
    """
    缩放点积注意力
        简单来说，当Query和Key的维度相同时，我们直接计算它们的点积，以节省计算资源。
        假设Query和Key的都满足 均值为0，方差为1 的正态分布
        那么它们的点积的均值为0，方差为d（d为Query和Key的维度）。
        因此，为了避免点积过大，我们将其除以sqrt(d)，使得均值为0，方差为1。
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，dim_value)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度（即将“键－值”对的个数和d交换）
        # 变为 (batch_size, d, “键－值”对的个数)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        # scores的形状：(batch_size，查询的个数，“键－值”对的个数)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
        # 执行批量矩阵乘法，得到的形状为(batch_size, 查询的个数, dim_value)


if __name__ == "__main__":

    # 测试masked_softmax函数
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

    # 测试AdditiveAttention
    queries = torch.normal(0, 1, (2, 1, 20)) # (batch_size, 查询个数, dim_query)
    keys = torch.ones((2, 10, 2))            # (batch_size, “键－值”对的个数, dim_key)

    # (1, 10, 4) = (1, “键－值”对的个数, dim_value)
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)  # (batch_size, “键－值”对的个数, dim_value) = (2, 10, 4)
    
    # values的小批量，两个值矩阵是相同的，但有效长度不同
    valid_lens = torch.tensor([2, 6])

    print(f'queries: {queries.shape}, keys: {keys.shape}, values: {values.shape}')

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))

    show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries',
                  titles=['Additive Attention Weights'])



    
    # 测试DotProductAttention
    queries = torch.normal(0, 1, (2, 1, 2)) # (batch_size, 查询个数, dim_query)
    # keys 和 values 沿用上面的
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))

    show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries',
                  titles=['Dot Product Attention Weights'])