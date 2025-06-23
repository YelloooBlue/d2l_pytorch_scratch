import math
import torch
from torch import nn

# 来自9.7
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # 此时X的形状为(batch_size,num_steps)
    # valid_len的形状为(batch_size,)，表示每个序列的有效长度
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    # torch.arange(maxlen)生成一个从0到maxlen-1的张量
    # [None, :]将其转换为行向量(1, maxlen)
    # valid_len[:, None]将valid_len转换为列向量(batch_size, 1)

    # 当(1, maxlen)的行向量与(batch_size, 1)的列向量进行比较时，
    # 会广播成(batch_size, maxlen)的布尔矩阵

    # mask的形状为(batch_size, maxlen)，表示每个位置是否有效
    X[~mask] = value
    return X

# 来自10.3
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

# 来自10.3
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

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)

    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # 将最后一个维度num_hiddens拆分为num_heads个头
    # 此时X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)

    X = X.permute(0, 2, 1, 3)
    # 此时X的形状:(batch_size，num_heads，查询或者“键－值”对的个数, num_hiddens/num_heads)

    return X.reshape(-1, X.shape[2], X.shape[3])
    # 最终输出的形状:(batch_size*num_heads, 查询或者“键－值”对的个数, num_hiddens/num_heads)
    # 这样是为了后续能够「多头并行计算」。

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        """
            为了方便计算，我们将Q（查询）、K（键）和V（值）都变换为num_hiddens维度。
            这样每个头输出的维度为num_hiddens/num_heads。
        """

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:(batch_size，查询或者“键－值”对的个数，qkv各自的维度)
        # valid_lens　的形状: (batch_size，)或(batch_size，查询的个数)

        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # 经过transpose_qkv后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)

        # 针对valid_lens进行「多头/并行」计算的处理
        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        
        output = self.attention(queries, keys, values, valid_lens)
        # output的形状:(batch_size*num_heads，查询的个数，num_hiddens/num_heads)

        output_concat = transpose_output(output, self.num_heads)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        
        return self.W_o(output_concat)
        # 返回的形状:(batch_size，查询的个数，num_hiddens)

if __name__ == "__main__":

    # 测试MultiHeadAttention
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, num_heads, 0.5)
    print(attention.eval())

    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens =  6, torch.tensor([3, 2])
    Q = torch.ones((batch_size, num_queries, num_hiddens))
    K = torch.ones((batch_size, num_kvpairs, num_hiddens))
    V = K
    print(f'Q: {Q.shape}, K: {K.shape}, V: {V.shape}, valid_lens: {valid_lens.shape}')
    
    output = attention(Q, K, V, valid_lens)
    print(f'output: {output.shape}')