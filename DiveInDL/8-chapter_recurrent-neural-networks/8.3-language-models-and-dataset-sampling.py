import torch
import re
import collections
from matplotlib import pyplot as plt
import random

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为最后一个词元没有下一个词元作为标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch] # 其实就是X向量右移一个位置
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


if __name__ == '__main__':

    my_seq = list(range(35))

    """
    对于语言建模，目标是基于到目前为止我们看到的词元来预测下一个词元， 因此标签是「移位了一个词元」的原始序列
    对于以下操作
        num_steps 即小批量子序列的长度，即窗口大小，也就是模型的「上下文长度」
        batch_size 即小批量的大小，即每个小批量包含多少个子序列
    """

    print('# 随机抽样生成小批量子序列:')
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)
        print()

    print('=' * 20)
    print('# 顺序分区生成小批量子序列:')
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)
    

"""
如何选用seq_data_iter_random和seq_data_iter_sequential？
- seq_data_iter_random：适用于需要随机抽样的场景，能打破序列的顺序性，适合训练模型时需要多样性和随机性的情况。
- seq_data_iter_sequential：适用于需要保持序列顺序的场景，适合处理时间序列数据或需要保留上下文信息的情况。

需要长距离依赖：
    如语言模型、机器翻译、阅读理解等长文本场景，优先选择「顺序分区」。
    例：预测文章中的下一个词，需要前文完整信息。
局部模式为主：
    如情感分析、文本分类、社交媒体评论等短文本场景，可选择「随机抽样」。
    例：判断评论的情感倾向，通常依赖局部关键词。

"""