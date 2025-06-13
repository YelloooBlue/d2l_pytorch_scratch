import torch
import re
import collections
from matplotlib import pyplot as plt
import random

def read_time_machine():
    """读取《时间机器》文本数据到列表"""
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()

    # 忽略标点符号，大写，空格等
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):  #@save
    """将文本行拆分为单词或字符词元"""
    if token == 'word': # 按空格拆分单词
        return [line.split() for line in lines]
    elif token == 'char': # 按字符拆分
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], # x[0]是词元，x[1]是频率
                                   reverse=True)
        
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens # reserved_tokens是保留的词元列表

        # 生成从 词元 到 索引 的映射
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        

        for token, freq in self._token_freqs:

            # 如果词元的频率小于最小频率，则跳过
            if freq < min_freq: 
                break

            # 如果词元不在词表中，则添加到词表
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1 # 同时创建从 词元 到 索引 的映射

    def __len__(self):
        return len(self.idx_to_token)

    # 将词元转换为索引，如果是单个词元，则返回单个索引，否则返回索引列表
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    # 将索引转换为词元，如果是单个索引，则返回单个词元，否则返回词元列表
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def load_corpus_time_machine(max_tokens=-1):  #@save
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') # 按字符拆分
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == '__main__':

    # ================== 一元语法模型 ==================

    tokens = tokenize(read_time_machine())

    # 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
    # corpus = [token for line in tokens for token in line]

    corpus = [] # 这里使用corpus其实并不太合适，因为corpus内通常存储的是词元的索引（数字），而不是词元本身
    for line in tokens:
        for token in line:
            corpus.append(token) 

    vocab = Vocab(corpus)
    print('词表大小:', len(vocab))
    print('词频前10的token:', vocab.token_freqs[:10])
    
    # 画出词元频率分布图
    freqs = [freq for _, freq in vocab.token_freqs]
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(freqs)), freqs, marker='o')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    # 可以看出跟推荐系统很像，有长尾分布

    # =================== 二元语法模型 ==================
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])] # 二元语法模型，pair = (前一个词元, 后一个词元)
    bigram_vocab = Vocab(bigram_tokens)
    print('二元语法模型词表大小:', len(bigram_vocab))
    print('词频前10的二元token:', bigram_vocab.token_freqs[:10])

    # =================== 三元语法模型 ==================
    trigram_tokens = [pair for pair in zip(corpus[:-2], corpus[1:-1], corpus[2:])] # 三元语法模型，pair = (前一个词元, 当前词元, 后一个词元)
    trigram_vocab = Vocab(trigram_tokens)
    print('三元语法模型词表大小:', len(trigram_vocab))
    print('词频前10的三元token:', trigram_vocab.token_freqs[:10])

    # 将三个画在一起
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(freqs)), freqs, marker='o', label='unigram')
    plt.plot(range(len(bigram_vocab.token_freqs)), [freq for _, freq in bigram_vocab.token_freqs], marker='o', label='bigram')
    plt.plot(range(len(trigram_vocab.token_freqs)), [freq for _, freq in trigram_vocab.token_freqs], marker='o', label='trigram')
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    