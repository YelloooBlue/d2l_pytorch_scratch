# 8.2-text-preprocessing

import torch
import re
import collections

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


if __name__ == "__main__":

    # ============= 模块测试 =============
    
    # 读取《时间机器》文本数据
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])

    # 将文本行拆分为单词或字符词元
    print("# 按行拆分的词元预览 tokens:")
    tokens = tokenize(lines)
    for i in range(11):
        print(tokens[i])

    # 构建词表
    print("# 词表预览 vocab:")
    vocab = Vocab(tokens)
    print(list(vocab.token_to_idx.items())[:10])

    # 测试词表
    print(f'# 根据文本行tokens[0]和tokens[10]获取索引:')
    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])

    print('=' * 20)

    # ============= 正式读取数据 =============

    # corpus, vocab = load_corpus_time_machine()
    # print(f'# 按char分词，corpus长度: {len(corpus)}')
    # print(f'# 语料库 corpus 前10个预览: {corpus[:10]}')

    # print(f'# 词表 vocab 大小: {len(vocab)}')
    # print(f'# 词表 vocab 前10个预览: {vocab[:10]}')


    # 拆分
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') # 按字符拆分
    print(f'# 按char分词，tokens长度: {len(tokens)}')
    print(f'# tokens 前10行预览: {tokens[:10]}')

    vocab = Vocab(tokens)
    print(f'# 词表 vocab 大小: {len(vocab)}') # 这里只有28是因为，英文字符集只有26个字母，加上空格和<unk>
    print(f'# 词表 vocab 前10个预览: {vocab.idx_to_token[:10]}')

    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    print(f'# 语料库 corpus 长度: {len(corpus)}')
    print(f'# 语料库 corpus 前10个预览: {corpus[:10]}')
    

    
"""
各个概念的例子
    - token : 词元，文本中的基本单位，可以是单词、字符等，例如 "hello"、"world"、"h"、"e"、"l"、"l"、"o"
    - tokens : 词元列表，在本例中是一个二维列表，每个子列表代表一行文本的词元，例如 [["h", "e", "l", "l", "o"], ["w", "o", "r", "l", "d"]]
    - vocab : 词表，包含所有唯一的词元及其索引映射，例如 {"<unk>": 0, "h": 1, "e": 2, "l": 3, "o": 4, "w": 5, "r": 6, "d": 7}
    - corpus : 语料库，包含所有词元的索引列表，例如 [1, 2, 3, 3, 4, 5, 6, 7]，其中每个数字对应于词表中的一个词元
"""