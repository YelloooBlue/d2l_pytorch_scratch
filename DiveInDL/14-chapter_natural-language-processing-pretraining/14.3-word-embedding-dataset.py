import math
import os
import random
import collections
import torch
import matplotlib.pyplot as plt
import numpy as np

# ============================ 读取PTB数据集 ===========================

def read_ptb():
    """将PTB数据集加载到文本行的列表中"""
    data_dir ="../data/ptb"  # PTB数据集的目录
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

# ============================ 词表构建 ===========================

# 来自8.2
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

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
    
# 来自8.2
def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# ============================ 下采样高频词 ===========================

def subsample(sentences, vocab):
    """下采样高频词"""
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

# ============================ 中心词和上下文词对 ===========================

def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中心词-上下文词”对，每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# ============================= 负采样 ===========================

class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1)) # 词表中所有词的索引
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0 # 目前已使用的缓存个数

    def draw(self):

        # 检查患缓存是否用完
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000) # 从词表中随机采样10000个词
            self.i = 0
        
        # 作废当前缓存
        self.i += 1
        return self.candidates[self.i - 1]
    
def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记）
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    
    """
        counter 为词表中每个词的频率，词频越高，采样权重越大。
        采样权重的指数0.75是一个经验值，目的是平衡高频词和低频词的采样概率。
        此时sampling_weights为一个列表，包含了词表中每个词的采样权重。
    """

    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# ============================ 绘图 ===========================

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    plt.figure(figsize=(6, 4))
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)

# ============================ 辅助函数 ===========================

def batchify(data):
    """返回带有负采样的跳元模型的小批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data) # 最长样本
    """
        data是一个列表，包含了中心词、上下文词和负采样噪声词的元组。
        c 是上下文词的列表，n是负采样噪声词的列表。
        形如（
            (中心词, [上下文词1, 上下文词2, ...], [噪声词1, 噪声词2, ...]),
            (中心词, [上下文词1, 上下文词2, ...], [噪声词1, 噪声词2, ...]),
            ...
        ）
    """

    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center] # 中心词

        # 将上下文词和负采样噪声词填充到相同长度
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        
        # 生成掩码，1表示有效词，0表示填充词
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]

        # 标签，1表示上下文词，0表示负采样噪声词
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))

# ============================ 封装数据集 ===========================

class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """下载PTB数据集，然后将其加载到内存中"""
    num_workers = 1
    sentences = read_ptb()                  # 读取到行
    vocab = Vocab(sentences, min_freq=10)   # 构建词表
    subsampled, counter = subsample(sentences, vocab) # 下采样高频词
    corpus = [vocab[line] for line in subsampled]     # 将词表映射到索引
    
    # 获取中心词和上下文词对（正样本）
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    
    # 获取负采样的噪声词（负样本）
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    
    print(f'# “中心词”的数量: {len(all_centers)}')
    print(f'# “上下文词”的数量: {sum([len(contexts) for contexts in all_contexts])}')
    print(f'# “噪声词”的数量: {sum([len(negatives) for negatives in all_negatives])}')

    """
        此时all_centers是中心词的列表，与vocab的索引对应，
        all_contexts是上下文词的列表，数量根据all_centers的数量结合max_window_size随机生成
        all_negatives是噪声词的列表，数量为all_contexts * num_noise_words
    """

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab


if __name__ == "__main__":
    
    # 测试读取PTB数据集
    sentences = read_ptb()
    print(f'# sentences数: {len(sentences)}')

    # 测试词表
    vocab = Vocab(sentences, min_freq=10)
    print(f'vocab size: {len(vocab)}')

    # 测试下采样 =========================================================
    subsampled, counter = subsample(sentences, vocab)
    """
        注意下采样不是负采样，而是对高频词进行下采样，减少它们的出现频率。
        数据集中的每个词都有概率被下采样掉，其中高频词被下采样的概率更高。
    """

    # 采用直方图对比下采样前后（画在同一张图上）
    show_list_len_pair_hist(
        ['subsampled', 'original'],
        'length of sentence',
        'number of sentences',
        subsampled, sentences
    )
    # plt.show()

    # "the" 的对比
    print(f'原始数据集中"the"的频率: {sum(l.count("the") for l in sentences)}')
    print(f'下采样后数据集中"the"的频率: {sum(l.count("the") for l in subsampled)}')
    # "join" 的对比
    print(f'原始数据集中"join"的频率: {sum(l.count("join") for l in sentences)}')
    print(f'下采样后数据集中"join"的频率: {sum(l.count("join") for l in subsampled)}')

    corpus = [vocab[line] for line in subsampled]
    corpus[:3]

    # 测试中心词和上下文词对 ===============================================
    tiny_dataset = [list(range(7)), list(range(7, 10))]
    print('数据集', tiny_dataset)
    for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
        print('中心词', center, '的上下文词是', context)

    all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    print(f'# “中心词-上下文词对”的数量: {sum([len(contexts) for contexts in all_contexts])}')

    # 测试负采样 ==========================================
    generator = RandomGenerator([2, 3, 4])
    print([generator.draw() for _ in range(10)])

    all_negatives = get_negatives(all_contexts, vocab, counter, 5)

    # 测试小批量
    x_1 = (1, [2, 2], [3, 3, 3, 3]) # 中心词1，上下文词2个，负采样噪声词4个
    x_2 = (1, [2, 2, 2], [3, 3]) 
    batch = batchify((x_1, x_2))

    names = ['centers', 'contexts_negatives', 'masks', 'labels']
    for name, data in zip(names, batch):
        print(name, '=', data)

    data_iter, vocab = load_data_ptb(512, 5, 5)
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
        break