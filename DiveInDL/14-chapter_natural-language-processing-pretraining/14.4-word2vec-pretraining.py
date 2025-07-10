import math
import torch
from torch import nn
import os
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

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

# ============================= 模型定义 ===========================

# 前向传播
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred

    """
        embed_v 的更新：
            每次更新时，embed_v(center)会向所有正样本上下文词的平均向量靠近，同时远离负样本向量。
            这使得 embed_v 捕捉到词的通用上下文分布（即语义）。
            「学习如何通过自身语义来预测上下文词」。
        embed_u 的更新：
            每个上下文词向量uo会根据不同的中心词wc被更新。
            例如，
            当中心词是 “king” 时，embed_u(“crown”)会靠近embed_v(“king”)；
            当中心词是 “prince” 时，embed_u(“crown”)会靠近embed_v(“prince”)。
            这使得 embed_u 捕捉到词的上下文依赖角色。
            「学习如何让自己与不同中心词的语义相匹配」。
    """

# ============================= 辅助函数 ===========================

class SigmoidBCELoss(nn.Module):
    # 带掩码的二元交叉熵损失
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

loss = SigmoidBCELoss()

# ============================= 计算相似词 ===========================

def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # 计算余弦相似性。增加1e-9以获得数值稳定性
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # 删除输入词
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

if __name__ == "__main__":

    # 检查CUDA和Metal是否可用
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Metal is available: {torch.backends.mps.is_available()}")
    print(f"Metal is built: {torch.backends.mps.is_built()}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"# Using device: {device}")

    # ============================= 加载数据集 ===========================

    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data_ptb(batch_size, max_window_size, num_noise_words)

    # 测试损失函数
    pred = torch.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
    label = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    l = loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
    print(f'loss: {l}')  # 输出损失值

    # 初始化模型参数
    embed_size = 100
    net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                    embedding_dim=embed_size),
                        nn.Embedding(num_embeddings=len(vocab),
                                    embedding_dim=embed_size))
    
    """
        net[0]是中心词的嵌入层，net[1]是上下文词和负采样噪声词的嵌入层。
        这两个嵌入层并没有直接连接在一起，而是通过skip_gram函数进行前向传播。
    """
    
    # ============================= 训练模型 ===========================

    lr, num_epochs = 0.002, 5

    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    record_loss = []  # 用于记录损失值

    # 记录总损失和样本数量用于计算平均损失
    total_loss = 0.0
    total_samples = 0

    for epoch in range(num_epochs):
        num_batches = len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            
            total_loss += l.sum().item()
            total_samples += l.numel()
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                avg_loss = total_loss / total_samples
                record_loss.append(avg_loss)
                print(f'Epoch {epoch+1}, Batch {i+1}/{num_batches}, Average Loss: {avg_loss:.4f}')
    
    print(f'Final average loss: {total_loss / total_samples:.3f}')

    # 绘制损失曲线
    plt.plot(record_loss, label='Average Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()

    # ============================= 计算相似词 ===========================

    get_similar_tokens('chip', 3, net[0])