import torch
import torch.nn.functional as F # 用于one-hot等
from torch import nn
from matplotlib import pyplot as plt
import collections
import random
import re

# ===================================== 文本处理 =====================================

def read_time_machine():
    """读取《时间机器》文本数据到列表"""
    with open('../data/timemachine.txt', 'r') as f:
        lines = f.readlines()

    # 忽略标点符号，大写，空格等
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word': # 按空格拆分单词
        return [line.split() for line in lines]
    elif token == 'char': # 按字符拆分
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

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

# ===================================== 读取时光机器数据集 =====================================

def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

# ===================================== 数据迭代器 以及 小批量生成 =====================================

def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
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
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
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

class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
    
# ===================================== 处理时光机器数据集 =====================================

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

# ===================================== 梯度裁剪 =====================================
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


################# 以上部分均为书前面章节给出的代码，仅添加部分注释 #################
################# 以下部分加入了个人的注释和修改，主要是为了增强代码的可读性和理解性 #################



# 定义完整RNN网络
class RNNModel(nn.Module):
    """RNN模型"""
    def __init__(self, vocab_size, num_hidden, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=num_hidden,
                          num_layers=num_layers)
        self.dense = nn.Linear(num_hidden, vocab_size)

    def forward(self, X, state):
        X = F.one_hot(X.T.long(), num_classes=len(vocab)).float()
        # X的形状为(num_steps, batch_size, vocab_size)
        Y, state_new = self.rnn(X, state)
        # Y的形状为(num_steps, batch_size, num_hidden)
        Y = self.dense(Y.reshape((-1, Y.shape[-1])))
        # Y的形状为(num_steps * batch_size, vocab_size)
        return Y, state_new

# 超参数
batch_size = 32  # 小批量大小
num_steps = 35  # 每个小批量的时间步数
num_hiddens = 512  # 隐藏层单元数
num_epochs = 500  # 训练轮数
lr = 1


if __name__ == '__main__':

    # =========== 初始化设备 ===========
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

    # 加载数据
    data_iter, vocab = load_data_time_machine(batch_size, num_steps) # 训出来一般 loss=0.3, perplexity=1.4
    # data_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True) # 训出来一般 loss=0.3, perplexity=1.4

    vocab_size = len(vocab)
    print(f'词表大小: {vocab_size}')

    # 定义RNN层，这里只包含循环部分
    rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
    print(f'RNN层数: {rnn_layer.num_layers}')
    
    # 测试
    state = torch.zeros((1, batch_size, num_hiddens))
    print(f'状态形状: {state.shape}')
    X = torch.rand(size=(num_steps, batch_size, vocab_size))
    print(f'输入形状: {X.shape}')
    Y, state_new = rnn_layer(X, state)
    print(f'输出形状: {Y.shape}, 新状态形状: {state_new.shape}')


    # 实例化RNN模型
    net = RNNModel(len(vocab), num_hiddens)
    net = net.to(device)

    # ========== 测试模型预测（在没有学习之前进行预测） ===========

    with torch.no_grad(): # 虽然不会影响梯度计算，但可以减少内存使用

        state = torch.zeros((1, 1, num_hiddens), device=device) # 默认1层RNN，测试用batch_size为1

        prefix = 'time traveller '
        num_predict = 10
        outputs = [vocab[prefix[0]]]  # 将第一个字符的索引添加到输出列表中，最开始为 [3]，即't' 的索引
    
        # 预热
        for y in prefix[1:]:
            X = torch.tensor([[outputs[-1]]], device=device)
            _, state = net(X, state) 

            # 只更新state，outputs仍用观测值
            outputs.append(vocab[y])

        print(f'预热后的输出索引: {outputs}')  # 打印预热后的输出索引
        print(f'预热后的输出字符: {vocab.to_tokens(outputs)}')  # 打印预热后的输出字符

        # 预测
        for _ in range(num_predict):
            X = torch.tensor([[outputs[-1]]], device=device)  # 使用最后一个输出作为下一个输入
            Y, state = net(X, state)  # 前向传播
            outputs.append(Y.argmax(dim=1).item())

    print(''.join(vocab.to_tokens(outputs)))  # 将索引转换为字符并打印


    # ========== 训练模型 ===========
    loss = nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器

    record_perplexity = []  # 用于记录每个epoch的困惑度

    for epoch in range(num_epochs):
        state = None
        l_sum, n = 0.0, 0

        for X, Y in data_iter:
            X, Y = X.to(device), Y.to(device)
            y = Y.T.reshape(-1) # 将标签展平为一维，与我们在forward函数中的处理对齐

            if state is None or state[0].shape[1] != X.shape[0]:  # 如果状态为None或batch_size不匹配
                state = torch.zeros((1, X.shape[0], num_hiddens), device=device)
            else:
                state = state.detach()

            # 前向传播
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()

            # 反向传播
            optimizer.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()

            l_sum += l.item() * y.numel()  # 累加损失
            n += y.numel()
        
        perplexity = torch.exp(torch.tensor(l_sum / n))
        record_perplexity.append(perplexity.item())

        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, perplexity {perplexity:.1f}, '
                  f'loss {l_sum / n:.4f}')
            
    # 绘制困惑度曲线
    plt.plot(range(1, num_epochs + 1), record_perplexity, label='Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training Perplexity')    
    plt.legend()
    plt.show()

    # ========== 测试模型预测（在学习之后进行预测） ===========
    with torch.no_grad():  # 虽然不会影响梯度计算，但可以减少内存使用
        state = torch.zeros((1, 1, num_hiddens), device=device)  # 默认1层RNN，测试用batch_size为1

        prefix = 'time traveller '
        num_predict = 50
        outputs = [vocab[prefix[0]]]  # 将第一个字符的索引添加到输出列表中，最开始为 [3]，即't' 的索引

        # 预热
        for y in prefix[1:]:
            X = torch.tensor([[outputs[-1]]], device=device)
            _, state = net(X, state)

            # 只更新state，outputs仍用观测值
            outputs.append(vocab[y])

        print(f'预热后的输出索引: {outputs}')  # 打印预热后的输出索引
        print(f'预热后的输出字符: {vocab.to_tokens(outputs)}')  # 打印预热后的输出字符

        # 预测
        for _ in range(num_predict):
            X = torch.tensor([[outputs[-1]]], device=device)  # 使用最后一个输出作为下一个输入
            Y, state = net(X, state)  # 前向传播
            outputs.append(Y.argmax(dim=1).item())

    print(''.join(vocab.to_tokens(outputs)))  # 将索引转换为字符并打印