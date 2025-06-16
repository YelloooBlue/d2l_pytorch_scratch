import torch
import torch.nn.functional as F # 用于one-hot等
from torch import nn
import random
import re
import collections
from matplotlib import pyplot as plt
import math

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


################# 以上部分均为书前面章节给出的代码，仅添加部分注释 #################
################# 以下部分加入了个人的注释和修改，主要是为了增强代码的可读性和理解性 #################


def get_params(vocab_size, num_hiddens, device):
    """初始化模型的参数"""
    num_inputs = num_outputs = vocab_size  # 语言模型的输入和输出都是词表大小
    
    # 输入门
    W_xi = torch.randn((num_inputs, num_hiddens), device=device) * 0.01
    W_hi = torch.randn((num_hiddens, num_hiddens), device=device) * 0.01
    b_i = torch.zeros(num_hiddens, device=device)

    # 遗忘门
    W_xf = torch.randn((num_inputs, num_hiddens), device=device) * 0.01
    W_hf = torch.randn((num_hiddens, num_hiddens), device=device) * 0.01
    b_f = torch.zeros(num_hiddens, device=device)

    # 输出门
    W_xo = torch.randn((num_inputs, num_hiddens), device=device) * 0.01
    W_ho = torch.randn((num_hiddens, num_hiddens), device=device) * 0.01
    b_o = torch.zeros(num_hiddens, device=device)

    # 记忆元
    W_xc = torch.randn((num_inputs, num_hiddens), device=device) * 0.01
    W_hc = torch.randn((num_hiddens, num_hiddens), device=device) * 0.01
    b_c = torch.zeros(num_hiddens, device=device)

    # 输出层参数
    W_hq = torch.randn((num_hiddens, num_outputs), device=device) * 0.01
    b_q = torch.zeros(num_outputs, device=device)

    # 将参数转换为可训练的张量
    params = [W_xi, W_hi, b_i,
              W_xf, W_hf, b_f,
              W_xo, W_ho, b_o,
              W_xc, W_hc, b_c,
              W_hq, b_q]
    
    for param in params:
        param.requires_grad_(True)
    
    return params


def lstm(inputs, state, params):
    """实现GRU的前向计算"""
    W_xi, W_hi, b_i = params[0:3] # 输入门
    W_xf, W_hf, b_f = params[3:6] # 遗忘门
    W_xo, W_ho, b_o = params[6:9] # 输出门
    W_xc, W_hc, b_c = params[9:12] # 记忆
    W_hq, b_q =  params[12:14]

    outputs = []
    (h, c) = state

    for X in inputs:

        # 计算输入门
        i = torch.sigmoid(X @ W_xi + h @ W_hi + b_i)
        # 计算遗忘门
        f = torch.sigmoid(X @ W_xf + h @ W_hf + b_f)
        # 计算输出门
        o = torch.sigmoid(X @ W_xo + h @ W_ho + b_o)


        # 计算候选记忆
        c_tilde = torch.tanh(X @ W_xc + h @ W_hc + b_c)

        # 遗忘过去记忆
        c = f * c + i * c_tilde

        # 更新隐藏状态
        h = o * torch.tanh(c)

        # 计算输出
        Y = h @ W_hq + b_q
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (h, c)


class LSTMModel:
    def __init__(self, vocab_size, num_hiddens): 
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)

    def __call__(self, X, state):
        # inputs的形状是(batch_size, num_steps)，
        # 需要将其转置为(num_steps, batch_size)，然后进行独热编码
        # 独热编码后，X的形状变为(num_steps, batch_size, vocab_size)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return lstm(X, state, self.params)
    
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

# 超参数
num_epochs = 500
num_hiddens = 256
lr = 1
batch_size = 32
num_steps = 35

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


    # =========== 初始化数据集 ===========
    train_iter, vocab = load_data_time_machine(batch_size, num_steps) # 训出来一般 loss=
    # train_iter, vocab = load_data_time_machine(batch_size, num_steps, use_random_iter=True) # 训出来一般 loss=

    vocab_size = len(vocab)
    print(f'词表大小: {vocab_size}')
    print(f'词表前10个词元: {vocab.idx_to_token[:10]}')  # 打印前10个词元

    # =========== 实例化模型 ===========
    net = LSTMModel(vocab_size, num_hiddens)

    # =========== 模型测试 ===========

    X = torch.arange(10).reshape((2, 5)).to(device)  # (batch_size, num_steps)

    # 初始化状态
    test_batch_size = X.shape[0]  # batch_size
    h = torch.zeros((test_batch_size, num_hiddens), device=device)
    c = torch.zeros((test_batch_size, num_hiddens), device=device)
    state = (h, c)

    # ！！！！！！！！！！！以上是LSTM的最大的不同，state里面新引入了c作为记忆元！！！！！！！！！！！！

    # 模拟前向传播
    print(" ===== 模拟前向传播 ===== ")
    print(f'输入形状: {X.shape}')  # (batch_size, num_steps)
    Y, state_new = net(X, state)  # 前向传播
    print(f'输出形状: {Y.shape}')  # (num_steps, batch_size, vocab_size)
    print(f'状态长度: {len(state_new)}')  # 状态是一个元组，长度为1
    print(f'新状态形状: {state_new[0].shape}')  # 新状态的形状是 (batch_size, num_hiddens)
    print(f'新记忆形状: {state_new[1].shape}')  # 新状态的形状是 (batch_size, num_hiddens)
    print(" ===== 模拟前向传播结束 ===== ")

    # ========== 测试模型预测（在没有学习之前进行预测） ===========

    with torch.no_grad(): # 虽然不会影响梯度计算，但可以减少内存使用

        h = torch.zeros((1, num_hiddens), device=device)
        c = torch.zeros((1, num_hiddens), device=device)
        state = (h, c)

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

    # =========== 训练模型 ===========
    loss = torch.nn.CrossEntropyLoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.params, lr=lr)  # 定义优化器
    print("开始训练模型...")

    record_perplexity = []  # 用于记录每个epoch的困惑度

    for epoch in range(num_epochs):
        
        state = None
        l_sum, n = 0.0, 0

        for X, Y in train_iter:
            # 将输入和标签移动到设备上
            X, Y = X.to(device), Y.to(device)
            y = Y.T.reshape(-1)

            if state is None:
                # 初始化状态
                h = torch.zeros((X.shape[0], num_hiddens), device=device)
                c = torch.zeros((X.shape[0], num_hiddens), device=device)
                state = (h, c)
            else:
                # state = state.detach()  # detach状态，避免梯度累积
                for s in state:
                    s = s.detach() # s.detach_() 也行，s = 逻辑更清晰点
                    

            # 前向传播
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()  # 平均损失
            
            # 反向传播
            optimizer.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
            
            # 记录损失和困惑度
            l_sum += l.item() * y.numel()
            n += y.numel()
        
        perplexity = math.exp(l_sum / n)  # 计算困惑度
        record_perplexity.append(perplexity)

        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch + 1}, loss {l_sum / n:.4f}')
            print(f'困惑度: {perplexity:.4f}')

    # 绘制困惑度曲线
    plt.plot(record_perplexity, label='Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training Perplexity')
    plt.legend()
    plt.show()

    print("模型训练完成！")

    # 测试模型预测
    with torch.no_grad():

        h = torch.zeros((1, num_hiddens), device=device)
        c = torch.zeros((1, num_hiddens), device=device)
        state = (h, c)

        prefix = 'time traveller '
        outputs = [vocab[prefix[0]]]  # 将第一个字符的索引添加到输出列表中
        num_predict = 50
        # 预热
        for y in prefix[1:]:
            X = torch.tensor([[outputs[-1]]], device=device)
            _, state = net(X, state)  # 前向传播
            outputs.append(vocab[y])  # 只更新outputs
        # 预测
        for _ in range(num_predict):
            X = torch.tensor([[outputs[-1]]], device=device)  # 使用最后一个输出作为下一个输入
            Y, state = net(X, state)  # 前向传播
            outputs.append(Y.argmax(dim=1).item())  # 取最大值的索引
        print(''.join(vocab.to_tokens(outputs)))  # 将索引转换为字符并打印
