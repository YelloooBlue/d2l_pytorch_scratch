import torch
from torch import nn
import collections
import matplotlib.pyplot as plt
from torch.utils import data
import math

# ===================================== 文本预处理和词表构建 =====================================

def read_data_nmt():
    with open('../data/fra-eng/fra.txt', 'r', encoding='utf-8') as f:
        return f.read()
    
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

class Vocab:  #@save
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

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
    
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)

    # array的形状为(batch_size, num_steps)
    # valid_len的形状为(batch_size,)，表示每个序列的有效
    return array, valid_len

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab



# ！！！！！！！！！！！ 以上部分为数据预处理和词表构建 ！！！！！！！！！！！
# ！！！！！！！！！！！ 没有做修改，直接复制之前的代码 ！！！！！！！！！！！

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


# ===================================== 注意力机制 ====================================


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

class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# 这里我们不用9.6的Encoder，而是用nn.Module自己实现
class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)

        # 为词表中的每个词元创建一个嵌入向量
        self.embedding = nn.Embedding(vocab_size, embed_size) 
        
        # 模型输入为 当前词元的嵌入向量
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # 此时X的形状为(batch_size,num_steps)，
        X = self.embedding(X)
        # 嵌入后，X的形状变为(batch_size,num_steps,embed_size)
        
        # 为了与RNN的输入形状匹配，交换X的前两个维度
        X = X.permute(1, 0, 2)

        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)

        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

# 这里我们不用9.6的Decoder，而是用nn.Module自己实现
class AttentionDecoder(nn.Module):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout) # key_size和query_size都为num_hiddens
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        outputs, hidden_state = enc_outputs
        # 此时outputs的形状为(num_steps, batch_size, num_hiddens)
        # 此时hidden_state的形状为(num_layers, batch_size, num_hiddens)
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
        # outputs的形状转换为(batch_size, num_steps, num_hiddens)

    def forward(self, X, state):
        # X: (batch_size,num_steps)
        enc_outputs, hidden_state, enc_valid_lens = state
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,num_hiddens)
        
        X = self.embedding(X).permute(1, 0, 2)
        # X_embedding的形状为(batch_size,num_steps,embed_size)
        # X的形状转换为(num_steps,batch_size,embed_size)

        # 遍历每个时间步
        outputs, self._attention_weights = [], []
        for x in X:
            # x的形状为(batch_size,embed_size)
            
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # query的形状为(batch_size,1,num_hiddens) 查询个数为1

            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens) # keys和values都为enc_outputs
            # context的形状为(batch_size,1,num_hiddens) = (batch_size, 查询的个数, dim_value)

            
            # 在特征维度上连结
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # x先变形为(batch_size,1,embed_size)
            # 然后在特征维度上连结，得到(batch_size,1,embed_size+num_hiddens)
            
            
            x = x.permute(1, 0, 2)
            # 将x变形为(1,batch_size,embed_size+num_hiddens)
            out, hidden_state = self.rnn(x, hidden_state)
            # out的形状为(1,batch_size,num_hiddens)

            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        
        
        # 此时outputs是一个列表，包含了每个时间步的输出
        # outputs的形状为(num_steps,batch_size,num_hiddens)
        # self._attention_weights是一个列表，包含了每个时间步的注意力权重

        outputs = self.dense(torch.cat(outputs, dim=0))
        # 全连接层变换后，outputs的形状为
        # (num_steps,batch_size,vocab_size)

        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]
        # 返回的outputs的形状为(batch_size,num_steps,vocab_size)
                                        

    @property
    def attention_weights(self):
        return self._attention_weights

# ==================================== 模型训练和评估 ====================================

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        
        # 将weight有效的部分设置为1，无效的部分设置为0
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        
        # 将reduction设置为'none'，否则无法加权
        self.reduction='none'

        # pred转置为(batch_size, vocab_size, num_steps)
        # MaskedSoftmaxCELoss中内置了Softmax操作
        # 可以将（B, C, T) 转换为对应类别的（B, T)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        
        # 只要非<pad>的部分参与计算
        # 此时 unweighted_loss 的形状为 (batch_size, num_steps)
        # 记录了每个位置的损失值
        # 沿着 时间步（num_steps）维度计算平均损失，得到每个序列的损失
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        
        # weighted_loss的形状为(batch_size,)，表示每个序列的平均损失
        return weighted_loss

def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

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

def bleu(pred_seq, label_seq, k):
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


if __name__ == '__main__':

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

    # 定义Bahdanau注意力机制的Seq2Seq编码器和解码器
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    encoder.eval()
    decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
    decoder.eval()

    # 测试编码器和解码器
    print("================= Encoder Test =================")
    X = torch.zeros((4, 7), dtype=torch.long)  # (batch_size,num_steps)
    enc_outputs = encoder(X)
    print(f'X shape: {X.shape}')
    print(f'enc_outputs.outputs shape: {enc_outputs[0].shape}') # (num_steps, batch_size, num_hiddens) = (7, 4, 16)
    print(f'enc_outputs.state shape: {enc_outputs[1].shape}')   # (num_layers, batch_size, num_hiddens) = (2, 4, 16)

    print("================= Decoder Test =================")
    state = decoder.init_state(enc_outputs, None)                       # [转置的enc_outputs, hidden_state, enc_valid_lens]
    print(f'init_state len: {len(state)}')
    print(f'init_state[0](转置的enc_outputs) shape: {state[0].shape}')   # (batch_size, num_steps, num_hiddens) = (4, 7, 16)
    print(f'init_state[1](hidden_state) shape: {state[1].shape}')       # (num_layers, batch_size, num_hiddens) = (2, 4, 16)


    dec_output, state = decoder(X, state)
    print(f'dec_output shape: {dec_output.shape}')              # (batch_size, num_steps, vocab_size) = (4, 7, 10)
    print(f'state[0](enc_outputs) shape: {state[0].shape}')  # (batch_size, num_steps, num_hiddens) = (4, 7, 16)
    print(f'state[1](hidden_state) shape: {state[1].shape}')  # (num_layers, batch_size, num_hiddens) = (2, 4, 16)


    # exit()


    #  ==================================== 模型训练 ====================================
    
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs = 0.005, 250

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    
    
    # 检查是否有训练好的模型，如果有则不需要重新训练
    try:
        net.load_state_dict(torch.load('../models/10.4-bahdanau-attention.pth'))
        print("Loaded pre-trained model.")
        net = net.to(device)
    except FileNotFoundError:
        print("No pre-trained model found, training from scratch.")

        # 初始化权重
        net.apply(xavier_init_weights)
        net = net.to(device)

        # 定义损失函数
        loss = MaskedSoftmaxCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        # 训练模型
        net.train()
        record_loss = []
        for epoch in range(num_epochs):
            total_loss, num_tokens = 0.0, 0
            for batch in train_iter:
                # 获取批量数据
                X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch] 
                # X，Y的形状为(batch_size, num_steps)
                # X_valid_len，Y_valid_len的形状为(batch_size,)

                """
                    举个例子，当序列长度为8时，
                    此时 X 为 [h, e, l, l, o, <pad>, <pad>, <pad>]
                    同时 Y 为 [b, o, n, j, o, u, r, <pad>]
                    X_valid_len 为 5，Y_valid_len 为 7
                """

                # 在目标序列的前面添加一列<bos>
                bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0], device=device).reshape(-1, 1)
                dec_X = torch.cat((bos, Y[:, :-1]), 1) # 构成decoder的输入

                """
                    关于强制教学 Teacher Forcing：
                    在训练过程中，我们直接使用目标序列Y的前面部分作为解码器的输入，
                    这被称为强制教学（Teacher Forcing）。这样可以加快收敛速度，
                    对于dec_X，我们将其“错位”一个时间步，并在前面添加一个<bos>标记，
                    即 [b, o, n, j, o, u, r, <pad>] 
                    变为 [<bos>, b, o, n, j, o, u, r]，
                """
                
                # 前向传播
                Y_hat, _ = net(X, dec_X, X_valid_len) # 这里的X_valid_len是源语言的有效长度，在本例中不影响解码器的输入
                loss_value = loss(Y_hat, Y, Y_valid_len)

                """
                    在前向传播时
                    针对输入序列X和解码器的输入dec_X，计算模型的输出Y_hat。
                    输入序列X会在模型中被Encoder处理，得到context（即Encoder的最后状态），
                    然后解码器使用这个context和dec_X来生成预测的目标序列Y_hat。
                    对于[<bos>, b, o, n, j, o, u, r]，解码器会预测下一个词元，
                    我们期待得到的输出是[b, o, n, j, o, u, r, <eos>]，
                """

                # 反向传播
                optimizer.zero_grad()
                loss_value.sum().backward()
                grad_clipping(net, 1)
                optimizer.step()

                total_loss += loss_value.sum().item()
                num_tokens += Y_valid_len.sum().item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {total_loss / num_tokens:.3f}')

            record_loss.append(total_loss / num_tokens)

        # 绘制损失曲线
        plt.plot(range(1, num_epochs + 1), record_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.show()

        # 保存模型
        torch.save(net.state_dict(), '../models/10.4-bahdanau-attention.pth')

    # ==================================== 模型评估 ====================================
    
    net.eval()

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        
        # 将源语言的句子转换为索引
        src_tokens = src_vocab[eng.lower().split(' ')] + [src_vocab['<eos>']]
        enc_valid_len = torch.tensor([len(src_tokens)], device=device)
        src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
        # 此时 src_tokens 的形状为 (num_steps,)，表示源语言的输入序列
        
        """
            举个例子
            [g, o, ., <eos>] 经过填充后变为
            [g, o, ., <eos>, <pad>, <pad>, <pad>]
        """

        # 执行Encoder的前向传播
        enc_X = torch.tensor(src_tokens, dtype=torch.long, device=device)
        enc_X = torch.unsqueeze(enc_X, 0)  # 添加batch维度，形状为(1, num_steps)
        enc_outputs = net.encoder(enc_X, enc_valid_len)

        # 获得context，并初始化Decoder的状态
        dec_state = net.decoder.init_state(enc_outputs, enc_valid_len) # 这里的enc_valid_len目前没有实际作用

        # 执行Decoder的前向传播
        dec_X = torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device)
        dec_X = dec_X.unsqueeze(0)  # 添加batch维度，形状为(1, 1)        
        output_seq = []
        
        attention_weight_seq = [] # 10.4新增
        save_attention_weights = True  # 是否保存注意力权重

        for _ in range(num_steps):

            # 与encoder最大的区别是，解码器的输入是前一个时间步的输出dec_X和隐藏状态dec_state
            Y_hat, dec_state = net.decoder(dec_X, dec_state)
            """
                注意在这里，有两种方法
                1. Sutskever等人提出的seq2seq模型中，解码器的输入是前一个时间步的输出，和「前一步的隐藏状态」。
                2. Cho等人提出，解码器的输入是前一个时间步的输出和「encoder的最后一个状态」。
            """
            dec_X = Y_hat.argmax(dim=2)  # 取最大概率的词元
            predicted_token = dec_X.squeeze(0).type(torch.int32).item()  # 取出预测的词元

            # 10.4新增
            if save_attention_weights:
                attention_weight_seq.append(net.decoder.attention_weights)

            # 如果预测的词元是<eos>，则停止解码
            if predicted_token == tgt_vocab['<eos>']:
                break

            output_seq.append(predicted_token)

        translation = ' '.join(tgt_vocab.to_tokens(output_seq))
        print(f'源语言: {eng}, 预测的目标语言: {translation}, 真实的目标语言: {fra}')
        print(f'BLEU分数: {bleu(translation, fra, k=2):.3f}')

    # 可视化注意力权重
    # 转移到cpu
    dec_attention_weight_seq = []
    for step in attention_weight_seq:
        t = []
        for i in range(len(step)):
            t.append(step[i].cpu())
        dec_attention_weight_seq.append(t)

    attention_weights = torch.cat([step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((1, 1, -1, num_steps))
    show_heatmaps(attention_weights[:, :, :, :len(engs[-1].split()) + 1],
                  xlabel='Key posistions', ylabel='Query posistions')