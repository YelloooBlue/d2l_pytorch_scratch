import torch
from torch import nn
import os
import collections
import math
import matplotlib.pyplot as plt
import json
import multiprocessing
import re
import time  # 添加时间模块用于性能监控

################################## BERT模型组件 ###################################################

# ===================================== Encoder =====================================

# 来自9.7
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

# 来自10.3
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
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
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 来自10.5
def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # num_hiddens/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])

# 来自10.5
def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

# 来自10.5
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

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

# 来自10.7
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

# 来自10.7
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

# 来自10.7
class EncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens)) # Y的形状:(batch_size,num_steps,num_hiddens)
        return self.addnorm2(Y, self.ffn(Y))

# 来自14.8
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
    
# ===================================== 任务 =====================================

# 来自14.8
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):

        # X的形状:(batch_size，序列长度，num_hiddens)
        # pred_positions的形状:(batch_size，num_pred_positions)

        num_pred_positions = pred_positions.shape[1] # 获取预测位置的数量
        pred_positions = pred_positions.reshape(-1)  # 将预测位置展平为一维
        
        batch_size = X.shape[0]
        # 假设batch_size=2，num_pred_positions=3
        batch_idx = torch.arange(0, batch_size) # [0, 1]
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions) # [0, 0, 0, 1, 1, 1]

        masked_X = X[batch_idx, pred_positions]
        """
            batch_idx: tensor([0, 0, 0, 1, 1, 1])
            pred_positions: tensor([1, 5, 2, 6, 1, 5])
            X[batch_idx, pred_positions] 取出的是 X[0, 1], X[0, 5], X[0, 2], X[1, 6], X[1, 1], X[1, 5]
        """
        # masked_X的形状:(batch_size*num_pred_positions, num_hiddens)

        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    
# 来自14.8
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
    
# ===================================== 整合 =====================================

# 来自14.8
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

################################### BERT微调数据集构建 ###################################

def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word': # 按空格拆分单词
        return [line.split() for line in lines]
    elif token == 'char': # 按字符拆分
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# 来自14.8
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

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

# =========================== SNLI数据集 ===========================

class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]

        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # 使用4个进程
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)

def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签"""
    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

###################################### 下游模型 ######################################

# ============================ BERT下游任务分类器 ===========================

class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
    
# ============================ 辅助函数 ===========================

def evaluate_accuracy_gpu(net, data_iter, device=None): #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    correct_predictions, total_predictions = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            predictions = net(X)
            correct_predictions += (predictions.argmax(dim=1) == y).sum().item()
            total_predictions += y.numel()
    return correct_predictions / total_predictions

def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = "../checkpoints/bert.small.torch"
    # 定义空词表以加载预定义词表
    vocab = Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 加载预训练BERT参数
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab


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
    
    # 加载预训练BERT模型
    bert, vocab = load_pretrained_model('bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=[device])

    # 加载SNLI数据集（用于BERT微调）
    # 优化批次大小和worker数量以提升训练速度
    batch_size, max_len = 64, 128  # 减小批次大小，减少内存占用，提高训练稳定性
    
    # 根据平台和CPU核心数优化worker数量
    if os.name == 'nt':  # Windows
        num_workers = 0  # Windows上多进程可能有问题
    else:
        num_workers = min(8, multiprocessing.cpu_count())
    
    print(f"Using batch_size={batch_size}, num_workers={num_workers}")
    
    data_dir = '../data/snli_1.0'
    train_set = SNLIBERTDataset(read_snli(data_dir, True), max_len, vocab)
    test_set = SNLIBERTDataset(read_snli(data_dir, False), max_len, vocab)
    
    # 使用pin_memory和persistent_workers加速数据加载
    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size, num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 定义下游任务分类器
    net = BERTClassifier(bert)

    # 设置训练参数
    lr, num_epochs = 2e-5, 5  # 使用更合适的学习率
    trainer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)  # 使用AdamW优化器
    loss = nn.CrossEntropyLoss()  # 简化损失函数，避免手动求和
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.LinearLR(trainer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    
    # 使用混合精度训练加速
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler() if device.type == 'cuda' else None

    # ============================ 训练模型 ===========================
    
    # 检查是否有训练好的模型，如果有则不需要重新训练
    train_from_scratch = False
    try:
        net.load_state_dict(torch.load('../models/15.7-natural-language-inference-bert.pth', map_location=device))
        print("Loaded pre-trained model.")
        # train_from_scratch = False
    except FileNotFoundError:
        print("No pre-trained model found, training from scratch.")
        train_from_scratch = True
    
    # 将模型移动到设备（只执行一次）
    net = net.to(device)

    if train_from_scratch:

        record_train_loss = []
        record_train_acc = []
        record_test_acc = []

        # 多GPU训练（如果可用）
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            net = nn.DataParallel(net)

        print("Starting training...")
        net.train()
        
        # 记录训练开始时间
        training_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0

            # 添加进度显示
            print(f"Epoch {epoch + 1}/{num_epochs}")
            
            for i, (X, y) in enumerate(train_iter):
                # 数据传输到设备
                if isinstance(X, list):
                    X = [x.to(device, non_blocking=True) for x in X]
                else:
                    X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                # 使用混合精度训练
                if scaler:
                    with autocast():
                        y_hat = net(X)
                        l = loss(y_hat, y)
                else:
                    y_hat = net(X)
                    l = loss(y_hat, y)

                trainer.zero_grad()
                
                if scaler:
                    scaler.scale(l).backward()
                    scaler.step(trainer)
                    scaler.update()
                else:
                    l.backward()
                    trainer.step()

                # 统计
                epoch_loss += l.item()
                epoch_correct += (y_hat.argmax(dim=1) == y).sum().item()
                epoch_total += y.size(0)
                num_batches += 1
                
                # 每100个batch显示一次进度
                if (i + 1) % 100 == 0:
                    current_acc = epoch_correct / epoch_total
                    current_loss = epoch_loss / num_batches
                    print(f"  Batch {i+1}, Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")

            # 计算epoch统计
            epoch_time = time.time() - epoch_start_time
            train_loss = epoch_loss / num_batches
            train_acc = epoch_correct / epoch_total
            
            record_train_loss.append(train_loss)
            record_train_acc.append(train_acc)
            
            # 更新学习率
            scheduler.step()
            
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # 每2个epoch评估一次测试集（减少评估频率）
            if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                net.eval()
                test_acc = evaluate_accuracy_gpu(net, test_iter, device)
                record_test_acc.append(test_acc)
                print(f"  Test Acc: {test_acc:.4f}")
                net.train()
            else:
                record_test_acc.append(record_test_acc[-1] if record_test_acc else 0.0)
        
        total_training_time = time.time() - training_start_time
        print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")

        # 保存模型
        print("Training completed. Saving model...")
        # 如果使用了DataParallel，需要保存module的state_dict
        model_to_save = net.module if hasattr(net, 'module') else net
        torch.save(model_to_save.state_dict(), '../models/15.7-natural-language-inference-bert.pth')
        print("Model saved successfully!")

        # 绘制损失和准确率（画在同一张图上）
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), record_train_loss, label='Train Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid()
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), record_train_acc, label='Train Acc', marker='o')
        plt.plot(range(1, num_epochs + 1), record_test_acc, label='Test Acc', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.show()

    # =========================== 预测和性能监控 ===========================
    
    print("\n" + "="*50)
    print("Performance Summary:")
    print("="*50)
    
    # 最终评估
    print("Evaluating final model performance...")
    start_time = time.time()
    
    net.eval()
    final_test_acc = evaluate_accuracy_gpu(net, test_iter, device)
    
    eval_time = time.time() - start_time
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print(f"Evaluation Time: {eval_time:.2f} seconds")
    print(f"Model Parameters: {sum(p.numel() for p in net.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
    
    # 示例预测函数
    def predict_snli(premise, hypothesis, net, vocab, max_len, device):
        """
        预测前提和假设之间的关系
        """
        net.eval()
        premise_tokens = tokenize([premise.lower()])[0]
        hypothesis_tokens = tokenize([hypothesis.lower()])[0]
        
        # 截断
        while len(premise_tokens) + len(hypothesis_tokens) > max_len - 3:
            if len(premise_tokens) > len(hypothesis_tokens):
                premise_tokens.pop()
            else:
                hypothesis_tokens.pop()
        
        tokens, segments = get_tokens_and_segments(premise_tokens, hypothesis_tokens)
        token_ids = vocab[tokens] + [vocab['<pad>']] * (max_len - len(tokens))
        segments = segments + [0] * (max_len - len(segments))
        valid_len = len(tokens)
        
        # 转换为tensor
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        segments_tensor = torch.tensor([segments], dtype=torch.long).to(device)
        valid_lens_tensor = torch.tensor([valid_len]).to(device)
        
        with torch.no_grad():
            outputs = net([tokens_tensor, segments_tensor, valid_lens_tensor])
            prediction = outputs.argmax(dim=1).item()
        
        label_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        return label_map[prediction]
    
    # 示例预测
    print("\n" + "="*30 + " 示例预测 " + "="*30)
    examples = [
        ("A person on a horse jumps over a broken down airplane.", "A person is training his horse for a competition."),
        ("A person on a horse jumps over a broken down airplane.", "A person is at a diner, ordering an omelette."),
        ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse.")
    ]
    
    for premise, hypothesis in examples:
        prediction = predict_snli(premise, hypothesis, net, vocab, max_len, device)
        print(f"前提: {premise}")
        print(f"假设: {hypothesis}")
        print(f"预测关系: {prediction}")
        print("-" * 80)