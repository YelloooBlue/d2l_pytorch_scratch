import os
import re
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
import collections
import matplotlib.pyplot as plt


# 来自8.2
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word': # 按空格拆分单词
        return [line.split() for line in lines]
    elif token == 'char': # 按字符拆分
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# 来自9.5
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

# 来自3.3
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

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

# =========================== 数据集读取 ===========================

def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集文本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def load_data_imdb(batch_size, num_steps=500):
    """返回数据迭代器和IMDb评论数据集的词表"""
    data_dir = '../data/aclImdb'
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = tokenize(train_data[0], token='word')
    test_tokens = tokenize(test_data[0], token='word')
    vocab = Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

# =========================== SNLI数据集 ===========================

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

class SNLIDataset(torch.utils.data.Dataset):
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = tokenize(dataset[0])
        all_hypothesis_tokens = tokenize(dataset[1])
        if vocab is None:
            self.vocab = Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
    
def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表"""
    num_workers = 1
    data_dir = '../data/snli_1.0'
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab

# =========================== 嵌入模型 ===========================

class TokenEmbedding:
    """GloVe嵌入"""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = "../data/glove.6B.100d" 
        # GloVe网站：https://nlp.stanford.edu/projects/glove/
        """
            形如
            the 0.418 0.24968 0.41242 -0.41242 -0.21242 0.21242 ...
            have 0.418 0.24968 0.41242 -0.41242 -0.21242 0.21242 ...
        """
        # fastText网站：https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # 跳过标题信息，例如fastText中的首行
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

# =========================== 模型定义 ===========================

def mlp(num_inputs, num_hiddens, flatten):
    net = []
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_inputs, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    net.append(nn.Dropout(0.2))
    net.append(nn.Linear(num_hiddens, num_hiddens))
    net.append(nn.ReLU())
    if flatten:
        net.append(nn.Flatten(start_dim=1))
    return nn.Sequential(*net)

"""
    注意：【加强对方序列中自己感兴趣的部分，找到对方序列中与自己序列相关的部分】
    例如：“小明在吃红色的苹果” 和“小明在吃绿色的水果”这两个句子，
         感兴趣的部分可能有“红色”和“绿色”，“苹果”和“水果”。

    1. 将「前提」和「假设」的通过MLP映射到同一维度的向量空间中，为A和B。
    2. bmm计算出相似度矩阵e，e[i,j]表示「序列A」的第i个词元与「序列B」的第j个词元的相似度。
        (batch_size, seq_len_A, seq_len_B)
    3. 对e的每一行进行softmax，得到beta，beta[i,j]表示「序列B」的第j个词元对「序列A」的第i个词元的注意力权重。
    4. 同理，对e的每一列进行softmax，得到alpha，alpha[i,j]表示「序列A」的第i个词元对「序列B」的第j个词元的注意力权重。
"""

class Attend(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Attend, self).__init__(**kwargs)
        self.f = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B):
        # A/B的形状：（批量大小，序列A/B的词元数，embed_size）
        # f_A/f_B的形状：（批量大小，序列A/B的词元数，num_hiddens）
        f_A = self.f(A)
        f_B = self.f(B)
        
        # e为相似度矩阵，形状为（批量大小，序列A的词元数，序列B的词元数）
        e = torch.bmm(f_A, f_B.permute(0, 2, 1))

        beta = torch.bmm(F.softmax(e, dim=-1), B)
        # softmax后的e的形状为（批量大小，序列A的词元数，序列B的词元数）。表示B中每个词元对A中每个词元的重要性
        # beta中的每个词元是「B中所有词元」的加权平均，权重由e的softmax结果决定

        alpha = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), A)
        # 同理，alpha中的每个词元是「A中所有词元」的加权平均，权重由e的softmax结果决定
        return beta, alpha
    
"""
    对比：【比较自身 和 对方序列中自己感兴趣的部分，找出这两个部分之间的关系】
    例如：比较“红色”和“绿色”，以及“苹果”和“水果”，找出它们之间的关系。

    输入 A和B的形状都是（批量大小，各自序列长度（词元数），embed_size）。
    输入 beta和alpha的形状都是（批量大小，各自序列长度（词元数），embed_size）。
        储存了自身每个词元对「对方」序列的关注程度。
    区别在于
        A/B是各自在原始序列空间中的词元向量，包含「自身」每个词元的语义信息。
        beta/alpha则是针对「对方」序列中每个词元的语义信息进行加权平均后的结果，对「对方」序列进行关注后得到的向量。
        V_A和V_B是对比向量，包含了「自身」序列和「对方」序列的语义信息。
"""
    
class Compare(nn.Module):
    def __init__(self, num_inputs, num_hiddens, **kwargs):
        super(Compare, self).__init__(**kwargs)
        self.g = mlp(num_inputs, num_hiddens, flatten=False)

    def forward(self, A, B, beta, alpha):
        V_A = self.g(torch.cat([A, beta], dim=2))
        V_B = self.g(torch.cat([B, alpha], dim=2))
        return V_A, V_B
    
"""
    聚合：【将对比向量进行聚合，得到最终的输出】
    例如：”小明“=”小明，“红色”!=”绿色”，“苹果”约等于”水果“，将这些对比向量进行聚合，得到最终的输出。

    1. V_A和V_B的形状都是（批量大小，序列长度（词元数），num_hiddens）。
    2. 对V_A和V_B的每个序列进行求和，得到（批量大小，num_hiddens）的向量。此时每个序列都被压缩为一个向量，包含了该序列中所有词元的语义信息。
    3. 将V_A和V_B的求和结果进行连结，得到（批量大小，2 * num_hiddens）的向量。
    4. 将连结后的向量送入一个线性层，得到最终的输出Y_hat，形状为（批量大小，num_outputs）。
"""
    
class Aggregate(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, **kwargs):
        super(Aggregate, self).__init__(**kwargs)
        self.h = mlp(num_inputs, num_hiddens, flatten=True)
        self.linear = nn.Linear(num_hiddens, num_outputs)

    def forward(self, V_A, V_B):
        # 对两组比较向量分别求和
        V_A = V_A.sum(dim=1) 
        V_B = V_B.sum(dim=1)
        # 将两个求和结果的连结送到多层感知机中
        Y_hat = self.linear(self.h(torch.cat([V_A, V_B], dim=1)))
        return Y_hat
    
class DecomposableAttention(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_inputs_attend=100,
                 num_inputs_compare=200, num_inputs_agg=400, **kwargs):
        super(DecomposableAttention, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.attend = Attend(num_inputs_attend, num_hiddens)
        self.compare = Compare(num_inputs_compare, num_hiddens)
        # 有3种可能的输出：蕴涵、矛盾和中性
        self.aggregate = Aggregate(num_inputs_agg, num_hiddens, num_outputs=3)

    def forward(self, X):
        premises, hypotheses = X
        A = self.embedding(premises)
        B = self.embedding(hypotheses)
        beta, alpha = self.attend(A, B)
        V_A, V_B = self.compare(A, B, beta, alpha)
        Y_hat = self.aggregate(V_A, V_B)
        return Y_hat

# =========================== 辅助函数 ===========================

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

def predict_snli(net, vocab, premise, hypothesis, device=None):
    """预测前提和假设之间的逻辑关系"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=device)
    hypothesis = torch.tensor(vocab[hypothesis], device=device)
    output = net([premise.reshape((1, -1)),
                  hypothesis.reshape((1, -1))])
    print(f"蕴含：{output[0][0]:.4f}, 矛盾：{output[0][1]:.4f}, 中性：{output[0][2]:.4f}")
    label = torch.argmax(output, dim=1)
    
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

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

    # # 测试数据集读取
    # data_dir = "../data/snli_1.0"
    # train_data = read_snli(data_dir, is_train=True)
    # test_data = read_snli(data_dir, is_train=False)

    # # 打印前3个样本的前提、假设和标签
    # for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    #     print('前提：', x0)
    #     print('假设：', x1)
    #     print('标签：', y)

    # # 统计标签分布
    # for data in [train_data, test_data]:
    #     print([[row for row in data[2]].count(i) for i in range(3)])

    # # 测试SNLI数据集封装
    # train_iter, test_iter, vocab = load_data_snli(128, 50)
    # print(len(vocab))

    # for X, Y in train_iter:
    #     print(X[0].shape)
    #     print(X[1].shape)
    #     print(Y.shape)
    #     break

    # 读取数据集
    batch_size, num_steps = 256, 50
    train_iter, test_iter, vocab = load_data_snli(batch_size, num_steps)

    # 定义模型
    embed_size, num_hiddens = 100, 200
    net = DecomposableAttention(vocab, embed_size, num_hiddens)

    # 导入预训练的GloVe嵌入
    glove_embedding = TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)

    # 训练
    lr, num_epochs = 0.001, 4
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")

    # ============================ 训练模型 ===========================
    
    # 检查是否有训练好的模型，如果有则不需要重新训练
    train_from_scratch = False
    try:
        net.load_state_dict(torch.load('../models/15.5-natural-language-inference-attention.pth'))
        print("Loaded pre-trained model.")
        net = net.to(device)
        # train_from_scratch = False
    except FileNotFoundError:
        print("No pre-trained model found, training from scratch.")
        train_from_scratch = True

    if train_from_scratch:

        record_train_loss = []
        record_train_acc = []
        record_test_acc = []

        # 多GPU训练
        # net = nn.DataParallel(net, device_ids=devices).to(devices[0]) if devices else net.to(device)

        for epoch in range(num_epochs):

            num_samples = 0
            num_labels = 0

            loss_train_sum = 0.0
            acc_train_sum = 0.0

            net = net.to(device)
            net.train()
            for i, (X, y) in enumerate(train_iter):

                # X, y = X.to(device), y.to(device) 在本例中，X是一个元组，包含前提和假设，必须逐个迁移gpu
                if isinstance(X, list):
                    # 微调BERT中所需
                    X = [x.to(device) for x in X]
                else:
                    X = X.to(device)
                y = y.to(device)

                y_hat = net(X)
                l = loss(y_hat, y)

                trainer.zero_grad()
                l.sum().backward()
                trainer.step()

                loss_train_sum += l.sum().item()
                # acc_train_sum += accuracy(y_hat, y)
                acc_train_sum += (y_hat.argmax(dim=1) == y).sum().item()

                # num_samples += X.shape[0]
                num_samples += y.shape[0]  # 在本例中，X是一个元组，包含前提和假设，所以我们使用y.shape[0]来计算样本数量
                num_labels += y.numel()  # 计算所有元素的数量，适用于多标签分类任务

            train_loss = loss_train_sum / num_samples
            train_acc = acc_train_sum / num_labels  # 注意由于是多标签分类任务，这里用的是num_labels

            record_train_loss.append(train_loss)
            record_train_acc.append(train_acc)
            print(f"epoch {epoch + 1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}")

            # 评估
            net.eval()
            # test_acc = eval_accuracy(net, test_iter, device) # 这个是不准确的
            test_acc = evaluate_accuracy_gpu(net, test_iter, device)
            record_test_acc.append(test_acc)
            print(f"\t test acc {test_acc:.4f}")

        # 绘制损失和准确率（画在同一张图上）
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), record_train_loss, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), record_train_acc, label='Train Acc')
        plt.plot(range(1, num_epochs + 1), record_test_acc, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Training Record')
        plt.legend()
        plt.grid()
        plt.show()

        # 保存模型
        torch.save(net.state_dict(), '../models/15.5-natural-language-inference-attention.pth')

    # =========================== 预测 ===========================
    

    print(predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'bad', '.'], device=device))
    print(predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'good', '.'], device=device))
    print(predict_snli(net, vocab, ['he', 'is', 'good', '.'], ['he', 'is', 'not', 'bad', '.'], device=device))
    print(predict_snli(net, vocab, ['he', 'eats', 'an', 'apple', '.'], ['he', 'eats', 'a', 'fruit', '.'], device=device))

    print("预测结果：")

    # 蕴含
    print(predict_snli(net, vocab, ['she', 'is', 'a', 'teacher', '.'], ['she', 'teaches', '.'], device=device))
    print(predict_snli(net, vocab, ['the', 'cat', 'is', 'on', 'the', 'roof', '.'], ['the', 'cat', 'is', 'on', 'the', 'top'], device=device))

    # 矛盾
    print(predict_snli(net, vocab, ['the', 'cat', 'is', 'on', 'the', 'roof', '.'], ['the', 'cat', 'is', 'not', 'on', 'the', 'roof'], device=device))
    print(predict_snli(net, vocab, ['the', 'cat', 'is', 'on', 'the', 'roof', '.'], ['the', 'cat', 'is', 'on', 'the', 'ground'], device=device))

    # 中性
    print(predict_snli(net, vocab, ['the', 'cat', 'is', 'on', 'the', 'roof', '.'], ['she', 'is', 'a', 'teacher', '.'], device=device))
    print(predict_snli(net, vocab, ['the', 'cat', 'is', 'on', 'the', 'roof', '.'], ['the', 'dog', 'is', 'on', 'the', 'roof'], device=device))

