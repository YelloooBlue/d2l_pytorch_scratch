import os
import torch
from torch import nn
import collections
from torch.utils import data
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

# =========================== 网络定义 ===========================

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size) # 稍后用GloVe嵌入初始化
        
        # 将bidirectional设置为True以获取双向循环神经网络
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2) # 2个输出类别：正面和负面情感

    def forward(self, inputs):
        # inputs的形状是（批量大小，时间步数）
        # 因为长短期记忆网络要求其输入的第一个维度是时间维，
        # 所以在获得词元表示之前，输入会被转置。

        # 使用GloVe嵌入
        embeddings = self.embedding(inputs.T) # 输出形状为（时间步数，批量大小，词向量维度）

        # 【双向/多层】使得参数在GPU上连续存储，
        self.encoder.flatten_parameters() 
        
        # 返回上一个隐藏层在不同时间步的隐状态，
        outputs, _ = self.encoder(embeddings) # outputs的形状是（时间步数，批量大小，2*隐藏单元数），2=双向
        
        # 取出双向的初始和最终时间步的隐状态（反向和正向的最终隐状态）
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) # 拼接形状为（批量大小，2*2*隐藏单元数）
        
        # 连结初始和最终时间步的隐状态，作为全连接层的输入
        outs = self.decoder(encoding)
        return outs
        # outs的形状是（批量大小，2），2=正面和负面情感
    
# =========================== 辅助函数 ===========================

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])

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

def predict_sentiment(net, vocab, sequence, device):
    """预测文本序列的情感"""
    sequence = torch.tensor(vocab[sequence.split()], device=device) # 将文本序列转换为索引
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'

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

    # 读取数据集测试
    
    # data_dir = "../data/aclImdb"
    # train_data = read_imdb(data_dir, is_train=True)
    # print('训练集数目：', len(train_data[0]))
    # for x, y in zip(train_data[0][:3], train_data[1][:3]):
    #     print('标签：', y, 'review:', x[0:60])

    # train_tokens = tokenize(train_data[0], token='word')
    # vocab = Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

    # # 直方图
    # plt.figure(figsize=(8, 6))
    # plt.xlabel('# tokens per review')
    # plt.ylabel('count')
    # plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50))
    # plt.show()

    # 读取数据集
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)

    # 定义网络
    embed_size, num_hiddens, num_layers = 100, 100, 2
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    net.apply(init_weights)

    # 使用预训练的GloVe嵌入
    glove_embedding = TokenEmbedding('glove.6B.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    print(embeds.shape)

    # 冻结嵌入层
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    # 训练
    lr, num_epochs = 0.01, 5
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")

    # ============================ 训练模型 ===========================
    
    # 检查是否有训练好的模型，如果有则不需要重新训练
    train_from_scratch = False
    try:
        net.load_state_dict(torch.load('../models/15.2-sentiment-analysis-rnn.pth'))
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
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)

                trainer.zero_grad()
                l.sum().backward()
                trainer.step()

                loss_train_sum += l.sum().item()
                # acc_train_sum += accuracy(y_hat, y)
                acc_train_sum += (y_hat.argmax(dim=1) == y).sum().item()

                num_samples += X.shape[0]
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
        torch.save(net.state_dict(), '../models/15.2-sentiment-analysis-rnn.pth')

    # =========================== 预测 ===========================

    net.eval()
    print(predict_sentiment(net, vocab, 'this movie is so great', device))  # 预测正面情感
    print(predict_sentiment(net, vocab, 'this movie is so bad', device))  # 预测负面情感