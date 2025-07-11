import os
import torch
from torch import nn

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
        data_dir = "../data/glove.6B.50d" 
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
    
# ==================================== 查找函数 ====================================

# TOP-K近邻函数
def knn(W, x, k):
    # 增加1e-9以获得数值稳定性
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

# 查找相似词
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # 排除输入词
        print(f'{embed.idx_to_token[int(i)]}：cosine相似度={float(c):.3f}')

# 查找类比词
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # 删除未知词
    
if __name__ == "__main__":

    # 读取GloVe嵌入
    glove_6b50d = TokenEmbedding('glove.6b.50d')
    print("词典大小:", len(glove_6b50d))
    print("读取测试:", glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])

    # 查找相似词
    print("相似词查询:")
    print(get_similar_tokens('chip', 3, glove_6b50d))
    print(get_similar_tokens('baby', 3, glove_6b50d))
    print(get_similar_tokens('beautiful', 3, glove_6b50d))

    # 查找类比词
    print("类比词查询:")
    print(get_analogy('man', 'woman', 'son', glove_6b50d))
    print(get_analogy('beijing', 'china', 'tokyo', glove_6b50d))
    print(get_analogy('china', 'beijing', 'japan', glove_6b50d))
    print(get_analogy('bad', 'worst', 'big', glove_6b50d))
