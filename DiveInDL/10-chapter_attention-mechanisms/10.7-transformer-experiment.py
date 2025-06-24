import torch
from torch import nn



if __name__ == "__main__":


    # =================== 使用normal_初始化权重 ===================
    
    w1 = torch.empty(5, 16)  # 假设是num_embeddings, embedding_dim
    w2 = torch.empty(5, 256)

    # 使用torch.init.normal_进行初始化
    nn.init.normal_(w1)
    nn.init.normal_(w2)

    # 打印张量的均值和方差
    print("Using normal_ initialization:")
    print(f"W1 Mean: {w1.mean()}, Variance: {w1.var()}")
    print(f"W2 Mean: {w2.mean()}, Variance: {w2.var()}")

    """
        可以看出，使用normal_初始化的权重分布并不会根据embedding_dim的大小而变化
        Var(w1)和Var(w2)的值都是接近1.0的，这表明它们的方差是相似的。
    """

    # =================== 使用xavier_normal_初始化权重 ===================

    w3 = torch.empty(5, 16)  # 假设是num_embeddings, embedding_dim
    w4 = torch.empty(5, 256)

    # 使用torch.init.xavier_normal_进行初始化
    nn.init.xavier_normal_(w3)
    nn.init.xavier_normal_(w4)

    # 打印张量的均值和方差
    print("\nUsing xavier_normal_ initialization:")
    print(f"W3 Mean: {w3.mean()}, Variance: {w3.var()}")
    print(f"W4 Mean: {w4.mean()}, Variance: {w4.var()}")

    """
        而使用xavier_normal_初始化的权重分布会根据embedding_dim的大小而变化
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        Var(w3)和Var(w4)是根据embedding_dim的大小而变化的。
        可以看出，w3的方差约为 1 / sqrt(16) = 0.25，而w4的方差约为 1 / sqrt(64) = 0.125
    """

    # 如果将 w3 和 w4 都乘以 sqrt(16) 或 sqrt(256)，    
    w3 *= 4.0  # sqrt(16)
    w4 *= 16.0  # sqrt(256)
    
    # 打印张量的均值和方差
    print("\nAfter scaling:")
    print(f"W3 Mean: {w3.mean()}, Variance: {w3.var()}")
    print(f"W4 Mean: {w4.mean()}, Variance: {w4.var()}")

    # =================== 测试nn.embedding ===================

    # 测试nn.embedding
    embedding1 = nn.Embedding(5, 16)
    embedding2 = nn.Embedding(5, 256)

    # 创建测试batch（16，5，10）
    X = torch.randint(0, 5, (5, 5, 10))

    # 获取嵌入向量
    embedded1 = embedding1(X)
    embedded2 = embedding2(X)

    print(f"Embedded1 Shape: {embedded1.shape}, Embedded2 Shape: {embedded2.shape}")

    # 计算每一行（即每个token）的均值和方差，以及数值范围
    for i in range(embedded1.shape[0]):
        mean1 = embedded1[i].mean().item()
        var1 = embedded1[i].var().item()
        min1 = embedded1[i].min().item()
        max1 = embedded1[i].max().item()

        mean2 = embedded2[i].mean().item()
        var2 = embedded2[i].var().item()
        min2 = embedded2[i].min().item()
        max2 = embedded2[i].max().item()

        print(f"Row {i} - Embedded1: Mean={mean1}, Variance={var1}, Min={min1}, Max={max1}")
        print(f"Row {i} - Embedded2: Mean={mean2}, Variance={var2}, Min={min2}, Max={max2}")
    
    # 如果手动初始化权重
    w5 = torch.empty(5, 16)
    w6 = torch.empty(5, 256)

    # 使用torch.init.xavier_normal_进行初始化
    nn.init.xavier_normal_(w5)
    nn.init.xavier_normal_(w6)

    # 实例nn.Embedding并手动设置权重
    embedding3 = nn.Embedding(5, 16)
    embedding4 = nn.Embedding(5, 256)

    # 手动设置权重
    embedding3.weight.data = w5
    embedding4.weight.data = w6

    # 获取嵌入向量
    embedded3 = embedding3(X)
    embedded4 = embedding4(X)

    print(f"Embedded3 Shape: {embedded3.shape}, Embedded4 Shape: {embedded4.shape}")

    # 计算每一行（即每个token）的均值和方差，以及数值范围
    for i in range(embedded3.shape[0]):
        mean3 = embedded3[i].mean().item()
        var3 = embedded3[i].var().item()
        min3 = embedded3[i].min().item()
        max3 = embedded3[i].max().item()

        mean4 = embedded4[i].mean().item()
        var4 = embedded4[i].var().item()
        min4 = embedded4[i].min().item()
        max4 = embedded4[i].max().item()

        print(f"Row {i} - Embedded3: Mean={mean3}, Variance={var3}, Min={min3}, Max={max3}")
        print(f"Row {i} - Embedded4: Mean={mean4}, Variance={var4}, Min={min4}, Max={max4}")