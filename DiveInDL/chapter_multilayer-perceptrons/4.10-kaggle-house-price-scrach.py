import pandas
import torch
import numpy as np
import matplotlib.pyplot as plt

# ================ 读取数据 ================

train_data = pandas.read_csv('../data/house-prices-advanced-regression-techniques/train.csv')
test_data = pandas.read_csv('../data/house-prices-advanced-regression-techniques/test.csv')
print(train_data.shape, test_data.shape)
print(train_data.head())
print(test_data.head())

# ================ 数据预处理 ================

all_features = pandas.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) # 去掉ID，训练集还要去掉SalePrice

# 对数值型特征进行标准化
num_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 数值型特征的索引
all_features[num_features] = all_features[num_features].apply(lambda x: (x - x.mean()) / x.std())  # 标准化
all_features[num_features] = all_features[num_features].fillna(0)  # 填充缺失值

# 对类别型特征进行独热编码
cat_features = all_features.dtypes[all_features.dtypes == 'object'].index.tolist()  # 类别型特征的索引
all_features = pandas.get_dummies(all_features, columns=cat_features, dummy_na=True)  # 独热编码，包含缺失值

# 将独热bool类型转换为数值float型（新版本需要）
all_features = all_features.astype({col: 'float32' for col in all_features.select_dtypes(include=[bool]).columns})
print(all_features.shape)

# print(all_features.dtypes.to_string())
# print("非数值列：", all_features.select_dtypes(exclude=[np.number]).columns.tolist())
assert all_features.select_dtypes(exclude=[np.number]).shape[1] == 0, "存在非数值列！"

# 把之前合并的训练集和测试集分开，并转换为torch张量
n_train = train_data.shape[0]  # 训练集的样本数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)  # 训练集特征
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)  # 测试集特征
train_labels = torch.tensor(train_data['SalePrice'].values, dtype=torch.float32)  # 训练集标签
train_labels = train_labels.reshape((-1, 1))  # 转换为二维张量
print(train_features.shape, train_labels.shape, test_features.shape)
# 转前：torch.Size([1460, 330]) torch.Size([1460]) torch.Size([1459, 330])
# 转后：torch.Size([1460, 330]) torch.Size([1460, 1]) torch.Size([1459, 330])
# 需要转化为列向，否则可能会触发广播错误

# ================ 训练模型 ================

loss = torch.nn.MSELoss()  # 均方误差损失函数
num_inputs = train_features.shape[1]  # 特征数
num_outputs = 1  # 输出数（回归问题）

# net = torch.nn.Sequential(
#     torch.nn.Linear(num_inputs, num_outputs)
# )
# 不能直接定义，因为在k折交叉验证中需要多次创建模型
def get_net():
    return torch.nn.Sequential(
        # torch.nn.Linear(num_inputs, num_outputs),
        torch.nn.Linear(num_inputs, 64),  # 第一层，输入层到隐藏层
        torch.nn.ReLU(),  # 激活函数
        torch.nn.Linear(64, 64),  # 第二层，隐藏层到隐藏层
        torch.nn.ReLU(),  # 激活函数
        torch.nn.Linear(64, num_outputs)  # 输出层，隐藏层到输出层
    )

def log_rmse(net, features, labels):
    with torch.no_grad(): # 不需要计算梯度，否则会浪费计算图内存
        pred = net(features)
        cliped_preds = torch.clamp(pred, min=1.0, max=float('inf'))  # 限制预测值的范围
        rmse = torch.sqrt(loss(cliped_preds.log(), labels.log()))  # 计算对数均方根误差
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels, num_epochs=100, lr=0.01, weight_decay=0.001, batch_size=64):
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)  # 创建训练集数据集
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  # Adam优化器

    record_train_rmse = []  # 记录训练集RMSE
    record_test_rmse = []  # 记录测试集RMSE

    for epoch in range(num_epochs):
        for X, y in train_loader:
            optimizer.zero_grad()  # 清零梯度
            l = loss(net(X), y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
        record_train_rmse.append(log_rmse(net, train_features, train_labels))
        if test_features is not None and test_labels is not None:
            record_test_rmse.append(log_rmse(net, test_features, test_labels))
    
    return record_train_rmse, record_test_rmse

def get_k_fold_data(k, i, features, labels):
    num_data = features.shape[0]  # 数据总数
    assert k > 1 and i >= 0 and i < k, "k 必须大于 1 且 i 在 [0, k) 范围内"
    assert features.shape[0] == labels.shape[0], "特征和标签的样本数必须相同"

    fold_size = num_data // k  # 每折的大小
    indices = list(range(num_data))  # 索引列表
    # np.random.shuffle(indices)  # 打乱索引顺序

    fold_indices = indices[i * fold_size: (i + 1) * fold_size]  # 当前折的索引
    train_indices = [idx for idx in indices if idx not in fold_indices]  # 除了当前折的索引，即为训练集的索引

    return features[train_indices], labels[train_indices], features[fold_indices], labels[fold_indices]

def k_fold_cross_validation(k, features, labels, num_epochs=100, lr=0.01, weight_decay=0.001, batch_size=64):

    record_loss_sum_train = []  # 记录每折训练集的RMSE
    record_loss_sum_test = []  # 记录每折测试集的RMSE

    for i in range(k):
        print(f"Fold {i + 1}/{k}")
        train_features, train_labels, test_features, test_labels = get_k_fold_data(k, i, features, labels)
        net = get_net()  # 获取新的模型
        train_rmse, test_rmse = train(net, train_features, train_labels, test_features, test_labels, num_epochs, lr, weight_decay, batch_size)
        record_loss_sum_train.append(train_rmse[-1])  # 记录最后一个epoch的训练集RMSE
        record_loss_sum_test.append(test_rmse[-1])  # 记录最后一个epoch的测试集RMSE

        print(f"Fold {i + 1} - Train RMSE: {train_rmse[-1]:.4f}, Test RMSE: {test_rmse[-1]:.4f}")

    return record_loss_sum_train / k, record_loss_sum_test / k


# ================ 运行k折交叉验证 ================
k = 5  # 折数
num_epochs = 100  # 训练轮数
lr = 0.01  # 学习率
weight_decay = 100  # 权重衰减
batch_size = 256  # 批量大小
# train_rmse, test_rmse = k_fold_cross_validation(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

record_loss_sum_train, record_loss_sum_test = [], []
for i in range(k):
    print(f"Fold {i + 1}/{k}")
    train_features_fold, train_labels_fold, test_features_fold, test_labels_fold = get_k_fold_data(k, i, train_features, train_labels)
    net = get_net()  # 获取新的模型
    train_rmse, test_rmse = train(net, train_features_fold, train_labels_fold, test_features_fold, test_labels_fold, num_epochs, lr, weight_decay, batch_size)
    
    # 画图
    if i == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_rmse, label='Train RMSE', color='blue')
        plt.plot(range(1, num_epochs + 1), test_rmse, label='Test RMSE', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('RMSE over Epochs for Fold 1')
        plt.legend()
        plt.grid()
        plt.show()
    
    record_loss_sum_train.append(train_rmse[-1])  # 记录最后一个epoch的训练集RMSE
    record_loss_sum_test.append(test_rmse[-1])  # 记录最后一个epoch的测试集RMSE

    print(f"Fold {i + 1} - Train RMSE: {train_rmse[-1]:.4f}, Test RMSE: {test_rmse[-1]:.4f}")

# ================ 计算平均RMSE ================
print(f"Average Train RMSE: {np.mean(record_loss_sum_train):.4f}, Average Test RMSE: {np.mean(record_loss_sum_test):.4f}")

# ================ 最终模型训练 ================
final_net = get_net()  # 获取新的模型
final_train_rmse, _ = train(final_net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
print(f"Final Train RMSE: {final_train_rmse[-1]:.4f}")

# ================ 预测测试集 ================
final_net.eval()  # 切换到评估模式
with torch.no_grad():
    test_preds = final_net(test_features)  # 预测测试集
    test_preds = torch.clamp(test_preds, min=1.0, max=float('inf'))  # 限制预测值的范围
    test_preds = test_preds.numpy()  # 转换为numpy数组
# 保存预测结果
submission = pandas.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds.flatten()})  # 转换为DataFrame
submission.to_csv('../data/house-prices-advanced-regression-techniques/submission.csv', index=False)  # 保存到CSV文件