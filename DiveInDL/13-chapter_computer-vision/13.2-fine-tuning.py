import os
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt

def eval_accuracy(net, data_iter, device):
    """计算模型在数据集上的准确率"""
    net.eval()  # 设置为评估模式
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            acc_sum += (net(X).argmax(dim=1) == y).sum().item()
            n += y.shape[0]
    return acc_sum / n

# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True, devices=None):
    
    # 数据迭代器
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction="none")

    # 优化器
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
        

    # ============================ 训练模型 ===========================
    record_train_loss = []
    record_train_acc = []
    record_test_acc = []

    # 多GPU训练
    net = nn.DataParallel(net, device_ids=devices).to(devices[0]) if devices else net.to(device)

    for epoch in range(num_epochs):

        loss_train_sum, n = 0.0, 0
        acc_train_sum = 0.0
        
        net.train()
        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)

            trainer.zero_grad()
            l.sum().backward()
            trainer.step()

            loss_train_sum += l.sum().item()
            acc_train_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        train_loss = loss_train_sum / n
        train_acc = acc_train_sum / n

        record_train_loss.append(train_loss)
        record_train_acc.append(train_acc)
        print(f"epoch {epoch + 1}, train loss {train_loss:.4f}, train acc {train_acc:.4f}")

        # 评估
        net.eval()
        test_acc = eval_accuracy(net, test_iter, device)
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

    # =========================== 预览数据 ===========================
    data_dir = "../data/hotdog"
    train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
    test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

    # 可视化
    hotdogs = [train_imgs[i][0] for i in range(8)]
    not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
    plt.figure(figsize=(10, 5))
    for i in range(8):
        plt.subplot(2, 8, i + 1)
        plt.imshow(hotdogs[i])
        plt.axis('off')
        plt.subplot(2, 8, i + 9)
        plt.imshow(not_hotdogs[i])
        plt.axis('off')
    plt.show()

    # =========================== 定义数据增强 ===========================

    # 使用RGB通道的均值和标准差，以标准化每个通道
    normalize = torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 数据来自ImageNet

    train_augs = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224), # 从图像中裁切随机大小和长宽比的区域，并缩放到224x224
        torchvision.transforms.RandomHorizontalFlip(), # 随机水平翻转图像
        torchvision.transforms.ToTensor(),
        normalize])

    test_augs = torchvision.transforms.Compose([
        torchvision.transforms.Resize([256, 256]),     # 将图像缩放到256x256
        torchvision.transforms.CenterCrop(224),        # 从中心裁切224x224的区域 
        torchvision.transforms.ToTensor(),
        normalize])
    
    # =========================== 定义模型 ===========================
    
    # 导入预训练模型
    pretrained_net = torchvision.models.resnet18(pretrained=False)
    pretrained_net.load_state_dict(torch.load("../checkpoints/resnet18-f37072fd.pth"))
    print(pretrained_net.fc)  # 查看最后一层

    # 定义微调模型
    finetune_net = torchvision.models.resnet18(pretrained=False)
    finetune_net.load_state_dict(torch.load("../checkpoints/resnet18-f37072fd.pth"))
    finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)  # 修改最后一层为2个输出
    nn.init.xavier_uniform_(finetune_net.fc.weight)

    # =========================== 训练微调模型 ===========================
    
    train_fine_tuning(finetune_net, 5e-5)

    # ============================ 从头开始训练 ===========================

    scratch_net = torchvision.models.resnet18()
    scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)  # 修改最后一层为2个输出
    nn.init.xavier_uniform_(scratch_net.fc.weight)
    train_fine_tuning(scratch_net, 5e-4, param_group=False)