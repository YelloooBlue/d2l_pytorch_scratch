import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

# 双线性插值核
def bilinear_kernel(in_channels, out_channels, kernel_size):
    
    # 计算因子
    factor = (kernel_size + 1) // 2
    """
        该因子决定了从核中心到边缘的权重衰减速率
    """
    
    # 计算核的中心位置
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    # 生成坐标
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    """
        假设kernel_size为3，则
            og[0] = [[0],
            [1],
            [2]]
            og[1] = [[0, 1, 2]]
    """
    
    # 计算权重
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    """
        假设kernel_size为3，则
            filt = [[0.25, 0.5, 0.25],
                    [0.5, 1.0, 0.5],
                    [0.25, 0.5, 0.25]]
    """

    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt

    """
        在本例中，in_channels和out_channels相等
        所以填充的是weight的对角线
    """

    return weight
    # 输出(in_channels, out_channels, kernel_size, kernel_size)的双线性插值核


# ============================ 数据集 ===========================

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

# 读取数据集
def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = "../data/VOCdevkit/VOC2012"
    num_workers = 1
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter

# ============================ 训练辅助函数 ===========================
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

# 准确率（精度）
def accuracy(y_hat, y):
    # 一个batch_size中有多少个样本预测正确
    y_hat = torch.argmax(y_hat, dim=1) # 形状是(batch_size,)
    return (y_hat == y).float().sum().item() # 形状是(batch_size,)

# def eval_accuracy(net, data_iter, device):
#     """计算模型在数据集上的准确率"""
#     net.eval()  # 设置为评估模式
#     acc_sum, n = 0.0, 0
#     with torch.no_grad():
#         for X, y in data_iter:
#             X, y = X.to(device), y.to(device)
#             acc_sum += (net(X).argmax(dim=1) == y).sum().item()
#             n += y.shape[0]
#     return acc_sum / n

"""
    这里不能用eval_accuracy，因为它是用的是y.shape[0]
    而evaluate_accuracy_gpu是用的y.numel()，即计算所有元素的数量
    针对不同的任务类型，例如图像分类和目标检测，可能需要不同的评估方法
    目标检测更适合 evaluate_accuracy_gpu
"""

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

    # =========================== 测试数据集 ===========================

    voc_dir = "../data/VOCdevkit/VOC2012"
    train_features, train_labels = read_voc_images(voc_dir, True)

    n = 5
    imgs = train_features[0:n] + train_labels[0:n]
    imgs = [img.permute(1,2,0) for img in imgs]

    # 显示前5张图像和标签
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(imgs[i])
        plt.axis('off')
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(imgs[i + n])
        plt.axis('off')
    plt.show()
    
    # =========================== 预训练模型 ===========================

    # 导入预训练模型
    pretrained_net = torchvision.models.resnet18(pretrained=False)
    pretrained_net.load_state_dict(torch.load("../checkpoints/resnet18-f37072fd.pth"))
    
    print(list(pretrained_net.children())[-3:])  # 查看最后三层
    """
        Sequential          保留，输出为512维的特征图，宽度和高度变为原来的1/32
        AdaptiveAvgPool2d   不需要
        Linear              不需要
    """

    # 去掉最后两层
    net = nn.Sequential(*list(pretrained_net.children())[:-2])

    # 查看
    X = torch.rand(size=(1, 3, 320, 480))
    print(net(X).shape) # 宽度和高度变为原来的1/32

    # =========================== 修改网络 ===========================

    num_classes = 21  # 21个类别

    # 添加1x1卷积层，将512维的特征图映射到21个类别
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    
    # 添加转置卷积层，将特征图上采样到原来的大小
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=64, padding=16, stride=32))

    # =========================== 测试上采样（转置卷积） ===========================
    conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
    conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

    # DiveInDL/img/catdog.jpg
    img = torchvision.transforms.ToTensor()(Image.open('DiveInDL/img/catdog.jpg'))
    X = img.unsqueeze(0)
    Y = conv_trans(X)
    out_img = Y[0].permute(1, 2, 0).detach()

    plt.imshow(out_img)
    plt.show()

    # =========================== 读取数据集 ===========================
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)

    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size)

    # =========================== 训练模型 ===========================

    num_epochs, lr, wd = 5, 0.001, 1e-3
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

    # ============================ 训练模型 ===========================
    
    # 检查是否有训练好的模型，如果有则不需要重新训练
    train_from_scratch = False
    try:
        net.load_state_dict(torch.load('../models/13.11-fcn.pth'))
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
        torch.save(net.state_dict(), '../models/13.11-fcn.pth')

    # =========================== 预测 ===========================

    net.eval()  # 设置为评估模式

    def predict(img):
        X = test_iter.dataset.normalize_image(img).unsqueeze(0)
        pred = net(X.to(device)).argmax(dim=1)
        return pred.reshape(pred.shape[1], pred.shape[2])
    
    def label2image(pred):
        colormap = torch.tensor(VOC_COLORMAP, device=device)
        X = pred.long()
        return colormap[X, :]

    voc_dir = "../data/VOCdevkit/VOC2012"
    test_images, test_labels = read_voc_images(voc_dir, False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1,2,0), pred.cpu(),
                torchvision.transforms.functional.crop(
                    test_labels[i], *crop_rect).permute(1,2,0)]
        
    # 显示预测结果
    plt.figure(figsize=(18, 8))
    for i in range(n):
        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(imgs[i * 3])
        plt.axis('off')
        plt.title('Original')
        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(imgs[i * 3 + 1])
        plt.axis('off')
        plt.title('Predicted')
        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(imgs[i * 3 + 2])
        plt.axis('off')
        plt.title('Ground Truth')
    plt.tight_layout()
    plt.show()