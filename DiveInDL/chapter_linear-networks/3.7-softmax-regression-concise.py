import torch
from torch import nn
from torchvision import datasets, transforms

# 超参数
batch_size = 256
lr = 0.1
num_epochs = 10

num_inputs = 28 * 28  # 展平图像
num_outputs = 10  # 类别数

def load_dataset(batch_size):
    transform = transforms.ToTensor() # 也可以通过加入展平层来简化这部分代码
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":

    # 加载数据集
    train_loader, test_loader = load_dataset(batch_size)

    # 定义模型
    model = nn.Sequential(
        nn.Flatten(),  # 展平层
        nn.Linear(num_inputs, num_outputs),
        # 不需要 nn.Softmax(dim=1)  # nn.CrossEntropyLoss() 会自动处理 softmax
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(num_epochs):

        loss_sum = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_sum / len(train_loader):.4f}')

    print("Training complete.")
   
