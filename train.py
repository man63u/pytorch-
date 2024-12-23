import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import CNN

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.RandomCrop(32, padding=4), # 随机裁剪
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))# 归一化：将图像像素值标准化为均值和标准差
])

# 替换默认 URL 为国内镜像
torchvision.datasets.CIFAR10.url = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128,
                         shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128,
                        shuffle=False, num_workers=2)

# 初始化模型
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练函数
def train(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0 # 累计损失初始化为0
        for i, (inputs, labels) in enumerate(trainloader):  # 从训练数据加载器trainloader中迭代获取小批量数据（enumerate用于获取小批量的索引i和对应的数据（inputes，labels））
            inputs, labels = inputs.to(device), labels.to(device) # 将数据移到CPU\GPU

            optimizer.zero_grad()  # 清空上一次的梯度
            outputs = model(inputs)  # 向前传播，得到模型输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 优化模型参数 逐步调整参数，使损失函数的值尽可能小

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')  # 每100个小批量打印一次损失
                running_loss = 0.0  # 重置累计损失

        # 每个epoch结束后评估模型
        evaluate()


# 评估函数
def evaluate():
    model.eval()  # 将模型设为评估模式
    correct = 0  # 用于记录预测正确的样本数
    total = 0  # 总样本数
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in testloader:  # 遍历每个测试机的批量
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # 向前传播，计算模型输出
            _, predicted = torch.max(outputs.data, 1)  # 获取输出的最大值对应的类别
            total += labels.size(0)  # 累计样本总数
            correct += (predicted == labels).sum().item()

    print(f'准确率: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    train(epochs=50)