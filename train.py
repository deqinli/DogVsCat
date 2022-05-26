import torch
from torch.utils.data import DataLoader
from AlexNet import AlexNet
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import numpy as np


from DataLoader import DogCatDataSet


random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

epochs = 70  # 训练次数
batch_size = 32  # 批处理大小
num_workers = 4  # 多线程的数目
use_gpu = torch.cuda.is_available()

# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

])


train_dataset = DogCatDataSet(img_dir="./data/train", transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

test_dataset = DogCatDataSet(img_dir="./data/validation", transform=data_transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)



net = AlexNet(num_classes=2)


if use_gpu:
    net = net.cuda()
# print(net)



# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

if __name__ == '__main__':

    # 开始训练
    net.train()
    for epoch in range(epochs):
        if((epoch + 1) % 20 == 0):
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = lr * 0.1
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            # inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            # import pdb
            # pdb.set_trace()
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # print("epoch: ",  epoch, " loss: ", loss.item())
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
        epoch + 1, running_loss / train_total * batch_size, 100 * train_correct / train_total))

        # 模型测试
        correct = 0
        test_loss = 0.0
        test_total = 0

        net.eval()
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total * batch_size, 100 * correct / test_total))
    torch.save(net, "./model/AlexNet.pth")
