# Use tensors to speed up loading data onto the GPU during training.

import h5py
#import h5py_cache as h5c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import torch.multiprocessing
import sys
import time
#torch.multiprocessing.set_start_method('spawn')

from hdf5_dataset import H5Dataset
from network import Net, NetCCFFF

from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/lr00005_fc2nin_bn_200epoch_3')


train_data_path = './data/for_train.h5'
test_data_path ='./data/for_test.h5'
torch.cuda.empty_cache()
def train(model, criterion, optimizer, data, device):
    # Get the inputs and transfer them to the CPU/GPU.
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Reset the parameter gradients.
    optimizer.zero_grad()

    # Forward + backward + optimize.
    outputs = model(inputs)
    loss = criterion(outputs, labels.long())
    loss.backward()
    optimizer.step()

    return loss

def eval(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    print('Testing the network on the test data ...')

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)#输出预测的类别。1表示行的最大值。
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()    #统计标签与预测相同的个数。

    accuracy = 100.0 * float(correct) / float(total)
    print('Accuracy of the network on the test set: %.3f%%' % (
        accuracy))

    return accuracy


with h5py.File(train_data_path, 'r') as db:
    num_train = len(db['images'])
print('Have', num_train, 'total training examples')
num_epochs = 200
max_in_memory = 80000
print_step = 100
repeats = 1
early_stop_loss = 0.0000001
start_idx = 0
end_idx = max_in_memory
iter_per_epoch = int(np.ceil(num_train / float(max_in_memory)))
indices = np.arange(0, num_train, max_in_memory)
indices = list(indices) + [num_train]
print('iter_per_epoch:', iter_per_epoch)
print(indices)


# Use GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the test data.
print('Loading test data ...')
test_set = H5Dataset(test_data_path, 0, 10000)
test_loader = torchdata.DataLoader(test_set, batch_size=64, shuffle=True)

# Create the network.
input_channels = test_set.images.shape[-1]

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 判断当前模式是训练模式还是预测模式
    if not is_training:
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值和方差。这里我们需要保持
            # X的形状以便后面可以做广播运算
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X - mean) ** 2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super(BatchNorm, self).__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成0和1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, X):
        # 如果X不在内存上，将moving_mean和moving_var复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
##############           net
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        BatchNorm(out_channels, num_dims=4),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk


import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


net = nn.Sequential(
    #     nin_block(12, 96, kernel_size=11, stride=4, padding=0),
    #     nn.MaxPool2d(kernel_size=2, stride=1),
    #     nin_block(96, 256, kernel_size=5, stride=1, padding=2),

    #     nn.MaxPool2d(kernel_size=2, stride=1),
    nn.Conv2d(12, 20, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nin_block(50, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=2, stride=1),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 2, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    FlattenLayer())

######################################################   net
#net = NetCCFFF(input_channels)

print(net)

print('Copying network to GPU ...')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs.")
    net = nn.DataParallel(net)
net.to(device)

# Define the loss function and optimizer.
# LR = 0.1
# LR = 0.01
LR = 0.0005
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)

optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0005)

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
accuracy = eval(net, test_loader, device)
accuracies = []
accuracies.append(accuracy)

early_stop = False
losses = []
loss = None
print('Training ...')

for epoch in range(num_epochs):
    print('epoch: %d/%d' % (epoch + 1, num_epochs))
    net.train()

    for param_group in optimizer.param_groups:
        print('learning rate:', param_group['lr'])

    for j in range(iter_per_epoch):
        print('iter: %d/%d' % (j + 1, iter_per_epoch))
        print('Loading data block [%d, %d] ...' % (indices[j], indices[j + 1]))
        dset = []
        train_loader = []
        dset = H5Dataset(train_data_path, indices[j], indices[j + 1])
        #train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True, num_workers=2)
        train_loader = torchdata.DataLoader(dset, batch_size=64, shuffle=True)

        running_loss = 0.0

        for r in range(repeats):
            if r > 1:
                print('repeat: %d/%d' % (r + 1, repeats))
            for i, data in enumerate(train_loader):
                loss = train(net, criterion, optimizer, data, device)

                # print statistics
                running_loss += loss.item()
                if i % print_step == print_step - 1:
                    # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / print_step,
                                      epoch * len(train_loader) + i)

                    print('epoch: {}, batch: {}, loss: {:.4f}'.format(epoch + 1, i + 1, running_loss / print_step))
                    losses.append(running_loss)
                    running_loss = 0.0
            # Evaluate the network on the test dataset.
            accuracy = eval(net, test_loader, device)
            accuracies.append(accuracy)
            model_path = './model/model_' + str(r)+ '.pwf'

            torch.save(net.state_dict(), model_path)
            net.train()
            if early_stop:
                break
        if early_stop:
            break
    if early_stop:
        break

    # Evaluate the network on the test dataset.
    accuracy = eval(net, test_loader, device)
    accuracies.append(accuracy)
    model_path = './model/model_' + str(accuracy) + '.pwf'
    torch.save(net.state_dict(), model_path)
    #scheduler.step()

print('Finished Training')

model_path = './model/model.pwf'
torch.save(net.state_dict(), model_path)

with open('loss_stats.txt', 'w') as f:
    for l in losses:
        f.write("%s\n" % str(l))
with open('accuracy_stats.txt', 'w') as f:
    for a in accuracies:
        f.write("%s\n" % str(a))


