import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNELS = [20, 50, 500]
#CHANNELS = [40, 100, 500]
#CHANNELS = [10, 20, 100]

#CHANNELS = [6, 16, 120, 84]
#CHANNELS = [12, 32, 120, 84]
#CHANNELS = [32, 32, 120, 84]

class NetCCFFF(nn.Module):
    def __init__(self, input_channels):
        super(NetCCFFF, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, CHANNELS[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], 5)
        self.fc1 = nn.Linear(CHANNELS[1] * 12 * 12, CHANNELS[2])
        self.fc2 = nn.Linear(CHANNELS[2], CHANNELS[3])
        self.fc3 = nn.Linear(CHANNELS[3], 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Net(nn.Module):
    def __init__(self, input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, CHANNELS[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], 5)
        self.fc1 = nn.Linear(CHANNELS[1] * 12 * 12, CHANNELS[2])
        self.fc2 = nn.Linear(CHANNELS[2], 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class one_one_Net(nn.Module):
    def __init__(self, input_channels):
        super(one_one_Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, CHANNELS[0], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(CHANNELS[0], CHANNELS[1], 5)

        self.fc1 = nn.Linear(CHANNELS[1] * 12 * 12, CHANNELS[2])
        self.fc2 = nn.Linear(CHANNELS[2], 2)

    #         self.fc2 =nn.Linear(1,CHANNELS[2])
    #         self.fc3 = nn.Linear(CHANNELS[2], 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape)
        #         x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
####  batch normolizaton
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
###          没使用batch normalization
class one_fl_Net(nn.Module):
    def __init__(self, input_channels):
        super(one_fl_Net, self).__init__()

        #         self.conv1 = nn.Conv2d(input_channels, CHANNELS[0], 5)
        #         self.pool = nn.MaxPool2d(2, 2)

        self.a_1 = nn.Conv2d(12, 96, kernel_size=11, stride=4, padding=0)
        self.a_one = nn.Conv2d(96, 96, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.b_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.b_one = nn.Conv2d(256, 256, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.c_1 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.c_one = nn.Conv2d(384, 384, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.consoft = nn.Conv2d(384, 50, kernel_size=1, stride=1, padding=1)
        self.fc1 = nn.Linear(50 * 12 * 12, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.a_1(x))
        x = F.relu(self.a_one(F.relu(self.a_one(x))))
        x = self.pool(x)

        x = F.relu(self.b_1(x))
        x = F.relu(self.b_one(F.relu(self.b_one(x))))
        x = self.pool(x)

        x = F.relu(self.c_1(x))
        x = F.relu(self.c_one(F.relu(self.c_one(x))))
        x = self.pool(x)

        x = self.consoft(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#  在前面的卷层加入batch-normalztion

class one_fl_Net_con_nomal(nn.Module):
    def __init__(self, input_channels):
        super(one_fl_Net_con_nomal, self).__init__()

        self.a_1 = nn.Conv2d(12, 96, kernel_size=11, stride=4, padding=0)
        self.batch_1 = BatchNorm(96, num_dims=4)
        self.a_one = nn.Conv2d(96, 96, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.b_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.batch_2 = BatchNorm(256, num_dims=4)
        self.b_one = nn.Conv2d(256, 256, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.c_1 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.batch_3 = BatchNorm(384, num_dims=4)
        self.c_one = nn.Conv2d(384, 384, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.consoft = nn.Conv2d(384, 50, kernel_size=1, stride=1, padding=1)
        self.fc1 = nn.Linear(50 * 12 * 12, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.batch_1(self.a_1(x)))
        x = F.relu(self.a_one(F.relu(self.a_one(x))))
        x = self.pool(x)

        x = F.relu(self.batch_2(self.b_1(x)))
        x = F.relu(self.b_one(F.relu(self.b_one(x))))
        x = self.pool(x)

        x = F.relu(self.batch_3(self.c_1(x)))
        x = F.relu(self.c_one(F.relu(self.c_one(x))))
        x = self.pool(x)

        x = self.consoft(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#####33       卷积层和全连接层都加入了bn层。
class one_fl_Net_con_bn_fl_bn(nn.Module):
    def __init__(self, input_channels):
        super(one_fl_Net_con_bn_fl_bn, self).__init__()

        self.a_1 = nn.Conv2d(12, 96, kernel_size=11, stride=4, padding=0)
        self.batch_1 = BatchNorm(96, num_dims=4)
        self.a_one = nn.Conv2d(96, 96, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.b_1 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.batch_2 = BatchNorm(256, num_dims=4)
        self.b_one = nn.Conv2d(256, 256, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.c_1 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.batch_3 = BatchNorm(384, num_dims=4)
        self.c_one = nn.Conv2d(384, 384, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

        self.consoft = nn.Conv2d(384, 50, kernel_size=1, stride=1, padding=1)
        self.fc1 = nn.Linear(50 * 12 * 12, 50)
        self.batch_fl = BatchNorm(50, num_dims=2)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.batch_1(self.a_1(x)))
        x = F.relu(self.a_one(F.relu(self.a_one(x))))
        x = self.pool(x)

        x = F.relu(self.batch_2(self.b_1(x)))
        x = F.relu(self.b_one(F.relu(self.b_one(x))))
        x = self.pool(x)

        x = F.relu(self.batch_3(self.c_1(x)))
        x = F.relu(self.c_one(F.relu(self.c_one(x))))
        x = self.pool(x)

        x = self.consoft(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.batch_fl(self.fc1(x)))
        x = self.fc2(x)
        return x





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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

    accuracy = 100.0 * float(correct) / float(total)
    print('Accuracy of the network on the test set: %.3f%%' % (
        accuracy))

    return accuracy
