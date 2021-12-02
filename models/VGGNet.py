import torch
from torch.nn import Module,Sequential,Conv2d,ReLU,Linear,MaxPool2d,Flatten,BatchNorm2d,BatchNorm1d,Sigmoid,Softmax
import torch.nn as nn
import torch.nn.functional as F



class VGGNet(Module):
    def vgg_block(self,num_convs, input_channels, output_channels):
        net = [Conv2d(input_channels, output_channels, (3, 3), padding=1)]
        # net.append(BatchNorm2d(output_channels,track_running_stats=False))
        net.append(ReLU(True))

        for i in range(num_convs - 1):
            net.append(Conv2d(output_channels, output_channels, (3, 3), padding=1))
            # net.append(BatchNorm2d(output_channels,track_running_stats=False))
            net.append(ReLU(True))

        net.append(MaxPool2d((2, 2)))
        return Sequential(*net)

    def vgg_stack(self,num_convs,channels):
        net = []
        for n,c in zip(num_convs,channels):
            in_c = c[0]
            out_c = c[1]
            net.append(self.vgg_block(n,in_c,out_c))
        return Sequential(*net)

    def __init__(self) -> None:
        super().__init__()
        # self.feature = self.vgg_stack((2,2,3,3,3),((3,64),(64,128),(128,256),(256,512),(512,512)))
        self.feature = self.vgg_stack((1, 1, 2), ((3, 32), (32, 64), (64, 64)))
        self.conv1_1 = Conv2d(3,64,(3,3),padding=1)
        self.conv1_2 = Conv2d(64,64,(3,3),padding=1)
        self.pool1 = MaxPool2d((2,2))
        self.conv2_1 = Conv2d(64,128,(3,3),padding=1)
        self.conv2_2 = Conv2d(128,128,(3,3),padding=1)
        self.pool2 = MaxPool2d((2,2))
        self.conv3_1 = Conv2d(128,256,(3,3),padding=1)
        self.conv3_2 = Conv2d(256,256,(3,3),padding=1)
        self.conv3_3 = Conv2d(256,256,(3,3),padding=1)
        self.pool3 = MaxPool2d((2,2))
        self.conv4_1 = Conv2d(256,512,(3,3),padding=1)
        self.conv4_2 = Conv2d(512,512,(3,3),padding=1)
        self.conv4_3 = Conv2d(512,512,(3,3),padding=1)
        self.pool4 = MaxPool2d((2,2))

        # 别人的猫狗大战模型
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)  # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 第二个卷积层，输入通道数16，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认

        self.fc1 = nn.Linear(50 * 50 * 16, 128)  # 第一个全连层，线性连接，输入节点数50×50×16，输出节点数128
        self.fc2 = nn.Linear(128, 64)  # 第二个全连层，线性连接，输入节点数128，输出节点数64
        self.fc3 = nn.Linear(64, 1)

        # self.linear1 = Linear(4096,500)
        self.linear1 = Linear(40000, 1024)
        self.relu1 = ReLU(True)
        # self.linear2 = Linear(500,100)
        self.linear2 = Linear(1024, 100)
        self.relu2 = ReLU(True)
        self.linear3 = Linear(100,1)
        self.flatten = Flatten()
        self.fc = Sequential(
            Flatten(),
            # 这是给cat dog设计的
            Linear(40000, 500),
            # Linear(32768,500),
            # BatchNorm1d(4096),
            ReLU(True),
            Linear(500,100),
            # BatchNorm1d(1024),
            ReLU(True),
            Linear(100,2),
            # Sigmoid()
            Softmax()
            # 这是给cifar10设计的
            # Linear(512,100),
            # Linear(100,10)
        )

    def forward(self,x):
        # y = self.feature(x)
        # output = self.fc(y)
        # x = self.conv1_1(x)
        # # x = BatchNorm2d(64)(x)
        # x = self.relu1(x)
        # x = self.conv1_2(x)
        # # x = BatchNorm2d(64)(x)
        # x = self.relu1(x)
        # x = self.pool1(x)
        # x = self.conv2_1(x)
        # # x = BatchNorm2d(128)(x)
        # x = self.relu1(x)
        # x = self.conv2_2(x)
        # # x = BatchNorm2d(128)(x)
        # x = self.pool2(x)
        # x = self.conv3_1(x)
        # # x = BatchNorm2d(512)(x)
        # x = self.relu1(x)
        # x = self.conv3_2(x)
        # # x = BatchNorm2d(512)(x)
        # x = self.relu1(x)
        # x = self.conv3_3(x)
        # # x = BatchNorm2d(512)(x)
        # x = self.relu1(x)
        # x = self.pool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        # output = self.linear3(x)

        # 别人的网络模型
        # x = self.conv1(x)  # 第一次卷积
        # x = F.relu(x)  # 第一次卷积结果经过ReLU激活函数处理
        # x = F.max_pool2d(x, 2)  # 第一次池化，池化大小2×2，方式Max pooling
        #
        # x = self.conv2(x)  # 第二次卷积
        # x = F.relu(x)  # 第二次卷积结果经过ReLU激活函数处理
        # x = F.max_pool2d(x, 2)  # 第二次池化，池化大小2×2，方式Max pooling
        #
        # x = x.view(x.size()[0], -1)  # 由于全连层输入的是一维张量，因此需要对输入的[50×50×16]格式数据排列成[40000×1]形式
        # x = F.relu(self.fc1(x))  # 第一次全连，ReLU激活
        # x = F.relu(self.fc2(x))  # 第二次全连，ReLU激活
        # y = self.fc3(x)  # 第三次激活，ReLU激活
        y = self.feature(x)
        output = self.fc(y)

        # print('当前y的大小',y.shape)
        return output


