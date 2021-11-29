import torch
from torch.nn import Module,Sequential,Conv2d,ReLU,Linear,MaxPool2d,Flatten,BatchNorm2d,BatchNorm1d
import torch.nn as nn



class VGGNet(Module):
    def vgg_block(self,num_convs, input_channels, output_channels):
        net = [Conv2d(input_channels, output_channels, (3, 3), padding=1)]
        net.append(BatchNorm2d(output_channels,track_running_stats=False))
        net.append(ReLU(True))

        for i in range(num_convs - 1):
            net.append(Conv2d(output_channels, output_channels, (3, 3), padding=1))
            net.append(BatchNorm2d(output_channels,track_running_stats=False))
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
        self.feature = self.vgg_stack((3,3,3,3,3),((3,64),(64,128),(128,256),(256,512),(512,512)))
        self.fc = Sequential(
            Flatten(),
            # 这是给cat dog设计的
            Linear(32768,4096),
            BatchNorm1d(4096),
            ReLU(True),
            Linear(4096,1024),
            BatchNorm1d(1024),
            ReLU(True),
            Linear(1024,100),
            BatchNorm1d(100),
            ReLU(True),
            Linear(100,2)
            # 这是给cifar10设计的
            # Linear(512,100),
            # Linear(100,10)
        )

    def forward(self,x):
        y = self.feature(x)
        # print('当前y的大小',y.shape)
        return self.fc(y)


