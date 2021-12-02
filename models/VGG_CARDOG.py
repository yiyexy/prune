import torch
from torch.nn import Conv2d,MaxPool2d,Linear,ReLU,Module,Sequential,Flatten,Softmax,BatchNorm2d,BatchNorm1d

class VGG_CARDOG(Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = self.vgg_stack((3,3,3,3),((3,64),(64,128),(128,256),(256,512)))
        self.fc = Sequential(
            Flatten(),
            Linear(73728,1024),
            Linear(1024,128),
            Linear(128,2),
            Softmax()
        )

    def vgg_block(self,num_convs,in_channels,out_channels):
        net = [Conv2d(in_channels,out_channels,3,padding=1)]
        net.append(BatchNorm2d(out_channels))
        net.append(ReLU(True))
        for i in range(0,num_convs-1):
            net.append(Conv2d(out_channels,out_channels,3,padding=1))
            net.append(BatchNorm2d(out_channels))
            net.append(ReLU(True))
        net.append(MaxPool2d(2))
        return Sequential(*net)

    def vgg_stack(self,num_convs,channels):
        net = []
        for n,c in zip(num_convs,channels):
            in_channels = c[0]
            out_channels = c[1]
            net.append(self.vgg_block(n,in_channels,out_channels))
        return Sequential(*net)

    def forward(self,x):
        x = self.feature(x)
        return self.fc(x)
