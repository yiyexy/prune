import torch
from torch.nn import Sequential,Linear,Conv2d,Module

class LeNet(Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Linear(28*28,)
        )

    def forward(self,x):
        output = self.model(x)
