from torchvision.models.vgg import VGG
from torch.nn import Sequential,Linear,Dropout,ReLU,AvgPool2d,Softmax,Module,Flatten
from torchvision.models import vgg16
import torch
from DataLoad.read_data import train_dataloader

# def vgg16_pretrained():
#     model = Sequential(
#         vgg16(pretrained=True,),
#         AvgPool2d(),
#         Linear(1000),
#         ReLU(True),
#         Dropout(0.4),
#         Linear(100),
#         ReLU(True)
#
#     )

#
# model = vgg16(pretrained=True)
# for data in train_dataloader:
#     imgs,targets = data
#     imgs = imgs.cuda()
#     model = model.cuda()
#     outputs = model.features(imgs)
#     outputs2 = model.avgpool(outputs)
#     print(model(imgs))
# def new_model(model):
#     model.classifier = Sequential(
#         AvgPool2d(7),
#         Linear(512,100),
#         ReLU(True),
#         Dropout(0.4),
#         Linear(100,64),
#         ReLU(True),
#         Linear(64,2),
#         Softmax()
#     )
#     return model
#
# print(new_model(model))

class pretrain_VggCat(Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = vgg16(pretrained=True)
        # self.model.avgpool = Sequential()
        self.model.classifier = Sequential(
            AvgPool2d(7,7),
            Flatten(),
            Linear(512, 100),
            ReLU(True),
            Dropout(0.4),
            Linear(100, 64),
            ReLU(True),
            Linear(64, 2),
            Softmax()
        )

    def forward(self,x):
        # print(self.model)
        # outputs = self.model(x)
        outputs = self.model.features(x)
        outputs1 = self.model.avgpool(outputs)
        outputs2 = self.model.classifier(outputs1)
        return outputs2
