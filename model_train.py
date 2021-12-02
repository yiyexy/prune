import torch
from models.VGGNet import VGGNet
from utils.train import train
from utils.test import test
# from DataLoad.CIFAR10 import train_dataloader,test_dataloader
from DataLoad.read_data import train_dataloader,test_dataloader

lr = 1e-4
weight_decay = 1e-6
momentum = 0.5

device = torch.device('cuda')
model = VGGNet().cuda()
print(model)
# 神经网络中的参数默认是随机初始化的，这样就会导致每次训练的初始化参数不一致，这样设置可以保证每次初始化训练的参数一致
torch.cuda.manual_seed(12435)
# torch.nn.init.kaiming_uniform_(model.parameters(),mode='fan_in', nonlinearity='relu')


# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
# optimizer = torch.optim.SGD(model.parameters(),lr)
# for data in train_dataloader:
#     imgs,labels = data
#     print(imgs[1].shape)
epochs = 160

# train(model,device,train_dataloader,optimizer,epochs,'CIFAR10')
# test(model,device,test_dataloader,'CIFAR10')
train(model,device,train_dataloader,optimizer,epochs,'cat_dog')
test(model,device,test_dataloader,'cat_dog')


