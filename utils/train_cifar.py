import torch
from torch.nn import CrossEntropyLoss
from DataLoad.CIFAR10 import train_dataloader
from models.VGG_CIFAR import VGG_CIFAR
from models.pretrain_VggCifar import pretrain_VggCifar
from utils.test_cifar import test_cifar

def train_cifar():
    total_train_step = 0
    model = pretrain_VggCifar().cuda()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,nesterov=True)
    epochs = 160
    for i in range(epochs):
        if i == int(epochs*0.5):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-2
        elif i == int(epochs*0.75):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
        for batch_index,data in enumerate(train_dataloader):
            imgs,targets = data
            imgs,targets = imgs.cuda(),targets.cuda()
            outputs = model(imgs)
            optimizer.zero_grad()
            train_loss = CrossEntropyLoss()(outputs,targets).cuda()
            train_loss.backward()
            optimizer.step()
            total_train_step += 1
            if batch_index % 100 == 0:
                print('当前为第{}个epoch中的第{}个batch，此时损失为:{}'.format(i+1,batch_index,train_loss))
        if i % 10 == 0:
            test_cifar(model)



train_cifar()


