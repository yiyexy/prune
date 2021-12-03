import torch
from DataLoad.read_data import train_dataloader
from models.VGG_CARDOG import VGG_CARDOG
from models.pretrain_VggCat import pretrain_VggCat
from utils.test_cardog import test_cardog

def train_cardog():
    model = pretrain_VggCat().cuda()
    lr = 0.01
    model.train()
    optimizer = torch.optim.SGD(model.parameters(),lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr,momentum=0.9,nesterov=True)
    epochs = 20
    for i in range(0,epochs):
        for batch_index,data in enumerate(train_dataloader):
            imgs,targets = data
            imgs,targets = imgs.cuda(),targets.cuda()
            outputs = model(imgs)
            optimizer.zero_grad()
            train_loss = torch.nn.CrossEntropyLoss()(outputs,targets).cuda()
            train_loss.backward()
            optimizer.step()

            if batch_index % 100 == 0:
                print('当前为第{}个epoch中的第{}个batch，当前batch的损失为：{}'.format(i+1,batch_index+1,train_loss))

        if i % 10 == 0:
            test_cardog(model)
            if i == 10:
                torch.save(model.state_dict(),'Vgg_catdog.pth')
                print('模型以保存')


train_cardog()