import torch
from torch.utils.tensorboard import SummaryWriter
from utils.test import test
from DataLoad.CIFAR10 import test_dataloader

def train(model,device,train_dataloader,optimizer,epochs,dataset_str,log_interval = 100):
    total_train_step = 0
    model.train()
    writer = SummaryWriter('../log/'+dataset_str+'/train')
    for i in range(epochs):
        if i == int(epochs*0.5):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-2
        elif i == int(epochs*0.75):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
        for batch_index,data in enumerate(train_dataloader):
            imgs,targets = data
            imgs,targets = imgs.to(device),targets.to(device)
            outputs = model(imgs)
            optimizer.zero_grad()
            # print('outpurts.shape',outputs.shape)
            # targets = targets.reshape(-1,1)
            # batchsize = targets.shape[0]
            # print('targets.shape',targets.shape)
            # targets_one_hot = torch.zeros(batchsize,outputs.shape[1]).cuda().scatter(1,targets,1)
            # targets_one_hot = targets_one_hot.type(torch.LongTensor).cuda()
            # print('targets_one_hot.shape',targets_one_hot.dtype)
            train_loss = torch.nn.CrossEntropyLoss()(outputs,targets).cuda()
            train_loss.backward()
            optimizer.step()
            total_train_step += 1

            if batch_index % log_interval == 0:
                print("epoch次数为：{}，在此epoch中的batch数为：{}，Loss：{}".format(i+1,batch_index,train_loss))
                writer.add_scalar('train_loss',train_loss.item(),total_train_step)
        # for param_group in optimizer.param_groups:
        #     print('当前的学习率为：',param_group['lr'])
        if  i % 10 == 0:
            print('第{}个epoch时，测试集的结果'.format(i))
            test(model,device,test_dataloader,dataset_str)

    writer.close()