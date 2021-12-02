import torch
from torch.utils.tensorboard import SummaryWriter

def test(model,device,test_dataloader,dataset_str):
    model.eval()
    test_loss = 0
    total_accurcy = 0
    # writer = SummaryWriter('../log/'+dataset_str+'/test')
    len_test = len(test_dataloader.dataset)
    with torch.no_grad():
        for imgs,targets in test_dataloader:
            imgs,targets = imgs.to(device),targets.to(device)
            outputs = model(imgs)
            test_loss += torch.nn.CrossEntropyLoss()(outputs,targets).cuda().item()
            # cat_dog
            # test_loss += torch.nn.L1Loss()(outputs,targets).cuda().item()
            total_accurcy += (outputs.argmax(1) == targets).sum()
            # test_loss = torch.nn.MSELoss()(outputs, targets).cuda()
            # print('outputs: ',outputs)
            # total_accurcy += ((outputs+0.5).int() == targets).sum()

    print('整体测试集损失：{}'.format(test_loss))
    print('整体测试机准确率：{}'.format(total_accurcy.item()/len_test))
