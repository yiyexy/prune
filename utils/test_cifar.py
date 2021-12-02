import torch
from torch.utils.tensorboard import SummaryWriter
from DataLoad.CIFAR10 import test_dataloader
from models.VGG_CIFAR import VGG_CIFAR

def test_cifar(model):
    model.eval()
    test_loss = 0
    total_accurcy = 0
    # writer = SummaryWriter('../log/'+dataset_str+'/test')
    len_test = len(test_dataloader.dataset)
    with torch.no_grad():
        for imgs,targets in test_dataloader:
            imgs,targets = imgs.cuda(),targets.cuda()
            outputs = model(imgs)
            test_loss += torch.nn.CrossEntropyLoss()(outputs,targets).cuda().item()
            total_accurcy += (outputs.argmax(1) == targets).sum()

    print('整体测试集损失：{}'.format(test_loss))
    print('整体测试集准确率：{}'.format(total_accurcy.item()/len_test))
