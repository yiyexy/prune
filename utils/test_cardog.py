from DataLoad.read_data import test_dataloader
import torch

def test_cardog(model):
    model.eval()
    test_loss = 0
    len_data = len(test_dataloader.dataset)
    accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs,targets = imgs.cuda(),targets.cuda()
            outputs = model(imgs)
            test_loss += torch.nn.CrossEntropyLoss()(outputs,targets).cuda().item()
            accuracy += (outputs.argmax(1)==targets).sum()

        print('当前整个测试集的损失为：',test_loss)
        print('当前整个测试集的准确率为：',accuracy.item()/len_data)