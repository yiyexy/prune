from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision

transorm = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.491,0.482,0.446),(0.247,0.243,0.261))
])

train_data = CIFAR10('../datasets',train=True,transform=transorm,download=True)
test_data = CIFAR10('../datasets',train=False,transform=transorm,download=True)

batch_size = 64
train_dataloader = DataLoader(train_data,batch_size,shuffle=False)
test_dataloader = DataLoader(test_data,batch_size,False)

img,target = train_data[0]
print(img.shape)

