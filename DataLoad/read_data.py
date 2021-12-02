from torch.utils.data import DataLoader,Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

from torchvision import transforms
from torchvision.utils import make_grid

class MyData(Dataset):
    def __getitem__(self, index) :
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(img_path)
        # img = img/255.0
        img = self.transform(img)
        return img,label

    def __len__(self):
        assert len(self.img_list) == len(self.label_list)
        return len(self.img_list)

    def __init__(self,root_dir,train=True,transform=None) :
        self.root_dir = root_dir
        if train:
            img_cat_path = os.path.join(root_dir,"training_set/training_set/cats")
            img_cat_path_str = root_dir+'/training_set/training_set/cats/'
            img_dog_path = os.path.join(root_dir,'training_set/training_set/dogs')
            img_dog_path_str = root_dir+'/training_set/training_set/dogs/'
        else :
            img_cat_path = os.path.join(root_dir,'test_set/test_set/cats')
            img_cat_path_str = root_dir+'/test_set/test_set/cats/'
            img_dog_path = os.path.join(root_dir,'test_set/test_set/dogs')
            img_dog_path_str = root_dir+'/test_set/test_set/dogs/'
        img_cat_list = os.listdir(img_cat_path)
        img_dog_list = os.listdir(img_dog_path)
        len_cat = len(img_cat_list)
        len_dog = len(img_dog_list)
        for i in range(0,len_cat):
            img_cat_list[i] = img_cat_path_str + img_cat_list[i]
        for i in range(0,len_dog):
            img_dog_list[i] = img_dog_path_str + img_dog_list[i]
        self.transform = transform
        cat_label = [0 for i in range(0,len_cat)]
        dog_label = [1 for i in range(0,len_dog)]

        self.img_list = img_cat_list+img_dog_list
        self.label_list = cat_label+dog_label




root_dir = '../datasets/cat_dog'
transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
train_dataset = MyData(root_dir,True,transforms)
test_dataset = MyData(root_dir,False,transforms)

train_dataloader = DataLoader(train_dataset,16,True,num_workers=2)
test_dataloader = DataLoader(test_dataset,16,True,num_workers=2)




