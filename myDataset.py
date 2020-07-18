import os
from random import random
from PIL import Image, ImageFilter
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


class MyDataset(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        self.test = test
        # imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs = []

        dataset = ImageFolder(root)
        
        self.data_classes = dataset.classes
        imgs = [dataset.imgs[i][0] for i in range(len(dataset.imgs))]
        labels = [dataset.imgs[i][1] for i in range(len(dataset.imgs))]

        self.imgs_num = len(imgs)

    
        self.imgs = imgs
        self.labels = labels



        


        if transforms is not None:
            self.transforms = transforms
            

    def id_to_class(self, index):
        return self.data_classes(index)

    

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path = self.imgs[index]
        label = self.labels[index]
        
        img = Image.open(img_path)
        aug_img = self.transforms(img)
        data = aug_img
        return data, label

    def __len__(self):
        return self.imgs_num