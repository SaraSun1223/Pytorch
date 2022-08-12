from logging import root
from tkinter import image_names
from unicodedata import category
from torch.utils.data import Dataset
from PIL import Image
import os

#上面是引入各种包

#新建对象 继承Dataset类
class Sara_dataset(Dataset):
    def __init__(self, root):
        super(Sara_dataset, self).__init__()
        self.data = []
        self.targets = []
        for categorys in os.listdir(root):
            #print一下categories结果应该是dog和cat两类
            # print(categorys)
            category = os.path.join(root, categorys)
            # print(category)
            for images in os.listdir(category):
                image = os.path.join(category, images)
                self.data.append(image)
                self.targets.append(categorys)
                #这里是把所有的图片名称都print
                # print(images)


    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img)
        return img, target
    
    def _len_(self):
        return len(self.data)

#主函数

if __name__=='__main__':
    print("ok")
    root = "dataset\catvsdog"

    sara_dataset = Sara_dataset(root)

#print第一张图片的PIL数据
    img, target = sara_dataset.__getitem__(1)
    print(img, target)
