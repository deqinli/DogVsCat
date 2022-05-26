import torch.utils.data
import numpy as np
import os, random, glob
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



"""
说明：原始数据可在kaggle官网下载
    数据格式如下：
    ├── train
    │   ├── cat
    │   └── dog
    └── validation
        ├── cat
        └── dog
"""

# 定义数据集读取类
class DogCatDataSet(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform

        dog_dir = os.path.join(img_dir, "dog")
        cat_dir = os.path.join(img_dir, "cat")
        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(dog_dir, "*.jpg")))
        imgsLib.extend(glob.glob(os.path.join(cat_dir, "*.jpg")))
        random.shuffle(imgsLib)  # 打乱数据集
        self.imgsLib = imgsLib

    # 作为迭代器必须要有的
    def __getitem__(self, index):
        img_path = self.imgsLib[index]

        label = 1 if 'dog' in img_path.split('/')[-1] else 0 #狗的label设为1，猫的设为0

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgsLib)


# """ 读取并测试显示数据 """

# if __name__ == "__main__":
#
#     CLASSES = {0: "cat", 1: "dog"}
#     img_dir = "./image/"
#
#     data_transform = transforms.Compose([
#         transforms.Resize(256),  # resize到256
#         transforms.CenterCrop(224),  # crop到224
#         transforms.ToTensor(),
#     ])
#
#     dataSet = DogCatDataSet(img_dir=img_dir, transform=data_transform)
#     dataLoader = torch.utils.data.DataLoader(dataSet, batch_size=8, shuffle=True, num_workers=4)
#     image_batch, label_batch = iter(dataLoader).next()
#     for i in range(image_batch.data.shape[0]):
#         label = np.array(label_batch.data[i])          ## tensor ==> numpy
#         # print(label)
#         img = np.array(image_batch.data[i]*255, np.int32)
#         print(CLASSES[int(label)])
#         plt.imshow(np.transpose(img, [1, 2, 0]))
#         plt.show()
