import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch




# csv文件中数据集格式应该如下：
"""
a.jpg 0
b.jpg 1
c.jpg 2
...

p.jpg 1
"""



# 加载自己的数据集
class CustomImageDataset(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None,target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label


# labels_map = {
#     1:"cat",
#     2:"dog",
# }
# img_dir = "D:/Dataset/DogVsCat/train"
# train_data = CustomImageDataset(annotations_file="./image/train.csv",img_dir=img_dir)
# images,labels = train_data[0]
# figure = plt.figure()
# plt.title(labels_map[labels])
# plt.axis("off")
# # 图像通道转换: [3, 374, 500]->[374, 500, 3],便于显示
# print(images.shape)
# images = images.permute(1,2,0)
# print(images.shape)
# plt.imshow(images)
# plt.show()


