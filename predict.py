import torch
from AlexNet import AlexNet
import os
from PIL import Image
from torchvision import transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt



label_map = {
    0:"cat",
    1:"dog",
}

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load("./model/AlexNet.pth")
model.eval()
model.to(device)
img_dir = "./image/test/"
img_files = os.listdir(img_dir)
for idx in range(len(img_files)):
    img_path = img_dir + img_files[idx]
    print(img_path)
    img = Image.open(img_path)
    img_data = data_transform(img)
    img_data = torch.unsqueeze(img_data,dim=0)

    predict = model(img_data.to(device))

    # out = F.softmax(predict,dim=0)
    predict = torch.argmax(predict).cpu().numpy()
    print(predict)

    # print(out)
    figure = plt.figure()

    plt.title(label_map[int(predict)])
    plt.imshow(img)
    plt.show()