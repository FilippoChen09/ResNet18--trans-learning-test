import torchvision
import torch
import matplotlib.pyplot as plt
import cv2

# 设置超参数
filepath = 'milan girl.jpg'
image_height = 523
image_width = 527

model = torchvision.models.resnet18(pretrained=True)
# 实例化resnet18对象
# 设置参数pretrained=True则加载预训练参数，否则只有网络结构

totensor = torchvision.transforms.ToTensor()

# print(model)
# 可以打印出网络的结构

image = plt.imread(filepath)
input_image = totensor((cv2.resize(image,(image_width,image_height))))
# cv2.resize(img,(width,height))此方法中图片宽在前，高在后
input_image = input_image.reshape(1,3,523,527)
# 读取、resize二进制文件，并转换为torch的张量

output = model(input_image)
# 将待分类图片导入模型
prediction = torch.max(output,1)
print(prediction)