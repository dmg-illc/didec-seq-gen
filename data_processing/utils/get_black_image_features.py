import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.image as pim

# black image features for <sos> <eos> pads

#bottom up features come from this Faster R-CNN with ResNet-101
resnet101 = models.resnet101(pretrained=True)
print(resnet101)
modules = list(resnet101.children())[:-1] #
resnet101 = nn.Sequential(*modules)
resnet101.eval() # no fine-tuning

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#https://pytorch.org/docs/stable/torchvision/models.html
#All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
#where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using
#mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

# some random img from the dataset
img_id = '61513'
grey_img_file = '../data/images_bordered/' + img_id + '.bmp'
img = pim.imread(grey_img_file)

image = Image.open(grey_img_file).convert('RGB')

width = image.size[0]
height = image.size[1]

print(width, height)

all_black_mask = np.zeros((height, width))  # 1050, 1680 with borders reverse in np

new_img = img.copy()

# mask the image given the heatmap
new_img[:, :, 0] = new_img[:, :, 0] * all_black_mask
new_img[:, :, 1] = new_img[:, :, 1] * all_black_mask
new_img[:, :, 2] = new_img[:, :, 2] * all_black_mask

image = Image.fromarray(new_img, 'RGB')
image = image.crop((206, 50, 206 + 1267, 50 + 950))

image.save('black.png')

image = test_transform(image)
# print(image.shape) #(3,224,224)

input = Variable(image).view(1, image.shape[0], image.shape[1], image.shape[2])
output = resnet101(input)
output = output[0].squeeze(1).squeeze(1).data
black_img_features = output.numpy().tolist()

print(output)

with open('black_img_features.json', 'w') as file:
    json.dump(black_img_features, file)