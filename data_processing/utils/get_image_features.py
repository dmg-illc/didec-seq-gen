import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
import os
from PIL import Image

# image features for DS and FV masked images are generated

'''
#removes FC, avgpool2d is there
'''


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

def vectorize_images(pretrained_model, img_folder, task):

    features = dict()  # img_id -> features

    count = 0

    for subdirs, dirs, files in os.walk(img_folder):
        for f in files:
            if 'png' in f:

                file_path = os.path.join(img_folder, f)
                img_id = f.split('.')[0].split('_')[1]
                count += 1
                print(task, count, img_id)

                image = Image.open(file_path).convert('RGB')

                image = test_transform(image)
                # print(image.shape) #(3,224,224)

                input = Variable(image).view(1, image.shape[0], image.shape[1], image.shape[2])
                output = pretrained_model(input)
                output = output[0].squeeze(1).squeeze(1).data
                # print(output.shape) #torch.Size([2048])

                features[img_id] = output.numpy().tolist()

    print(count)

    file_name = '../data/features_sm_' + task + '.json'

    with open(file_name, 'w') as file:  #
        json.dump(features, file)


tasks = ['free_view', 'description']

for task in tasks:

    print(task)

    masked_images_path = '../data/masked_images/' + task

    vectorize_images(pretrained_model=resnet101, img_folder=masked_images_path, task=task)