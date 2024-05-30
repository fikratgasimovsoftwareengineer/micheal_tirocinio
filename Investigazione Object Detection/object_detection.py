import sys

import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchinfo import summary
from timeit import default_timer as timer
from tqdm.auto import tqdm
import pandas as pd
import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
0. Let's prepare a transform for the images of our dataset with torchvision.transforms
"""
# Write a transform for image
data_transform = transforms.Compose([
    transforms.Resize(size=(320, 320)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()])

"""
1. Getting a dataset
The dataset we'll be using is OxfordIIITPet from torchvision.datasets
"""
# Setup training data
train_data = datasets.OxfordIIITPet(
    root='/home/michel/datasets',
    split='trainval',
    target_types='category',
    download=True,
    transform=data_transform,
    target_transform=None)

test_data = datasets.OxfordIIITPet(
    root='/home/michel/datasets',
    split='test',
    target_types='category',
    download=True,
    transform=data_transform,
    target_transform=None)


print(len(train_data), len(test_data))
class_names = train_data.classes # list of all class names
print(class_names)
class_to_idx = train_data.class_to_idx # indexes of all class names
print(class_to_idx)

"""
1.2 Visualizing our data

# Plot some images
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.title(f'{label} - {class_names[label]}')
    plt.axis(False)
plt.show()
"""

"""
2. Prepare DataLoader
Dataloader turns our dataset into a python iterable
More specifically we want to turn our data into batches (or mini-batches)
"""
print(train_data, test_data)
# Setup the batch size hyperparameter
BATCH_SIZE = 32
# Turn datasets into iterables (batches)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
print(train_dataloader, test_dataloader)
# Let's check out what we've created
print(f'Lenght of train_dataloader: {len(train_dataloader)} '
      f'batches of {BATCH_SIZE}...')
print(f'Lenght of test_dataloader: {len(test_dataloader)} '
      f'batches of {BATCH_SIZE}...')

"""
3. Setting up a pretrained model
When setting up a pretrained model we only modify the input and output layer,
this is called feature extraction, usefull for small amount of data
If you have large amount of data you will do also fine-tuning, you might
change the input layer but will change some of the last layers
"""
# How to setup a pretrained model
weight = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight=weight)

"""
3.1 Getting a summary of our model with torchinfo.summary() for feature extraction
"""
summary(model=model,
        input_size=(1, 3, 320, 320), # BS, C, H, W
        col_names=['input_size', 'output_size', 'num_params', 'trainable'],
        col_width=20,
        row_settings=['var_names'])

"""
3.4 Freezing the base model and changing the output layer to suit our needs
With a feature extractor model, typically you will 'freeze' the base layers of a
pretrained/foundation model and update the output layers to suit your own problem
"""
# Freeze all the base layers in EffNetB0
for param in model.roi_heads.parameters():
    # print(param)
    param.requires_grad = False

# Update the classifier head of our model to suit our problem
print(model.roi_heads)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True), # the same of the original
    # In our case we need to change only the out_features to the 3 foods
    nn.Linear(in_features=3234, # feature vector coming in
              out_features=len(class_names))) # how many classes do we have (3)
print(model.roi_heads)
