"""
We've used some datasets with pytorch before
But how do you get youe own data into pytorch?
One of the ways to do so is via: custom datasets

Domain libraries
Depending on what you're working one you'll want to look into each of the pytorch
domain libraries for existing data loading functions and customizable data loading functions

Resources:
* Book version: https://www.learnpytorch.io/04_pytorch_custom_datasets/
* Ground truth notebook: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/04_pytorch_custom_datasets.ipynb



0. Importing Pytorch and setting up device-agnostic code
"""
import torch
from torch import nn
# 1.
import requests
import zipfile
from pathlib import Path
# 2.
import os
# 2.1
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# 3.
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup device-agnostic code
device = "cuda" if torch.cuda.is_available() else 'cpu'

"""
1. Get data
Our dataset is a subset of the Food 101 dataset
Food101 starts 101 different classes of food
Our dataset starts with 3 classes of food and only 10% of the images 
(75 training - 25 testing)
Why?
When starting our ML projects it's important to try things on a small scale 
and then increase the scale when necessary.
The whole point is to speed up how fast you can experiment
"""
# Setup path to a data folder
data_path = Path('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data')
image_path = data_path / 'pizza_steak_sushi'

# If the image folder doesn't exist, download it and prepare it...
if image_path.is_dir():
    print(f'{image_path} directory already exists... skipping download')
else:
    print(f"{image_path} doesn't exist, downloading now...")
    image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak and sushi data
with open(data_path / 'pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get('https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip')
    print('Downloading pizza, steak, sushi data...')
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
    print('Unzipping pizza, steak and sushi data...')
    zip_ref.extractall(image_path)
print(image_path)

"""
2. Becoming one with the data (data preparation and data exploration)
"""
def walk_through_dir(dir_path):
    """Walks through dir_path returning its contents"""
    for dirpath, dirnames, filesnames in os.walk(dir_path):
        print(f'There are {len(dirnames)} directories and {len(filesnames)} images in "{dirpath}"')

walk_through_dir(image_path)

# Setup train and testing paths
train_dir = image_path / 'train'
test_dir = image_path / 'test'
print(f'{train_dir}\n{test_dir}')

"""
2.1 Visualizing an image
Let's write some code to:
1. Get all of the image paths
2. Pick a random image path using Python's random.choice()
3. Get the image class name using pathlib.Path.parent.stem
4. Since we're working with images, let's open the image with Python's PIL
5. We'll then show the image and print metadata
"""
# Set seed
# random.seed(42)

# 1. Get all image paths
image_path_list = list(image_path.glob('*/*/*.jpg'))

# 2. Get a random image path
random_image_path = random.choice(image_path_list)
print(random_image_path)

# 3. Get image classe from path (the image class
# is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
print(image_class)

# 4. Open image
img = Image.open(random_image_path)

# 5. Print metadata
print(f'Random image path: {random_image_path}')
print(f'Image class: {image_class}')
print(f'Image height: {img.height}')
print(f'Image width: {img.width}')
# Use .show for see the image
# img.show()

# Turn the image in an array
img_as_array = np.asarray(img)

# Plot the image with matplotlib
plt.figure(figsize=(10, 7))
plt.imshow(img_as_array) # NOT NECESSARY as an array
plt.title(f'Image class: {image_class} | '
          f'Image shape: {img_as_array.shape} -> [height, width, color channels]')
plt.axis(False)
# plt.show()

"""
3. Transforming data
Before we can use our image data with Pytorch:
1. Turn your target data into tensors (in our case, numerical representation of our
image)
2. Turn it into a torch.utils.data.Dataset and subsequently a torch.utils.data.Dataloader,
we'll call these Dataset and DataLoader



3.1 Transforming data with torchvision.transforms
"""
# Write a transform for image, you can use compose or sequential
data_transform = transforms.Compose([
    # Resize our images to 64x64, only because our preious model were set for this size
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal (data augmentation)
    transforms.RandomHorizontalFlip(p=0.5), # we set a chance 50%
    # Turn the image in a torch.Tensor
    transforms.ToTensor()
])
tensor_img = data_transform(img) # as an array gives error
print(tensor_img, tensor_img.shape, tensor_img.dtype)

