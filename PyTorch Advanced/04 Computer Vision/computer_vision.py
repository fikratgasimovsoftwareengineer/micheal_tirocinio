"""
PyTorch computer vision

See reference notebook: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb
See reference online book: https://www.learnpytorch.io/03_pytorch_computer_vision/
"""
import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
print(torch.__version__)
print(torch.__version__)

"""
0. Computer vision libraries in PyTorch
* torchvision - base domain library for Pytorch computer vision
* torchvision.datasets - gets datasets and data loading functions for computer vision
* torchvision.models - get pretrained computer vision models that you can leverage for your own problems
* torchvision.transforms - functions for manipulating your vision data (images) to be suitable for use with an ML model
* torch.utils.data.Dataset - base dataset class for pytorch
* torch.utils.data.Dataloader - creates a python iterable over a dataset



1. Getting a dataset
The dataset we'll be using is FashionMNIST from torchvision.datasets
"""
# Setup training data
train_data = datasets.FashionMNIST(
    root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data', # where to download data to?
    train=True, # do we want the training set?
    download=True, # do we want to download yes/no?
    transform=torchvision.transforms.ToTensor(), # how do we want to transform the data?
    target_transform=None) # how do we want to transform the labels/targets?

test_data = datasets.FashionMNIST(
root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
    train=False, # do we want the test set?
    download=True,
    transform=ToTensor(),
    target_transform=None)

print(len(train_data), len(test_data))
# See the first training example
image, label = train_data[0]
print(image, label)
class_names = train_data.classes
print(class_names)
class_to_idx = train_data.class_to_idx
print(class_to_idx)
print(train_data.targets)

"""
1.1 Check input and output shapes of data
"""
print(f'\nImage shape: {image.shape} -> [color channels, height, widht]')
print(f'Image label: {class_names[label]}')

"""
1.2 Visualizing our data
"""
# plt.imshow(image.squeeze())
# plt.title(f'{label} - {class_names[label]}')
# plt.show()

# plt.imshow(image.squeeze(), cmap='gray')
# plt.title(f'{label} - {class_names[label]}')
# plt.axis(False)
# plt.show()

# Plot more images
# torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    # fig.add_subplot(rows, cols, i)
    # plt.imshow(img.squeeze(), cmap='gray')
    # plt.title(f'{label} - {class_names[label]}')
    # plt.axis(False)
# plt.show()
#  Now... we need a model with linearity or also with non-linearity?

"""
2. Prepare DataLoader
Right now, our data is in form of Pytorch datasets.
Dataloader turns our dataset into a python iterable
More specifically we want to turn our data into batches (or mini-batches)
Why?
1. It's more computationally efficient, your computer could not store in memory
all 60000 images in one hit, so we break it down (in this case batch size of 32 at time)
2. It gives our neural network more chances to update its gradients per epoch
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
# Now lets checkout what's inside the training dataloader
train_features_batch, train_labels_batch = next(iter(train_dataloader))
test_features_batch, test_labels_batch = next(iter(test_dataloader))
# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(class_names[label])
plt.axis(False)
print(f'Image size: {img.shape}')
print(f'Label: {label}, label size: {label.shape}')
plt.show()
