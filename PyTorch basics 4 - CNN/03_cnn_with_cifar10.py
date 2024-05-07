import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

transform = transforms.ToTensor()
train_data = datasets.CIFAR10(root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                            train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                           train=False, download=True, transform=transform)
print(train_data, test_data)
torch.manual_seed(101)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
class_names = ['plane', '  car', ' bird', '  cat', ' deer',
               '  dog', ' frog', 'horse', ' ship', 'truck']
for images, labels in train_loader:
    break
# I numeri che spuntano sono l'indice delle figure, ossia 3
# è un gatto, 5 un cane ecc.
print('Label: ', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))
im = make_grid(images, nrow=5)
plt.figure(figsize=(10, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.show()
