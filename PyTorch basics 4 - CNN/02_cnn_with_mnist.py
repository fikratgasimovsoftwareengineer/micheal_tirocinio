import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='.../Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                            train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='.../Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data',
                           train=False, download=True, transform=transform)
print(train_data, test_data)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)
# Le immagini nel CNN hanno 4 dimensioni
# Il prino indica il colore, il secondo i filtri,
# il terzo la grandezza dei filtri e il quarto lo stride
# Il numero di filtri è arbitrario
conv1 = nn.Conv2d(1, 6, 3, 1)
conv2 = nn.Conv2d(6, 16, 3, 1)
