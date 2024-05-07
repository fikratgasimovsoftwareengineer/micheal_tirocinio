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
# Il primo indica il colore, il secondo i filtri (che sono arbitrari),
# il terzo la grandezza dei filtri (3x3) e il quarto lo stride, ossia
# di quanti pixel il filtro si deve spostare
# Il numero di filtri è arbitrario
conv1 = nn.Conv2d(1, 6, 3, 1)
# Visto che l'immagine nel primo layer passa per 6 filtri, poi per
# il pooling e infine al secondo layer, per questo il secondo layer
# ha il canale d'entra con 6, la grandezza dei filtri e lo stride
# devono rimanere uguali
conv2 = nn.Conv2d(6, 16, 3, 1)
for i, (X_train, y_train) in enumerate(train_data):
    print(X_train.shape) # -> 4D batch (batch di un'immagine)
    break
x = X_train.view(1, 1, 28, 28) # da -> 1, 28, 28
x = F.relu(conv1(x))
# x.shape -> 1, 6, 26, 26
# Ricorda che nel cnn le immagini perdono infomazioni sui bordi
# per questo da 28 è diventato 26, non è un gran problema visto
# che le immagini di questo dataset sono molto centrali e vuote
# ai bordi, in altri casi potrebbe essere importante, per questo
# bisognerebbe passare il parametro padding al conv
x = F.max_pool2d(x, 2, 2)
# Visto che abbiamo un kernel e uno stride di 2 il risultato
# è di dimezzare altezza e larghezza
# x.shape -> 1, 6, 13, 13
x = F.relu(conv2(x))
# x.shape -> 1, 16, 11, 11
x = F.max_pool2d(x, 2, 2)
# In questo caso farà un approssimazione per difetto
# x.shape -> 1, 16, 5, 5
# In pratica si perdono pixel, si dimezza e si ripete, quindi da 28 a 5
# Usiamo -1 per tenere invariata la prima dimensione
x.view(-1, 16*5*5) # 1, 400



# Creiamo una classe
class ConvolutionalNetwork(nn.module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        # Con il continuare degli strati riduci un po' i neuroni
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)
