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



class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Visto che ora non passeremo un'immagine monocolore
        # il canale d'input deve essere più alto, in questo caso
        # per rispettare RGB metteremo 3
        self.conv1 = nn.Conv2d(3, 6, 5, 1) # (1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(6*6*16, 120) # 4*4*16
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 6*6*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(101)
model = ConvolutionalNetwork()
print(model)
for param in model.parameters():
    # Circa 80k parametri
    print(param.numel())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
start_time = time.time()
# Variabili tracker, non necessarie ma comode
epochs = 15
train_losses = []
test_losses = []
train_correct = []
test_correct = []
# For loop per le epoch
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    # Eseguiamo il training delle batch
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        # Visto che è una CNN non serve il flatten perchè accetta dati 2D
        # Istanziamo il modello
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        # Contiamo il numero delle previsioni corrette
        predicted = torch.max(y_pred.data, 1)[1]
        # Si fa la somma tra i true (1) e i false (0)
        batch_corr = (predicted==y_train).sum()
        trn_corr += batch_corr
        # Aggiorniamo i parametri
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Stampiamo i risultati a interim
        if b&1000 == 0:
            print(f'Epoch: {i+1} '
                  f'Batch: {b:4} [{10*b:6}/50000] '
                  f'Loss: {loss.item():10.8f} '
                  f'Accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
    train_losses.append(loss.item())
    train_correct.append(trn_corr)
    # Eseguiamo i testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Istanziamo il modello
            y_val = model(X_test)
            # Contiamo il numero delle previsioni corrette
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted==y_test).sum()
    loss = criterion(y_val, y_test)
    test_losses.append(loss.item())
    test_correct.append(tst_corr)
current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')

torch.save(model.state_dict(), 'my_CIFAR_model.pt')
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.title('Loss at epoch')
plt.legend()
plt.show()
plt.plot([t/500 for t in train_correct], label='train accuracy')
plt.plot([t/100 for t in test_correct], label='test accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()
print(test_correct)
num_c = test_correct[-1].item()
print(num_c * 100 / 10000)
# Vediamo la heatmap
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted==y_test).sum()
arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize=(9, 6))
sn.heatmap(df_cm, annot=True, fmt='d', cmap='BuGn')
plt.xlabel('prediction')
plt.ylabel('label (ground truth)')
plt.show()
