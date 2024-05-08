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
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Dataset cani e gatti
# https://drive.google.com/file/d/1fuFurVV8rcrVTAFPjhQvzGLNdnTi1jWZ/view
# Path cartella immagini
# /home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS
"""
# Testiamo l'import di una delle immagini
with Image.open('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/test/CAT/10107.jpg') as im:
    im.show()
"""
# Facciamo una lista di immagini
# Inserisci il path alla cartella dei cani e gatti, no test no train
path = '/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/'
# Inizializiamo una lista
img_names = []
# Iniziamo a caricare le immagini
for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder+'/'+img)
# Vediamo se abbiamo caricato le immagini, sono quasi 25k
print(len(img_names))
# Creiamo un dataframe delle dimensioni e forme di queste immagini
img_sizes = []
rejected = []
# Salviamo le immagini che il programma riesce ad aprire per non avere
# errori successivamente
for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)
# Vediamo quante immagini andavano bene
print(len(img_sizes))
print(len(rejected))
# Nessun problema
# Trasformiamo la lista in un dataframe
df = pd.DataFrame(img_sizes)
# I file sono base (0) e altezza (1)
print(df.head())
# Vediamo la larghezza media di tutte le immagini
print(df[0].describe())
# invece per l'altezza
print(df[1].describe())



# Aprimo l'immagine di un cane
dog = Image.open('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/CATS_DOGS/train/DOG/14.jpg')
# dog.show()
print(dog.size)
# Vediamo il valore di un pixel specifico, da 1 a 255 nell'RGB
print(dog.getpixel((0, 0)))
# Ora per lavorarci ci serve rende l'immagine un tensor
# Con compose possiamo passare diversi tipi di trasformazione
transform = transforms.Compose([
    transforms.ToTensor()
])
im = transform(dog)
print(type(im), im.shape) # colori, altezza, base
# Vediamo a grafico l'immagine
# ma per matplot le dimensioni dell'immagine non vanno bene
# le vuole: altezza, base, colori
# Quindi cambiamo l'ordine degl'indici
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# Vediamo l'immagine in formato tensor, sono da 0 a 1 i pixel
# i numeri si posso convertire nel formato del RGB da 1 a 255
# semplicemente moltiplicando per 255 e viceversa se vogliamo da 0 a 1
print(im)
