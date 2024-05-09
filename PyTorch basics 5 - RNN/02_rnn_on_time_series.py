import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Il seguente viene usato per illustrare i valori datatime con matplot
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Il seguente database indica i guadagni mensili di alcol in america
# dal 92 al 2019 in milioni
df = pd.read_csv('/home/michel/Desktop/TopNetwork/08: PyTorch/PYTORCH_NOTEBOOKS/Data/TimeSeriesData/Alcohol_Sales.csv',
    index_col=0, parse_dates=True)
print(df.columns)
print(len(df))
# Devi decidere cosa fare se il tuo dataframe ha dei valori nulli
# in questo caso non li ha
df = df.dropna()
df.plot(figsize=(12, 4))
plt.show()
# Il df tratta i numeri come object, rendiamoli float
y = df['S4248SM144NCEN'].values.astype(float)
# Ora prendiamo solo gli ultimi 12 mesi per i test data
# e il resto per i training data
test_size = 12
train_set = y[:-test_size]
test_set = y[-test_size:]
# Spesso se il dataset è normalizzato le RNN performano meglio
# per il modo in cui si aggiornano i pesi e i bias
scaler = MinMaxScaler(feature_range=(-1, 1))
# In questo caso bisogna dare valore minimo e massimo SOLO DEL TRAINING
# altrimenti avviene il data leakege, semplicemente la rete sarà a
# conoscenza di un valore massimo che non dovrebbe conoscere dai test data
# anche perchè poi perderebbero di senso visto che non dovrebbe conoscerli
scaler.fit(train_set.reshape(-1, 1))
# Per trasformare un set
train_norm = scaler.transform(train_set.reshape(-1, 1))
# Passiamo il set da un array a un tensor
train_norm = torch.FloatTensor(train_norm).view(-1)
window_size = 12
def input_data(seq, ws):
    # Facciamo una lista di tuple, con una sequenza e una previsione
    out = [] # ([0, 1, 2, 3], [4]), ([1, 2, 3, 4], [5]) ...
    L = len(seq)
    # Visto che non possiamo fare previsioni su tutto il train
    # a causa dei valori mancanti, le previsioni dovranno fermarsi
    # quando l'ultima previsione sarà l'ultimo valore del dataset
    for i in range(L - ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    return out

train_data = input_data(train_norm, window_size)



