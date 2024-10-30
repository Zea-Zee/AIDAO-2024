import numpy as np
import pandas as pd
import pickle

from scripts.data_utils import get_connectome
from scripts.classification_models import MLP
import torch
import torch.nn as nn
import torch.optim as optim

X = np.load('./data/ts_cut/HCPex/predict.npy')
print(X.shape)
X = get_connectome(X)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

n_samples = X.shape[0]
X_reshaped = X.reshape(n_samples, -1).astype('float32')
X_tensor = torch.tensor(X_reshaped)

model.eval()
with torch.no_grad():
    y_pred = model(X_tensor)  # Здесь X_tensor - ваш входной тензор
    y_pred_binary = (y_pred.view(-1) >= 0.5).float()  # Применяем порог
y_pred = y_pred_binary.numpy()
print(y_pred)

solution = pd.DataFrame(data=y_pred, columns=['prediction'])
solution.to_csv('./solution.csv', index=False)

type(y_pred)