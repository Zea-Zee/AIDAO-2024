# create script, which loads model, does all preprocessing and outputs solution.csv

import numpy as np
import pandas as pd
import pickle

from scripts.data_utils import get_connectome

# Загружаем данные
X = np.load('./data/ts_cut/HCPex/predict.npy')
X = get_connectome(X)

# Загружаем PCA и модель из файла
with open('model.pkl', 'rb') as file:
    saved_objects = pickle.load(file)

pca = saved_objects['pca']
model = saved_objects['model']

# Применяем сохранённый PCA для трансформации данных
n_samples = X.shape[0]
X_reshaped = X.reshape(n_samples, -1)
X_pca = pca.transform(X_reshaped)

# Предсказываем
y_pred = model.predict(X_pca)

# Сохраняем предсказания в CSV
solution = pd.DataFrame(data=y_pred, columns=['prediction'])
solution.to_csv('./solution.csv', index=False)
