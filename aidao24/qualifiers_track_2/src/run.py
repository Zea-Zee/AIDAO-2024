import numpy as np
import pandas as pd
import pickle

from scripts.data_utils import get_connectome

X = np.load('./data/ts_cut/HCPex/predict.npy')
print(X.shape)
X = get_connectome(X)

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