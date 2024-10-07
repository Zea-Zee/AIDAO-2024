import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Пример данных
X = np.random.rand(100, 20)
Y = np.random.randint(0, 2, 100)

# Настраиваем PCA и RandomForest
pca = PCA(n_components=0.9)
X_pca = pca.fit_transform(X)

model = RandomForestClassifier(class_weight='balanced', max_depth=2, n_estimators=250)
model.fit(X_pca, Y)

# Сохраняем PCA и модель в один файл
with open('model.pkl', 'wb') as file:
    pickle.dump({'pca': pca, 'model': model}, file)


n_samples = X.shape[0]
X_reshaped = X.reshape(n_samples, -1)
X_pca = pca.transform(X_reshaped)
print(f"X shape: {X_reshaped.shape}")
print(f"X pca shape: {X_pca.shape}")


y_pred = model.predict(X_pca)
print(y_pred)

solution = pd.DataFrame(data=y_pred, columns=['prediction'])
solution.to_csv('./solution.csv', index=False)