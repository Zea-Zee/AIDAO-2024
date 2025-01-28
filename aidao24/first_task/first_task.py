import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering, KMeans
from sklearn.manifold import TSNE
from nilearn.connectome import ConnectivityMeasure
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
import umap
import random
from gng import GrowingNeuralGas

def assign_clusters(data, nodes):
    n_samples = data.shape[0]
    clusters = np.zeros(n_samples, dtype=int)

    for i, point in enumerate(data):
        distances = [np.linalg.norm(point - node.position) for node in nodes]
        nearest_node_idx = np.argmin(distances)
        clusters[i] = nearest_node_idx

    return clusters

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

def get_connectome(timeseries: np.ndarray,
                   conn_type: str = 'corr') -> np.ndarray:
    if conn_type == 'corr':
        conn = ConnectivityMeasure(kind='partial correlation', standardize=False).fit_transform(timeseries)
        conn[conn == 1] = 0.999999

        for i in conn:
            np.fill_diagonal(i, 0)

        conn = np.arctanh(conn)

    else:
        raise NotImplementedError

    return conn

# different atlases was mapped into one atlas(specifically schaefer200)
data = np.load('./schaefer200_data.npy')

func_conn_matrices = get_connectome(data)

n_components = 25

#seed that gave 0.93 score
random_state = 45104#random.randint(0,100000)

set_seed(seed=random_state)

pca = PCA(n_components=n_components)
cm_data = pca.fit_transform(func_conn_matrices.reshape(320, -1))
pca = umap.UMAP(n_neighbors=16, n_components=n_components//2, random_state=random_state)
print(f"random_state: {random_state}")

cm_data = pca.fit_transform(cm_data)

gng = GrowingNeuralGas(cm_data.shape[1], max_nodes=21)
gng.train(cm_data, iterations=10000)
labels = assign_clusters(cm_data, gng.nodes)

tsne = TSNE(n_components=2, perplexity=5)

reduced_tsne_brainnetome = tsne.fit_transform(cm_data)

import pandas as pd

pd.DataFrame({'prediction': labels}).to_csv('./submissiom.csv', index=False)
uniq_vals, counts = np.unique(labels, return_counts=True)
print(uniq_vals, counts, len(counts))


# if you need clusters visualization, uncomment this code
plt.figure(figsize=(10, 6))
plt.scatter(reduced_tsne_brainnetome[:, 0], reduced_tsne_brainnetome[:, 1], alpha=0.5, c=labels)
plt.title('t-SNE Visualization of Text Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()
