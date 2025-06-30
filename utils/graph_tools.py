# utils/graph_tools.py
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

def compute_geodesic_distances(X, n_neighbors=10):
    print("🔹 Building k-NN graph...")
    knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance', include_self=False)
    print("🔹 Computing geodesic distance matrix...")
    D = shortest_path(knn_graph, method='D', directed=False)
    return D.astype(np.float32)  # Save RAM

def geodesic_kmeans(D, n_clusters=10):
    print("🔹 Embedding distance matrix using MDS...")
    mds = MDS(n_components=10, dissimilarity='precomputed', random_state=0)
    embedded = mds.fit_transform(D)
    print("🔹 Running geodesic K-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embedded)
    centroids = kmeans.cluster_centers_
    return labels, centroids
