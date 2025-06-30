import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from scipy.sparse.csgraph import shortest_path


def load_latents(path):
    """
    Load latent vectors from a .pt file and return as a NumPy array.
    """
    z = torch.load(path)
    return z.numpy() if torch.is_tensor(z) else z


def build_knn_graph(latents, k=10):
    """
    Build a k-nearest neighbors graph using sklearn and networkx.
    Each node represents a latent vector.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(latents)
    distances, indices = nbrs.kneighbors(latents)

    G = nx.Graph()
    for i in range(latents.shape[0]):
        for j in range(1, k):  # skip j=0 (self-loop)
            G.add_edge(i, indices[i][j], weight=distances[i][j])
    return G


def compute_geodesic_distances(graph):
    """
    Compute the geodesic distance matrix using shortest paths on the sparse graph.
    Returns a dense NumPy array D[i][j] with the shortest path distance from node i to j.
    """
    A = nx.to_scipy_sparse_array(graph, weight='weight', format='csr')
    D = shortest_path(csgraph=A, directed=False, return_predecessors=False)
    return D


def geodesic_kmeans(D, n_clusters=10, random_state=42):
    """
    Apply K-means clustering over a geodesic distance matrix using MDS projection.
    Input D must be a precomputed distance matrix (n x n).
    """
    mds = MDS(n_components=10, dissimilarity='precomputed', random_state=random_state)
    embedded = mds.fit_transform(D)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedded)
    return labels, kmeans.cluster_centers_
