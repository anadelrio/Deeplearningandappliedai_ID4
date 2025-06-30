# train/train_geodesic.py
from pathlib import Path
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.graph_tools import load_latents, build_knn_graph, compute_geodesic_distances, geodesic_kmeans

ROOT = Path(__file__).resolve().parent.parent
LATENTS_PATH = ROOT / "experiments" / "latent-spaces" / "latents.pt"
LABELS_PATH = ROOT / "experiments" / "latent-spaces" / "geodesic_labels.npy"
CENTROIDS_PATH = ROOT / "experiments" / "latent-spaces" / "geodesic_centroids.npy"

K = 10
N_CLUSTERS = 10

print("Loading latent vectors...")
latents = load_latents(LATENTS_PATH)

print("Building k-NN graph...")
graph = build_knn_graph(latents, k=K)

print("Computing geodesic distance matrix...")
D = compute_geodesic_distances(graph)

print("Running geodesic K-means...")
labels, centroids = geodesic_kmeans(D, n_clusters=N_CLUSTERS)

print("Saving results...")
np.save(LABELS_PATH, labels)
np.save(CENTROIDS_PATH, centroids)

print("Done.")
