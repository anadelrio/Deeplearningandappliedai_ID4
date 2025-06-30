# train/train_geodesic.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import numpy as np
from utils.graph_tools import compute_geodesic_distances, geodesic_kmeans

# Config
LATENTS_PATH = "experiments/latent-spaces/latents.pt"
LABELS_OUT = "experiments/latent-spaces/geodesic_labels.npy"
CENTROIDS_OUT = "experiments/latent-spaces/geodesic_centroids.npy"
N_CLUSTERS = 10
N_NEIGHBORS = 10
N_SAMPLES = 1000  # Reduce if memory is a problem

# Load latent vectors
print("🔹 Loading latent vectors...")
latents = torch.load(LATENTS_PATH)
if torch.is_tensor(latents):
    latents = latents.numpy()

latents = latents[:N_SAMPLES]
print(f"Latents shape: {latents.shape}")

# Compute geodesic distances
D = compute_geodesic_distances(latents, n_neighbors=N_NEIGHBORS)

# Run geodesic K-means
labels, centroids = geodesic_kmeans(D, n_clusters=N_CLUSTERS)

# Save outputs
print("🔹 Saving clustering results...")
os.makedirs(os.path.dirname(LABELS_OUT), exist_ok=True)
np.save(LABELS_OUT, labels)
np.save(CENTROIDS_OUT, centroids)

print("✅ Done.")
