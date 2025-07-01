# train/train_geodesic.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
import numpy as np
from sklearn.cluster import KMeans

# Config
LATENTS_PATH = "experiments/latent-spaces/latents.pt"
LABELS_OUT = "experiments/latent-spaces/geodesic_labels.npy"
CENTROIDS_OUT = "experiments/latent-spaces/geodesic_centroids.npy"
N_CLUSTERS = 10
N_SAMPLES = 1000  # Reduce if memory is a problem

# Load latent vectors
print("Loading latent vectors...")
latents = torch.load(LATENTS_PATH)
if torch.is_tensor(latents):
    latents = latents.numpy()

latents = latents[:N_SAMPLES]
print(f"Latents shape: {latents.shape}")

# Run KMeans directly in latent space
print("Running KMeans clustering...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
kmeans.fit(latents)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Save outputs
print("Saving clustering results...")
os.makedirs(os.path.dirname(LABELS_OUT), exist_ok=True)
np.save(LABELS_OUT, labels)
np.save(CENTROIDS_OUT, centroids)

print("Done.")

