import torch
import numpy as np
import os
from tqdm import tqdm

# Paths
LATENTS_PATH = "experiments/latent-spaces/latents.pt"
CENTROIDS_PATH = "experiments/latent-spaces/geodesic_centroids.npy"
OUTPUT_PATH = "experiments/latent-spaces/geodesic_codes.npy"

# Load latents
print("Loading latent vectors...")
latents = torch.load(LATENTS_PATH)  # shape: (N, D)
latents = latents.numpy()

# Load centroids
print("Loading centroids...")
centroids = np.load(CENTROIDS_PATH)  # shape: (K, D)

# Assign closest centroid
print("Assigning codes...")
dists = np.linalg.norm(latents[:, None, :] - centroids[None, :, :], axis=2)  # (N, K)
codes = np.argmin(dists, axis=1)  # (N,)

# Save result
np.save(OUTPUT_PATH, codes)
print(f"Codes saved to '{OUTPUT_PATH}'")
