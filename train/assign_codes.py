# train/assign_codes.py
import numpy as np
import torch
import os
from tqdm import tqdm

# -------------------- Paths --------------------
LATENTS_PATH       = "experiments/latent-spaces/latents.pt"
CENTROIDS_PATH     = "experiments/latent-spaces/geodesic_centroids.npy"
LABELS_OUT_PATH    = "experiments/latent-spaces/geodesic_labels.npy"
GENERATED_OUT_PATH = "experiments/latent-spaces/generated_codes.npy"
os.makedirs(os.path.dirname(LABELS_OUT_PATH), exist_ok=True)
# ------------------------------------------------

print("Loading latent vectors ...")
latents = torch.load(LATENTS_PATH)           # shape: (N, D)  -> expected (N, 20)
latents = latents.numpy()

print("Loading geodesic centroids ...")
centroids = np.load(CENTROIDS_PATH)          # shape: (K, D)

assert (
    latents.shape[1] == centroids.shape[1]
), f"Dim mismatch: latents {latents.shape}, centroids {centroids.shape}"

# ------------ Assign closest centroid ------------
print("Assigning each latent to nearest centroid ...")
# dists: (N, K)
dists = np.linalg.norm(latents[:, None, :] - centroids[None, :, :], axis=2)
labels = np.argmin(dists, axis=1)            # (N,) integer codes

# ------------ Save outputs -----------------------
print("Saving labels and generated latent vectors ...")
np.save(LABELS_OUT_PATH, labels)             # integer codes
generated = centroids[labels]                # (N, D) vectors compatible with VAE decoder
np.save(GENERATED_OUT_PATH, generated)

print(
    f"Saved:\n  • Labels -> {LABELS_OUT_PATH}\n"
    f"Generated codes -> {GENERATED_OUT_PATH}  (shape {generated.shape})"
)
