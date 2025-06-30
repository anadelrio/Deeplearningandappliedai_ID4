# utils/plot_clusters.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

# Load latents from .pt file
latents = torch.load("experiments/latent-spaces/latents.pt")
if torch.is_tensor(latents):
    latents = latents.numpy()

# Load clustering labels
labels = np.load("experiments/latent-spaces/geodesic_labels.npy")

# Optional: downsample to 1000 points for faster TSNE (comment out to use all)
latents = latents[:1000]
labels = labels[:1000]

# Reduce dimensionality with t-SNE
print("Running t-SNE... (this may take a minute)")
tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=30)
reduced = tsne.fit_transform(latents)

# Plot clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title("Latent space clusters (t-SNE projection)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.show()
