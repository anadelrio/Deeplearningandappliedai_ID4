# train/reconstruct_from_codes.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vae import VAE

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Config
CODES_PATH = "experiments/latent-spaces/generated_codes.npy"
CENTROIDS_PATH = "experiments/latent-spaces/geodesic_centroids.npy"
VAE_PATH = "experiments/checkpoints/vae_mnist.pt"
OUT_IMAGE = "experiments/reconstructions/generated_from_codes.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load codes and centroids
codes = np.load(CODES_PATH)
centroids = np.load(CENTROIDS_PATH)
latents = centroids[codes]  # shape: (seq_len, latent_dim)

# Average the latents into one vector to reconstruct one image
z = torch.tensor(latents.mean(axis=0)).unsqueeze(0).to(device).float()  # shape: (1, latent_dim)

# Load decoder
model = VAE(latent_dim=20).to(device)
model.load_state_dict(torch.load(VAE_PATH, map_location=device))
model.eval()

with torch.no_grad():
    recon = model.decode(z).cpu().squeeze().numpy()

# Plot
plt.imshow(recon.reshape(28, 28), cmap="gray")
plt.title("Generated image from autoregressive codes")
plt.axis("off")
os.makedirs(os.path.dirname(OUT_IMAGE), exist_ok=True)
plt.savefig(OUT_IMAGE)
plt.close()
print(f"Image saved to {OUT_IMAGE}")
