import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Paths
vae_img_path = "experiments/reconstructions/reconstruction_example.png"
vqvae_img_path = "experiments/reconstructions/generated_from_codes.png"
out_path = "experiments/reconstructions/comparison.png"

# Load images
vae_img = mpimg.imread(vae_img_path)
vqvae_img = mpimg.imread(vqvae_img_path)

# Plot side by side
fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].imshow(vae_img)
axs[0].set_title("VAE Reconstruction")
axs[0].axis("off")

axs[1].imshow(vqvae_img)
axs[1].set_title("VQ + Autoregressive Generation")
axs[1].axis("off")

plt.tight_layout()
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path)
plt.close()

print(f"Saved comparison to {out_path}")
