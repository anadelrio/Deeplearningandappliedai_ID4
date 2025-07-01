# train/evaluate_reconstructions.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from models.vae import VAE
import torch.nn.functional as F



# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "experiments/checkpoints/vae_mnist.pt"
save_dir = "reconstructions"
os.makedirs(save_dir, exist_ok=True)

# Load dataset
transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

# Load model
model = VAE().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Take one batch
dataiter = iter(testloader)
images, _ = next(dataiter)
images = images.to(device)
with torch.no_grad():
    recon, _, _ = model(images)

# Plot
images = images.cpu()
recon = recon.cpu()
recon = recon.view(-1, 1, 28, 28)

fig, axs = plt.subplots(2, 10, figsize=(15, 3))
for i in range(10):
    
    axs[0, i].imshow(images[i].squeeze(), cmap="gray")
    axs[0, i].axis('off')
    axs[1, i].imshow(recon[i].squeeze(), cmap="gray")
    axs[1, i].axis('off')

axs[0, 0].set_ylabel("Original", fontsize=12)
axs[1, 0].set_ylabel("Reconstructed", fontsize=12)
plt.tight_layout()
plt.savefig("experiments/reconstructions/reconstruction_example.png")
plt.close()

# Compute MSE (reconstruction loss)
recon_flat = recon.view(recon.size(0), -1)
images_flat = images.view(images.size(0), -1)
mse = F.mse_loss(recon_flat, images_flat).item()

print(f"MSE reconstruction loss (VAE): {mse:.6f}")


print("Reconstruction image saved in 'reconstructions/reconstruction_example.png'")
