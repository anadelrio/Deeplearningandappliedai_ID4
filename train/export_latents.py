# train/export_latents.py
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys
import os

# Añadir ruta raíz al path para importar VAE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import VAE

# --- Rutas base ---
ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "experiments" / "checkpoints" / "vae_mnist.pt"
LATENTS_PATH = ROOT / "experiments" / "latent-spaces" / "latents.pt"
LATENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Configuración ---
latent_dim = 20
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Modelo ---
vae = VAE(latent_dim=latent_dim).to(device)
vae.load_state_dict(torch.load(CHECKPOINT, map_location=device))
vae.eval()

# --- Dataset ---
transform = transforms.ToTensor()
dataset = datasets.MNIST(root=ROOT / "data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# --- Codificar a latentes ---
latents = []

with torch.no_grad():
    for x, _ in loader:
        x = x.to(device).view(-1, 28 * 28)
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        latents.append(z.cpu())

latents = torch.cat(latents, dim=0)
latents = latents[:5000]  # keep only the first 5000
torch.save(latents, LATENTS_PATH)
print(f"✅ Latents saved to: {LATENTS_PATH}")

