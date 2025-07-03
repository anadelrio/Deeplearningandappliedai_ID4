import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational Autoencoder for 28x28 grayscale images (e.g., MNIST)."""
    def __init__(self, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 400)
        self.fc3 = nn.Linear(400, 28*28)

    # -------- Encoder --------
    def encode(self, x: torch.Tensor):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    # -------- Reparameterization trick --------
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # -------- Decoder --------
    def decode(self, z: torch.Tensor):
        h = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h))

    # -------- Forward pass --------
    def forward(self, x: torch.Tensor):
        x = x.view(-1, 28 * 28)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
