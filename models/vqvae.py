import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
class VectorQuantizer(nn.Module):
    """
    Codebook + nearest-neighbor quantization (versión simple, sin EMA).

    Devuelve:
        • vq_loss  –   gradúa al codebook  (escalado por β)
        • commit   –   gradúa al encoder   (escalado por β)
        • z_q_st   –   latente cuantizado (straight-through)
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim
        self.beta           = beta

        # Codebook (K × D)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight,
                         -1.0 / num_embeddings,
                          1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        """
        z : (B, D) outputs of the encoder
        """
        # ---- distance L2 to each embedding (B × K) ----
        with torch.no_grad():
            emb = self.embedding.weight  # (K, D)
            dists = (
                z.pow(2).sum(1, keepdim=True)           # ‖z‖²
                + emb.pow(2).sum(1)                     # ‖e‖²
                - 2 * z @ emb.t()                       # −2·z·e
            )
            indices = dists.argmin(1)                   # (B,)

        # Cuantización
        z_q = self.embedding(indices)                   # (B, D)

        # Straight-through
        z_q_st = z + (z_q - z).detach()

        # Pérdidas (ambas escaladas por β)
        vq_loss = self.beta * F.mse_loss(z_q, z.detach())       # grads → codebook
        commit  = self.beta * F.mse_loss(z_q.detach(), z)       # grads → encoder

        return z_q_st, vq_loss, commit, indices


# ----------------------------------------------------
class VQVAE(nn.Module):
    """VQ-VAE (totally connected) for MNIST 28×28."""
    def __init__(self,
                 latent_dim: int = 20,
                 num_embeddings: int = 64,
                 beta: float = 0.25):
        super().__init__()

        # Encoder
        self.enc1 = nn.Linear(28 * 28, 400)
        self.enc2 = nn.Linear(400, latent_dim)

        # Codebook
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, beta)

        # Decoder
        self.dec1 = nn.Linear(latent_dim, 400)
        self.dec2 = nn.Linear(400, 28 * 28)

    # ---------- Encoder ----------
    def encode(self, x):
        h = F.relu(self.enc1(x))
        return self.enc2(h)

    # ---------- Decoder ----------
    def decode(self, z_q):
        h = F.relu(self.dec1(z_q))
        return torch.sigmoid(self.dec2(h))

    # ---------- Forward ----------
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        z   = self.encode(x)
        z_q, vq_loss, commit, _ = self.quantizer(z)
        recon = self.decode(z_q)
        return recon, vq_loss, commit


# ----------------------------------------------------
def vqvae_loss(recon_x: torch.Tensor,
               x: torch.Tensor,
               vq_loss: torch.Tensor,
               commit: torch.Tensor):
    """
    Loss total = BCE + vq_loss + commit
    Devuelve (total, bce, vq+commit) para logging.
    """
    bce = F.binary_cross_entropy(recon_x,
                                 x.view(-1, 28 * 28),
                                 reduction='sum')
    total = bce + vq_loss + commit
    return total, bce, (vq_loss + commit)
