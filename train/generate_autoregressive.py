# train/generate_autoregressive.py
"""
Generate MNIST-like samples by:
 1) Sampling code indices from the trained autoregressive RNN.
 2) Mapping codes → centroid vectors (Geo-VQ codebook).
 3) Decoding each centroid through the VAE decoder.
"""

import sys
from pathlib import Path
# ─────────────────────────────────────────────────────────────────────────────
# Make sure project root is on PYTHONPATH so `import models.vae` works:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# ─────────────────────────────────────────────────────────────────────────────
# Paths
CKPT_AR     = PROJECT_ROOT / "experiments" / "checkpoints" / "autoregressive_model.pt"
CKPT_VAE    = PROJECT_ROOT / "experiments" / "checkpoints" / "vae_mnist.pt"
CENTROIDS   = PROJECT_ROOT / "experiments" / "latent-spaces" / "geodesic_centroids.npy"
OUT_IMG     = PROJECT_ROOT / "experiments" / "images" / "ar_generated.png"
# ─────────────────────────────────────────────────────────────────────────────

class CodeRNN(nn.Module):
    """GRU-based autoregressive model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn   = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)                 # (B, T) → (B, T, E)
        _, h = self.rnn(x)                # h: (1, B, H)
        return self.fc(h.squeeze(0))      # (B, V)

# Load only the decoder half of your VAE:
from models.vae import VAE     # type: ignore # noqa

def sample_codes(model, seq_len, n_samples, start_token=0, temperature=1.0):
    """Autoregressively sample n_samples codes, return last token of each seq."""
    device = next(model.parameters()).device
    seq = torch.full((n_samples, seq_len), start_token,
                     dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for t in range(seq_len):
            logits = model(seq[:, :t+1])         # (B, V)
            probs  = torch.softmax(logits/temperature, dim=-1)
            nxt    = torch.multinomial(probs, 1).squeeze(-1)
            if t+1 < seq_len:
                seq[:, t+1] = nxt
    return seq[:, -1]  # shape (n_samples,)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=16, help="How many images")
    parser.add_argument("--seq_len",   type=int,   default=32, help="Sequence length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Learn vocab_size & latent_dim from centroids file:
    centroids = np.load(CENTROIDS)            # (K, D)
    vocab_size, latent_dim = centroids.shape

    # 2) Load AR checkpoint & infer embed_dim & hidden_dim:
    state = torch.load(CKPT_AR, map_location=device)
    embed_dim  = state["embed.weight"].shape[1]
    hidden_dim = state["rnn.weight_ih_l0"].shape[0] // 3

    # 3) Instantiate AR model and load weights:
    ar = CodeRNN(vocab_size, embed_dim, hidden_dim).to(device)
    ar.load_state_dict(state)
    ar.eval()

    # 4) Sample discrete codes:
    codes = sample_codes(ar, args.seq_len, args.n, temperature=args.temperature)
    codes_np = codes.cpu().numpy()

    # 5) Map codes → continuous latents via centroids:
    z = torch.from_numpy(centroids[codes_np]).float().to(device)  # (n, D)

    # 6) Decode with VAE decoder:
    vae = VAE(latent_dim=latent_dim).to(device)
    vae.load_state_dict(torch.load(CKPT_VAE, map_location=device))
    vae.eval()
    with torch.no_grad():
        recon = vae.decode(z).view(-1, 1, 28, 28)  # (n,1,28,28)

    # 7) Make a grid & save image:
    os.makedirs(OUT_IMG.parent, exist_ok=True)
    grid = make_grid(recon.cpu(), nrow=int(math.sqrt(args.n)), pad_value=1)  # (C,H,W)
    grid_np = grid.cpu().numpy()

    # Handle both 1-channel and 3-channel grids:
    if grid_np.shape[0] == 1:
        img = grid_np[0]          # (H, W)
        cmap = "gray"
    else:
        img = grid_np.transpose(1,2,0)  # (H, W, C)
        cmap = None

    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.imshow(img, cmap=cmap)
    plt.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
    print(f"Generated samples saved to {OUT_IMG}")

    # 8) Report average NLL of codes under AR model:
    with torch.no_grad():
        logits = ar(codes.unsqueeze(1).to(device))  # (n, V)
        nll = F.cross_entropy(logits, codes.to(device))
        print(f"Average NLL of sampled codes: {nll.item():.4f}")

if __name__ == "__main__":
    main()
