"""
Train a simple VAE on MNIST and log metrics epoch-by-epoch.
Usage:
    python train/train_vae.py --epochs 10 --batch_size 128 --latent_dim 20
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # type: ignore
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- rutas robustas ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # compute the root of the proyect using the ubication of the file
CHECKPOINT_DIR = ROOT / 'experiments' / 'checkpoints'
LOG_DIR = ROOT / 'experiments' / 'logs'

# aseguramos carpetas
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
from models.vae import VAE # type: ignore
from utils.metrics import elbo_loss # type: ignore


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(ROOT / 'data'),   # dataset en carpeta del proyecto
                              train=True, transform=transform, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # archivo de log CSV
    log_file = LOG_DIR / 'vae_train_log.csv'
    if not log_file.exists():
        log_file.write_text('epoch,total_loss,bce,kld\n')

    # ---------- loop de entrenamiento ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = bce_total = kld_total = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, bce, kld = elbo_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            bce_total += bce.item()
            kld_total += kld.item()

        # print y log
        print(f"Epoch {epoch:02d} | ELBO: {total_loss:.2f} | BCE: {bce_total:.2f} | KLD: {kld_total:.2f}")
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{total_loss},{bce_total},{kld_total}\n")

    # ---------- guardar modelo ----------
    ckpt_path = CHECKPOINT_DIR / 'vae_mnist.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved in: {ckpt_path.relative_to(ROOT)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--cpu', action='store_true', help='force training on CPU')
    args = parser.parse_args()
    train(args)

