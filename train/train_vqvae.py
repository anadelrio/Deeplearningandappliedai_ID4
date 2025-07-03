# train/train_vqvae.py
import argparse, os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vqvae import VQVAE, vqvae_loss   # type: ignore

ROOT           = Path(__file__).resolve().parent.parent
CKPT_DIR       = ROOT / "experiments" / "checkpoints"
LOG_DIR        = ROOT / "experiments" / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True,  exist_ok=True)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = VQVAE(latent_dim=args.latent_dim,
                  num_embeddings=args.embeddings,
                  beta=args.beta).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ds = datasets.MNIST(root=str(ROOT / "data"),
                        train=True, transform=transforms.ToTensor(), download=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    log_csv = LOG_DIR / "vqvae_train_log.csv"
    if not log_csv.exists():
        log_csv.write_text("epoch,total_loss,bce,codebook+commit\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = bce_tot = vqc_tot = 0.0

        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad()

            recon, vq_loss, commit = model(x)
            loss, bce, vqc = vqvae_loss(recon, x, vq_loss, commit)
            loss.backward()
            opt.step()

            tot  += loss.item()
            bce_tot += bce.item()
            vqc_tot += vqc.item()

        print(f"E{epoch:02d} | Loss {tot:.1f} | BCE {bce_tot:.1f} | VQ+Commit {vqc_tot:.1f}")
        with open(log_csv, "a") as f:
            f.write(f"{epoch},{tot},{bce_tot},{vqc_tot}\n")

    ckpt = CKPT_DIR / "vqvae_mnist.pt"
    torch.save(model.state_dict(), ckpt)
    print("saved to", ckpt.relative_to(ROOT))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=128)
    p.add_argument("--latent_dim",  type=int,   default=20)
    p.add_argument("--embeddings",  type=int,   default=64)
    p.add_argument("--beta",        type=float, default=0.01)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--cpu",         action="store_true")
    args = p.parse_args()
    train(args)
