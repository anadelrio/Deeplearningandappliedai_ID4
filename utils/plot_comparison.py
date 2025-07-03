import torch
import matplotlib.pyplot as plt
import argparse, os, sys
import numpy as np
from torchvision import datasets, transforms

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vae import VAE
from models.vqvae import VQVAE

# ── helpers ────────────────────────────────────────────────────────────────
def show_images(imgs, row_titles, out_file):
    """
    imgs        : list[list[Tensor]]  shape -> rows x cols
    row_titles  : list[str]           length == rows
    """
    n_rows = len(imgs)
    n_cols = len(imgs[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))

    for r in range(n_rows):
        for c in range(n_cols):
            img = imgs[r][c]
            if img.ndim == 1:  # shape (784,)
                img = img.view(28, 28)
            else:
                img = img.squeeze()
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(row_titles[r], rotation=0, labelpad=40, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)  # Ensure directory exists
    plt.savefig(out_file, dpi=300)
    plt.close()

# ── main ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load MNIST test samples
    test_ds = datasets.MNIST("data", train=False, transform=transforms.ToTensor(), download=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.n, shuffle=False)
    test_imgs, _ = next(iter(test_loader))
    test_imgs = test_imgs.to(device)

    # Load VAE
    vae = VAE(latent_dim=20).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
    vae.eval()
    vae_recon, _, _ = vae(test_imgs)

    # Load VQ-VAE
    vqvae = VQVAE(latent_dim=20, num_embeddings=64, beta=0.25).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_ckpt, map_location=device))
    vqvae.eval()
    vq_recon, _, _ = vqvae(test_imgs)

    # Load Geo-VQ reconstruction from saved codes
    codes = torch.from_numpy(np.load(args.geode_codes)).float().to(device)
    recon_geode = vae.decode(codes).view(-1, 1, 28, 28)

    # Ensure all have the same number of images
    N = min(args.n, len(recon_geode))
    rows = [
        test_imgs[:N].cpu(),
        vae_recon[:N].cpu(),
        vq_recon[:N].cpu(),
        recon_geode[:N].cpu()
    ]

    show_images(rows, ["Original", "VAE", "VQ-VAE", "Geo-VQ"], args.out)
    print(f"Saved: {args.out}")

# ── entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--geode_codes", type=str, required=True)
    parser.add_argument("--out", type=str, default="report/figures/reconstruction_comparison.png")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
