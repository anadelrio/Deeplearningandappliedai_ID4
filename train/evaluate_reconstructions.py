import argparse, os, sys, csv
from pathlib import Path
import torch
import numpy as np
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vae   import VAE           # type: ignore
from models.vqvae import VQVAE, vqvae_loss          # type: ignore
from utils.data   import get_test_loader            # type: ignore
from utils.metrics import elbo_loss #type: ignore

ROOT      = Path(__file__).resolve().parent.parent
CKPT_DIR  = ROOT / "experiments" / "checkpoints"
CSV_OUT   = ROOT / "experiments" / "logs" / "results_table.csv"
CSV_OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ---------------------------------------------------------
@torch.inference_mode()
def eval_vae(model: VAE, device):
    loader = get_test_loader()
    model.eval()
    total = bce_tot = kld_tot = 0.0
    for x, _ in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, bce, kld = elbo_loss(recon, x, mu, logvar)
        total += loss.item();  bce_tot += bce.item();  kld_tot += kld.item()
    return total, bce_tot, kld_tot

@torch.inference_mode()
@torch.inference_mode()
def eval_vqvae(model: VQVAE, device, beta: float = 0.01):
    loader = get_test_loader()
    model.eval()
    total = bce_tot = vqc_tot = 0.0
    for x, _ in loader:
        x = x.to(device)
        recon, vq_loss, commit = model(x)

        vq_loss  = beta * vq_loss
        commit   = beta * commit

        loss, bce, vqc = vqvae_loss(recon, x, vq_loss, commit)
        total += loss.item();  bce_tot += bce.item();  vqc_tot += vqc.item()
    return total, bce_tot, vqc_tot

# ---------- Main ------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    if args.model in ("vae", "all"):
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(CKPT_DIR / "vae_mnist.pt", map_location=device))
        tot, bce, kld = eval_vae(vae, device)
        rows.append(("VAE", tot, bce, kld))

    if args.model in ("vqvae", "all"):
        vq = VQVAE(beta=0.01).to(device)
        vq.load_state_dict(torch.load(CKPT_DIR / "vqvae_mnist.pt", map_location=device))
        tot, bce, vqc = eval_vqvae(vq, device, beta=0.01)
        rows.append(("VQ-VAE", tot, bce, vqc))

    if args.model in ("geode", "all"):
        
        vae = VAE().to(device)
        vae.load_state_dict(torch.load(CKPT_DIR / "vae_mnist.pt", map_location=device))
        vae.eval()

        # Codebook del VQ-VAE
        vq = VQVAE().to(device)
        vq.load_state_dict(torch.load(CKPT_DIR / "vqvae_mnist.pt", map_location=device))
        vq.eval()

        # Códigos generados
        codes = torch.from_numpy(
            np.load(ROOT / "experiments" / "latent-spaces" / "generated_codes.npy")
        ).long().to(device)                                           # (N,)

        # Vector cuantizado → (N, D)
        emb  = vq.quantizer.embedding.weight          # (K, D)
        z_q  = emb[codes]                             # (N, D)

        recon = vae.decode(z_q)                       # (N, 20, 784) en tu caso
        if recon.dim() == 3:                          #  ← aplanamos eje 1
            recon = recon.mean(dim=1)                 # (N, 784)

        recon = recon.view(-1, 28 * 28)               

        # Imágenes de test aplanadas 
        test_imgs, _ = next(iter(get_test_loader()))
        test_imgs = test_imgs[: len(recon)].to(device).view(-1, 28 * 28)

        # BCE
        bce = F.binary_cross_entropy(recon, test_imgs, reduction="sum")
        rows.append(("Geo-VQ", bce.item(), bce.item(), 0.0))

    # Guardar / AÑADIR resultados al CSV 
    import csv

    header = ["model", "total_loss", "bce", "vq_commit"]

    # Si el CSV NO existe, lo creamos con la cabecera
    if not CSV_OUT.exists():
        with open(CSV_OUT, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Siempre abrimos en modo "a" (append) para añadir nuevas filas
    with open(CSV_OUT, "a", newline="") as f:
        csv.writer(f).writerows(rows)

    print("Row(s) added to", CSV_OUT.relative_to(ROOT))
    # -----------------------------------------------------------


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["vae", "vqvae", "geode", "all"], default="all")
    args = p.parse_args()
    main(args)
