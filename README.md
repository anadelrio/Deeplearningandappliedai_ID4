
# VQ‑VAE with Geodesic Quantization

This project implements a modified VQ‑VAE pipeline where latent–space quantization is performed a‑posteriori using geodesic distances on a k‑NN graph.  
The goal is to evaluate whether this strategy yields better discrete representations than the classical end‑to‑end‑trained VQ‑VAE.

---

## Project Structure (current & upcoming)

```text
vqvae‑geodesic/
│
├── models/              # neural‑network architectures
│├── vae.py              # Variational Autoencoder (done)
│├── vqvae.py            # classical VQ‑VAE (todo)
│└── geodesic_vq.py      # quantization‑via‑geodesics (todo)
│
├── train/               # training & quantization scripts
│├── train_vae.py        # trains the VAE, logs metrics, saves checkpoint (done)
│├── train_vqvae.py      # baseline VQ‑VAE training (todo)
│└── train_geodesic.py   # k‑NN graph + geodesic k‑means (todo)
│
├── utils/               # helper functions
│├── metrics.py          # ELBO / BCE / KLD loss utilities
│├── plot.py             # loss curves & reconstruction visualisation
│├── datasets.py         # custom dataset wrappers (todo)
│└── graph_tools.py      # k‑NN graph & geodesic distances (todo)
│
├── experiments/         # auto‑generated results
│├── logs/               # CSV training logs
││└── vae_train_log.csv  # created by train_vae.py (done)
│└── checkpoints/        # model checkpoints
│  └── vae_mnist.pt      # trained VAE weights (done)
│
├── data/                # downloaded datasets
│└── MNIST/              # raw .gz files (torchvision ≥0.15 no longer writes processed/)
│
└── report/              # final PDF report (to be added)
```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## What each file does

File -> Description 

`models/vae.py` -> defines a simple fully‑connected Variational Autoencoder for 28×28 MNIST digits. Takes a batch of images, encodes to μ & logσ², samples z, decodes back to pixel space. 

`train/train_vae.py` -> loads MNIST, trains the VAE using ELBO loss (via `utils.metrics.elbo_loss`), logs losses to `experiments/logs/vae_train_log.csv`, and saves the checkpoint to `experiments/checkpoints/vae_mnist.pt`. 

`utils/metrics.py` -> implements `elbo_loss`: returns total ELBO, reconstruction BCE, and KL divergence. 

`utils/plot.py` -> reads the CSV log and plots ELBO/BCE/KLD curves; loads the checkpoint and shows original vs reconstruction image grids. 

`experiments/logs/vae_train_log.csv` -> auto‑generated CSV with `epoch,total_loss,bce,kld`. 

`experiments/checkpoints/vae_mnist.pt` -> serialized `state_dict` of the trained VAE. 

`data/MNIST/raw/` -> original MNIST files (`*.gz`). Torchvision ≥0.15 keeps them in memory.

> **Upcoming files** (`vqvae.py`, `graph_tools.py`, etc.) are placeholders for the next stages (classical VQ‑VAE baseline and geodesic quantization).

---

## 🚀 How to run

```bash
# 1) install deps
pip install -r requirements.txt

# 2) train the VAE
python train/train_vae.py --epochs 10

# 3) plot training curves & reconstructions
python utils/plot.py

# (future)
# 4) build k‑NN graph & geodesic quantisation
# python train/train_geodesic.py
```

---

## 📊 Metrics logged

- **ELBO** (total loss)  
- **BCE** reconstruction loss  
- **KL Divergence**  
- *(planned)* Geodesic clustering distortion, intra‑/inter‑cluster distances.

---

## 👥 Contributors

- **Ana** — VAE implementation, theory review, documentation.  
- **Darío** — Geodesic clustering, codebook quantisation (upcoming).

---

## 📚 References

- Tenenbaum *et al.*, **Isomap** (2000)  
- van den Oord *et al.*, **VQ‑VAE** (2017)  
- Duque *et al.*, **Geodesic Clustering in VAEs** (2021)
