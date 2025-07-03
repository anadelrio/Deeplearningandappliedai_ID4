# VQ‑VAE with Geodesic Quantization
This project implements a modified VQ‑VAE pipeline where latent–space quantization is performed a‑posteriori using geodesic distances on a k‑NN graph.  
The goal is to evaluate whether this strategy yields better discrete representations than the classical end‑to‑end‑trained VQ‑VAE.
---

## Project Structure

```
.
├── .gitignore                         # Ignore __pycache__, data cache, etc.
├── README.md                          
├── data.zip                           # zipped raw MNIST
├── requirements.txt                   # Python dependencies
│
├── experiments/                       # artefacts produced by running code
│   ├── checkpoints/                   # saved model weights
│   │   ├── vae_mnist.pt               # trained VAE
│   │   ├── vqvae_mnist.pt             # trained VQ-VAE
│   │   └── autoregressive_model.pt    # (placeholder for future work)
│   ├── latent-spaces/                 # numpy dumps of latent codes
│   │   ├── geodesic_centroids.npy     # centroids found by geodesic k-means
│   │   ├── geodesic_codes.npy         # integer code-book indices per sample
│   │   ├── geodesic_labels.npy        # cluster labels
│   │   └── generated_codes.npy        # codes synthesised for Geo-VQ recon
│   ├──logs/
│   │   ├── vae_train_log.csv          # epoch-wise ELBO, BCE, KL
│   │   ├── vqvae_train_log.csv        # epoch-wise BCE + commit loss
│   │   └── results_table.csv          # final numbers used in the paper
│   └──reconstructions/
│       ├──comparison.jpg
│       ├──generated_from_codes.jpg
│       └──reconstruction_example.jpg
│       
├── report/                           
│   ├── report_3julyversion.pdf        # 3 july report version
│   └── figures/                       
│       └──reconstruction_comparison.jpg  #image used in the report to compare the models
│     
├── models/                           
│   ├── vae.py                         # fully-connected VAE for 28×28 images
│   └── vqvae.py                       # VQ-VAE + VectorQuantizer class
│
├── train/                              # training & evaluation entry-points
│   ├── train_vae.py                   # trains the baseline VAE
│   ├── train_vqvae.py                 # trains VQ-VAE (and Geo-VQ when β=0)
│   ├── train_geodesic.py              # runs geodesic k-means on VAE latents
│   ├── export_latents.py              # dumps latent vectors to *.pt
│   ├── train_autoregressive.py        # Train PixelCNN over latent codes
│   ├── assign_codes.py                # Assign discrete codes to dataset
│   ├── generate_codes.py              # Generate and save codebooks
│   ├── reconstruct_from_codes.py      # Decode images from saved codes
│   └── evaluate_reconstructions.py    # writes metrics + saves comparison fig
│
└── utils/                        # helper functions / plotting utilities
    ├── data_utils.py                  # `get_train_loader`, `get_test_loader`
    ├── losses.py                      # `elbo_loss`, `vqvae_loss`
    ├── graph_tools.py                 # build k-NN graph, shortest paths
    ├── plot.py                        # plot per-epoch losses & recon samples
    ├── plot_clusters.py               # 2-D latent TSNE + cluster colouring
    └── plot_comparison.py             # generate reconstruction grid figure


```

---

## How to Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Unzip MNIST data
Make sure `data.zip` is extracted in the root folder to have `data/`.
### 3. Train models
```bash
python train/train_vae.py          # For VAE
python train/train_vqvae.py        # For VQ-VAE / Geo-VQ
```
### 4. Evaluate reconstructions
```bash
python -m train.evaluate_reconstructions --model vae
python -m train.evaluate_reconstructions --model vqvae
python -m train.evaluate_reconstructions --model geode
```
### 5. Generate the visualization
```bash
python utils/plot_comparison.py --vae_ckpt experiments/checkpoints/vae_mnist.pt --vqvae_ckpt experiments/checkpoints/vqvae_mnist.pt --geode_codes experiments/latent-spaces/generated_codes.npy --out report/figures/reconstruction_comparison.png --n 8
```
Results will be appended to `experiments/logs/results_table.csv`.

---

## Metrics

For each model, we compute:

- Total Loss: Overall objective value
- Binary Cross-Entropy (BCE): Reconstruction error
- KL Divergence / VQ Commit Loss: Latent regularization

---

## Notes

- Geo-VQ uses spherical k-means over latent codes.
- Trained weights are included under `experiments/checkpoints/`.
