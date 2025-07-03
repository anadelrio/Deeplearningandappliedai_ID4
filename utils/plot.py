import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms # type: ignore
from torch.utils.data import DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae import VAE # type: ignore

def plot_losses(log_file='experiments/logs/vae_train_log.csv'):
    df = pd.read_csv(log_file)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['total_loss'], label='ELBO (total)')
    plt.plot(df['epoch'], df['bce'], label='BCE (reconstruction loss)')
    plt.plot(df['epoch'], df['kld'], label='KL divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_reconstructions(model_path='experiments/checkpoints/vae_mnist.pt', n_images=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=n_images, shuffle=True)

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            recon, _, _ = model(x)
            break

    fig, axs = plt.subplots(2, n_images, figsize=(n_images, 2))
    for i in range(n_images):
        axs[0, i].imshow(x[i].cpu().squeeze(), cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
        axs[1, i].axis('off')

    axs[0, 0].set_title('Original')
    axs[1, 0].set_title('Reconstrucción')
    plt.suptitle('Reconstrucciones VAE')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_losses()
    show_reconstructions()
