
"""Utility functions for computing VAE-related losses and simple metrics."""
import torch
import torch.nn.functional as F

def elbo_loss(recon_x, x, mu, logvar, reduction='sum'):
    """Compute the Evidence Lower Bound (negative log-likelihood) for a VAE.

    Args:
        recon_x (Tensor): reconstructed batch, shape (N, 28*28)
        x (Tensor): original batch, shape (N, 1, 28, 28)
        mu (Tensor): latent means, shape (N, latent_dim)
        logvar (Tensor): latent log-variances, shape (N, latent_dim)
        reduction (str): 'sum' or 'mean'

    Returns:
        total_loss, recon_bce, kl_div
    """
    bce = F.binary_cross_entropy(
        recon_x, x.view(-1, 28*28), reduction=reduction
    )
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == 'mean':
        kld /= x.size(0)
        bce /= x.size(0)
    return bce + kld, bce, kld
