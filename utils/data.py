from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_test_loader(batch_size=5000):
    """
    Returns a DataLoader for the MNIST test set.

    Args:
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: MNIST test data loader.
    """
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transform,
        download=True
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
