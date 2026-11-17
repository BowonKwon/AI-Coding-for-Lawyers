from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loaders(data_root, batch_size=64, test_batch_size=1000):
    """(학습 로더, 시험 로더) 를 돌려준다."""
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size)
    return train_loader, test_loader
