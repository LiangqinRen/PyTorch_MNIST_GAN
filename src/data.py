import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader


def get_MNIST_dataloader(args):
    DATA_PATH = "data"
    EPOCH = args.epoch

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5), std=(0.5)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=DATA_PATH, train=True, transform=transform, download=True
    )

    dataloaders = {
        "name": "MNIST",
        "epoch": EPOCH,
        "train": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
    }

    return dataloaders
