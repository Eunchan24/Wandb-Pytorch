from random import shuffle
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch

def CIFAR10_Dataset(batch_size, transform):
    transform  =  transform
    # download training dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                            transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader