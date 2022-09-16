import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import MNIST
import torchvision.transforms as T

import time


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16

    # Dataset
    train_dataset = MNIST('../data/MNIST', train=True, download=True, transform=T.PILToTensor())
    test_dataset = MNIST('../data/MNIST', train=False, download=True, transform=T.PILToTensor())
    loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    times = []
    for i, (batch, label) in enumerate(loader):
        start = time.time()
        t = time.time() - start
        print(t, batch.shape)
        times.append(t)
    print("Average Time: ", sum(times)/len(times))
