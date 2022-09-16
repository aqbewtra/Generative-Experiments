import os
import time
from numpy import dtype
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

from simple_discriminator import SimpleDiscriminator
from simple_generator import SimpleGenerator
from losses import gan_loss
from save_grid import save_grid
from wasserstein_loss import wass_d_loss, wass_g_loss

from PIL import Image
import matplotlib.pyplot as plt

save_images_root = 'samples'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task Parameters
img_shape = (1, 28, 28)
z_dim = 100

# Hyperparameters
epochs = 30
batch_size = 16
lr = 4e-3
b1 = .5
b2 = .999

# Dataset
train_dataset = MNIST('../data/MNIST', train=True, download=True, transform=T.PILToTensor())
test_dataset = MNIST('../data/MNIST', train=False, download=True, transform=T.PILToTensor())
loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

# Generator
g = SimpleGenerator(input_dim=z_dim, out_features=28*28, img_shape=img_shape, batch_size=batch_size).to(device)
g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

# Disciminator
d = SimpleDiscriminator(in_features=28*28).to(device)
d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))

# Loss
adversarial_loss = nn.BCEWithLogitsLoss()

noise_set = torch.rand(batch_size, z_dim, device=device)

for epoch in range(epochs):
    running_d_loss = 0
    running_g_loss = 0

    for batch_id, (real_batch, MNIST_label) in enumerate(loader):
        d_optim.zero_grad()
        g_optim.zero_grad()

        ##### Discriminator Update #####
        d_start = time.time()

        real_batch = real_batch.to(device)
        
        z = torch.rand((batch_size, z_dim), device=g.lin1.weight.device)
        fake_batch = g(z)

        d_real = d(real_batch.type(torch.float32))
        d_fake = d(fake_batch.detach()) # Detach so gradients aren't tracked through generator

        # Traditional GAN Loss
        real_loss = adversarial_loss(d_real, torch.ones_like(d_real, device=d_real.device))
        fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake, device=d_fake.device))
        d_loss = (real_loss + fake_loss) / 2

        # Wasserstein Loss, Earth Mover Distance
        # d_loss = wass_d_loss(d_real, d_fake)

        d_loss.backward()
        d_optim.step()
        
        d_end = time.time()

        g_optim.zero_grad() # Don't want to accumulate gradients for the generator update
        d_optim.zero_grad()

        ##### Generator Update #####
        g_start = time.time()

        # Create new fake_batch
        z = torch.rand((batch_size, z_dim), device=g.lin1.weight.device)
        fake_batch = g(z)

        d_fake = d(fake_batch)

        # Traditional GAN Loss
        g_loss = adversarial_loss(d_fake, torch.ones_like(d_fake, device=d_fake.device))

        # Wasserstein Loss, or Earth Mover Distance
        # g_loss = wass_g_loss(d_fake)

        g_loss.backward()
        g_optim.step()

        g_end = time.time()

        g_optim.zero_grad()
        d_optim.zero_grad() 

        if batch_id % 100 == 0:
            save_grid(fake_batch, os.path.join(save_images_root, 'epoch{}batch{}'.format(epoch, batch_id)))
            
        print("EPOCH: {}/{} ".format(epoch+1, epochs), "Batch: {} / {}".format(batch_id, len(loader)))
        print("g_loss: ", g_loss.item())
        print("d_loss: ", d_loss.item())
        print("g_time: ", g_end - g_start)
        print("d_time: ", d_end - d_start)

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    print("Avg. g_loss: ", running_g_loss / len(loader))
    print("Avg. d_loss: ", running_d_loss / len(loader))