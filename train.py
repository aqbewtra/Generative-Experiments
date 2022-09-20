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

from PIL import Image
import matplotlib.pyplot as plt

save_images_root = 'samples'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Task Parameters
img_shape = (1, 28, 28)
z_dim = 100
g_features = 512
g_out_features = 28*28

# Hyperparameters
epochs = 30
batch_size = 512 if torch.cuda.is_available() else 128
g_lr = 4e-2
d_lr = 4e-6
g_b1 = .5
g_b2 = .999
d_b1 = .4
d_b2 = .9

# Dataset
train_dataset = MNIST('../data/MNIST', train=True, download=True, transform=T.PILToTensor())
test_dataset = MNIST('../data/MNIST', train=False, download=True, transform=T.PILToTensor())
loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

# Generator
g = SimpleGenerator(input_dim=z_dim, features=g_features, out_features=g_out_features, img_shape=img_shape).to(device)
g_optim = torch.optim.Adam(g.parameters(), lr=g_lr, betas=(g_b1, g_b2))

# Disciminator
d = SimpleDiscriminator(in_features=28*28).to(device)
d_optim = torch.optim.Adam(d.parameters(), lr=d_lr, betas=(d_b1, d_b2))

# Loss
adversarial_loss = nn.BCEWithLogitsLoss()

noise_set = torch.rand(batch_size, z_dim, device=device)

for epoch in range(epochs):
    running_d_loss = 0
    running_g_loss = 0

    for batch_id, (real_batch, MNIST_label) in enumerate(loader):
        d.zero_grad()
        g.zero_grad()

        ##### Discriminator Update #####
        d_start = time.time()

        real_batch = real_batch.to(device)
        
        z = torch.rand((batch_size, z_dim), device=g.lin1.weight.device)
        fake_batch = g(z)

        d_real = d(real_batch.type(torch.float32))
        d_fake = d(fake_batch.detach()) # Detach so gradients aren't tracked through generator

        # Traditional GAN Loss
        d_real_loss = adversarial_loss(d_real, torch.ones_like(d_real, device=d_real.device))
        d_fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake, device=d_fake.device))
        d_loss = (d_real_loss + d_fake_loss) / 2

        
        d_loss.backward()
        d_optim.step()
        
        d_end = time.time()

        g.zero_grad() # Don't want to accumulate gradients for the generator update
        d.zero_grad()

        ##### Generator Update #####
        g_start = time.time()

        # Create new fake_batch
        z = torch.rand((batch_size, z_dim), device=g.lin1.weight.device)
        fake_batch = g(z)

        d_fake = d(fake_batch)

        # Traditional GAN Loss
        g_loss = adversarial_loss(d_fake, torch.ones_like(d_fake, device=d_fake.device))

        g_loss.backward()
        g_optim.step()

        g_end = time.time()

        # g.zero_grad()
        # d.zero_grad() 

        if batch_id % 100 == 0:
            save_grid(fake_batch.cpu(), os.path.join(save_images_root, 'epoch{}batch{}'.format(epoch, batch_id)))
            print("EPOCH: {}/{} ".format(epoch+1, epochs), "Batch: {} / {}".format(batch_id, len(loader)))
            print("g_loss: ", g_loss.item())
            print("d_loss: ", d_loss.item())
            print("g_time: ", g_end - g_start)
            print("d_time: ", d_end - d_start)

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    print("Avg. g_loss: ", running_g_loss / len(loader))
    print("Avg. d_loss: ", running_d_loss / len(loader))