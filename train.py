from numpy import dtype
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

from simple_discriminator import SimpleDiscriminator
from simple_generator import SimpleGenerator
from losses import gan_loss

from PIL import Image
import matplotlib.pyplot as plt


# Task Parameters
img_shape = (1, 28, 28)
z_dim = 100

# Hyperparameters
epochs = 30
batch_size = 2
lr = 4e-3
b1 = .5
b2 = .999

# Dataset
train_dataset = MNIST('../data/MNIST', train=True, download=False, transform=T.PILToTensor())
test_dataset = MNIST('../data/MNIST', train=False, download=False, transform=T.PILToTensor())
loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

# Generator
g = SimpleGenerator(input_dim=z_dim, out_features=28*28, img_shape=img_shape, batch_size=batch_size)
g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

# Disciminator
d = SimpleDiscriminator(in_features=28*28)
d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))

# Loss
adversarial_loss = nn.BCEWithLogitsLoss()

noise_set = torch.rand(batch_size, z_dim)

for epoch in range(epochs):
    running_d_loss = 0
    running_g_loss = 0

    for batch_id, (x, MNIST_label) in enumerate(loader):
        d_optim.zero_grad()
        g_optim.zero_grad()

        z = torch.rand((batch_size, z_dim))
        x_gen = g(z)

        d_real = d(x.type(torch.float32))
        d_fake = d(x_gen)

        real_loss = adversarial_loss(d_real, torch.ones_like(d_real))
        fake_loss = adversarial_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward(retain_graph=True)
        d_optim.step()

        g_optim.zero_grad() # Don't want to accumulate gradients for the generator update
        d_optim.zero_grad()

        d_fake = d(x_gen)
        g_loss = adversarial_loss(d_fake, torch.ones_like(d_fake))

        g_loss.backward()
        g_optim.step()

        g_optim.zero_grad()
        d_optim.zero_grad() 

        if True: #batch_id % 100 == 0:
            print("EPOCH: {}/{} ".format(epoch+1, epochs), "Batch: {} / {}".format(batch_id, len(loader)))
            print("g_loss: ", g_loss.item())
            print("d_loss: ", d_loss.item())

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()

    print("Avg. g_loss: ", running_g_loss / len(loader))
    print("Avg. d_loss: ", running_d_loss / len(loader))
    imgs = [x_gen[i].squeeze_() for i in range(batch_size)]
    plt.imshow(torch.stack([imgs[0] for i in range(3)], dim=2).detach())
    for i, img in enumerate(imgs):
        T.ToPILImage()(img).save('samples/{}.png'.format(i))
    plt.show()