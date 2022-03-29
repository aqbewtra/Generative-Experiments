from numpy import dtype
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision.transforms as T

from simple_discriminator import SimpleDiscriminator
from simple_generator import SimpleGenerator

from PIL import Image

# Task Parameters
img_shape = (1, 28, 28)
z_shape = (10, 10, 10)

# Hyperparameters
batch_size = 16
lr = .01
b1 = .5
b2 = .999

# Dataset
train_dataset = MNIST('../data/MNIST', train=True, download=False, transform=T.PILToTensor())
test_dataset = MNIST('../data/MNIST', train=False, download=False, transform=T.PILToTensor())
loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

# Generator
g = SimpleGenerator(input_shape=z_shape, output_shape=img_shape, batch_size=batch_size)
g_optim = torch.optim.Adam(g.parameters(), lr=lr, betas=(b1, b2))

# Disciminator
d = SimpleDiscriminator(img_shape=img_shape)
d_optim = torch.optim.Adam(d.parameters(), lr=lr, betas=(b1, b2))

# Loss
adversarial_loss = nn.BCEWithLogitsLoss()

for batch_id, (x, MNIST_label) in enumerate(loader):
    print("Batch: {} / {}".format(batch_id, len(loader)))
    g_optim.zero_grad()
    d_optim.zero_grad()

    z = torch.rand((batch_size, *z_shape))
    x_gen = g(z)

    pred_real = d(x.to(dtype=torch.float32))
    pred_gen = d(x_gen)

    y_real = torch.ones((batch_size,))
    y_fake = torch.zeros((batch_size,))

    g_loss = adversarial_loss(pred_gen, y_real)
    d_loss = (adversarial_loss(pred_gen, y_fake) + adversarial_loss(pred_real, y_real)) / 2

    print("g_loss: ", g_loss.item())
    print("d_loss: ", d_loss.item())

    
    g_loss.backward(retain_graph=True)
    d_loss.backward(retain_graph=True)

    g_optim.step()
    d_optim.step()

imgs = [x_gen[i].squeeze_() for i in range(batch_size)]
import matplotlib.pyplot as plt
plt.imshow(torch.stack([imgs[0] for i in range(3)], dim=2).detach())
for i, img in enumerate(imgs):
    T.ToPILImage()(img).save('samples/{}.png'.format(i))
plt.show()


