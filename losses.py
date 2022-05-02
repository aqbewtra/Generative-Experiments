import torch

def gan_loss(x_real, x_fake, discriminator):
    dx_real = discriminator(x_real)
    dx_fake = discriminator(x_fake)
    d_loss = -torch.mean(torch.sigmoid(dx_real) + torch.sigmoid(1 - dx_fake))
    g_loss = -torch.mean(torch.sigmoid(dx_fake))
    return d_loss, g_loss