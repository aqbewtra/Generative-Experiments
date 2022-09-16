import torch

# Wasserstein Loss, or Earth Mover Distance
# https://arxiv.org/abs/1701.07875

def wass_d_loss(d_real, d_fake):
    d_loss = -torch.mean(d_real) + torch.mean(d_fake)
    return d_loss

def wass_g_loss(d_fake):
    g_loss = -torch.mean(d_fake)
    return g_loss
