import argparse
import os
import time

import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
import torchvision.transforms as T

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--outdir', default='output', type=str)
    # Dataloader args
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--pin_memory', default=False, action='store_true')
    # Model args
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--features', default=64, type=int)
    # Optimizer args
    parser.add_argument('--g_lr', default=1e-3, type=float)
    parser.add_argument('--c_lr', default=1e-4, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    # Scheduler args
    # Training
    parser.add_argument('--gen_interval', default=5, type=int)
    parser.add_argument('--sample_interval', default=100, type=int)
    parser.add_argument('--gp_weight', default=10, type=float)
    args = parser.parse_args()
    return args

def init_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self, latent_dim, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(  # N x latent_dim x 1 x 1
            self._block(latent_dim, features, 7, 1, 0),  # 7x7
            self._block(features, features, 4, 2, 1),  # 14x14
            self._block(features, features, 4, 2, 1),  # 28x28
            nn.Conv2d(features, 1, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(), # N x 1 x 28 x 28
        )
        init_weights(self)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x): 
        return self.gen(x.view(*x.shape, 1, 1))

class Critic(nn.Module):
    def __init__(self, features):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(  # Nx 1 x 28 x 28
            nn.Conv2d(1, features, kernel_size=5, stride=1, padding=0, bias=False),  # 24x24
            nn.GELU(),
            self._block(features, features, 3, 2, 1),  # 12x12
            self._block(features, features, 3, 2, 1),  # 6x6
            self._block(features, features, 3, 2, 1),  # 3x3
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=0, bias=False),  # 1x1
            nn.Flatten(1),
            nn.Linear(features, 1)
        )
        init_weights(self)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x): 
        return self.critic(x)

def main():
    args = get_args()
    
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = T.Compose([T.ToTensor()])
    train_ds = MNIST(root="../data/MNIST", train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.workers, persistent_workers=(True if args.workers > 0 else False),
            pin_memory=args.pin_memory)

    gen = Generator(latent_dim=args.latent_dim, features=args.features)
    critic = Critic(features=args.features)
    gen = gen.to(device)
    critic = critic.to(device)

    g_opt = torch.optim.Adam(gen.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.c_lr, betas=(args.beta1, args.beta2))

    ncols = 16*2
    nrows = 9*2
    if args.latent_dim==2:  # Interoplate the 2d latent samples
        end_v = 3  # 3 sigma covers 99.7% of normal
        xs = torch.linspace(-end_v, end_v, steps=ncols, device=device)
        ys = torch.linspace(-end_v, end_v, steps=nrows, device=device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        fixed_z = torch.dstack([grid_x, grid_y]).view(-1, 2)
    else:
        fixed_z = torch.randn(ncols*nrows, args.latent_dim, device=device)

    video_writer = None  # Will be a cv2.VideoWriter after first sample
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0

    epochs = 0
    steps = 0
    t0 = time.time()
    time_step = t0
    while True:
        epochs += 1
        print(f"{epochs=}")

        for real_samples, _ in train_loader:
            real_samples = real_samples.to(device)

            # Critic update with the gp loss
            c_opt.zero_grad()

            # Sample the same number of fakes as the batch
            with torch.no_grad():
                z = torch.randn(real_samples.size(0), args.latent_dim, device=device)
                fake_samples = gen(z)  # Remove from backwards

            # Compute the scores
            real_scores = critic(real_samples)
            fake_scores = critic(fake_samples)
            real_mean = torch.mean(real_scores)
            fake_mean = torch.mean(fake_scores)

            # Loss with gradient penalty
            alpha = torch.rand(real_samples.size(0), 1, 1, 1, requires_grad=True, device=device)
            inter_samples = alpha * real_samples + ((1 - alpha) * fake_samples)
            inter_scores = critic(inter_samples)
            grad_outputs = torch.ones(real_samples.size(0), 1, requires_grad=True, device=device)
            gradients = torch.autograd.grad(
                outputs=inter_scores,
                inputs=inter_samples,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
            c_loss = fake_mean - real_mean + args.gp_weight * gradient_penalty

            c_loss.backward()
            c_opt.step()
            steps += 1

            # Gen update every gen_interval steps
            if steps % args.gen_interval==0:
                g_opt.zero_grad()
                c_opt.zero_grad()

                z = torch.randn(real_samples.size(0), args.latent_dim, device=device)
                fake_samples = gen(z)
                fake_scores = critic(fake_samples)
                g_loss = -torch.mean(fake_scores)

                g_loss.backward()
                g_opt.step()

                now = time.time()
                dt = now - time_step
                duration = now - t0
                print(f"{duration/60:9.2f}m {dt:5.2f}s {steps=:05d} {g_loss=:.2e} {c_loss=:.2e} C(x)={real_mean:.2e} C(G(z)={fake_mean:.2e}")
                time_step = now

            if steps % args.sample_interval==0:
                with torch.no_grad():
                    fake_save = gen(fixed_z)
                grid = make_grid(fake_save, nrow=ncols, normalize=True)
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                if video_writer is None:
                    c, h, w = grid.shape
                    video_writer = cv2.VideoWriter(f"{args.outdir}/video.avi", fourcc, fps, (w, h))
                video_writer.write(ndarr)
                im = Image.fromarray(ndarr)
                im.save(f"{args.outdir}/fake_samples_{steps:05d}.jpg")

if __name__ == "__main__":
    main()