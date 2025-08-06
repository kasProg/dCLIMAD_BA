import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- UNet Backbone for Diffusion ---
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU())
        self.encoder2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU())
        self.decoder1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t_emb):
        # Embed timestep (expand to spatial)
        t = t_emb.view(-1, 1, 1, 1).expand(x.shape[0], 1, *x.shape[2:])
        x = torch.cat([x, t], dim=1)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.decoder1(x2)
        return self.out(x3 + x1)  # Skip connection

# --- Noise schedule ---
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

# --- DDPM Training Step ---
class DiffusionDownscaler:
    def __init__(self, unet, T=1000):
        self.unet = unet
        self.T = T
        self.betas = get_beta_schedule(T)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        sqrt_alpha_hat = self.alpha_hat[t].sqrt().view(-1, 1, 1, 1).to(x0.device)
        sqrt_one_minus = (1 - self.alpha_hat[t]).sqrt().view(-1, 1, 1, 1).to(x0.device)
        return sqrt_alpha_hat * x0 + sqrt_one_minus * noise

    def train_step(self, x_coarse, x_fine):
        device = x_fine.device
        batch_size = x_fine.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=device)
        noise = torch.randn_like(x_fine)

        x_t = self.q_sample(x_fine, t, noise)

        # Conditioning by concatenating x_coarse
        x_cond = torch.cat([x_coarse, x_t], dim=1)
        pred_noise = self.unet(x_cond, t.float() / self.T)

        return F.mse_loss(pred_noise, noise)
