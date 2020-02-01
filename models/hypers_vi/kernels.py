import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.kl import kl_divergence


class RBFKernel(nn.Module):
    def __init__(self, in_size):
        super().__init__()

        # Variational Parameters (log lengthscale and log scale factor) 
        log_init = torch.tensor(.5).log() * torch.ones(in_size + 1) \
                   + .05 * torch.randn(in_size + 1)
        self.log_mean = nn.Parameter(log_init)
        self.log_logvar = nn.Parameter(-2 * torch.ones(in_size + 1))

        self.register_buffer('prior_log_mean', torch.zeros_like(self.log_mean))
        self.register_buffer('prior_logvar', torch.zeros_like(self.log_logvar))

    def compute_kuu(self, x, kern_samples):
        x = x.unsqueeze(0)
        log_ls = kern_samples[..., :-1].unsqueeze(1).unsqueeze(1)
        log_sf = kern_samples[..., -1:].unsqueeze(1).unsqueeze(1)
        sx = x / torch.exp(log_ls)
        xx = torch.einsum("koad,kobd->koab", sx, sx)
        rx = xx.diagonal(offset=0, dim1=-2, dim2=-1)
        dist2 = -2.0 * xx + rx.unsqueeze(2) + rx.unsqueeze(3)
        return torch.exp(log_sf * 2.0) * torch.exp(-0.5 * dist2)

    def compute_kfu(self, x, z, kern_samples):
        x = x.unsqueeze(0).unsqueeze(0)
        z = z.unsqueeze(0)
        log_ls = kern_samples[..., :-1].unsqueeze(1).unsqueeze(1)
        log_sf = kern_samples[..., -1:].unsqueeze(1).unsqueeze(1)
        sx = x / torch.exp(log_ls)
        sz = z / torch.exp(log_ls)
        xz = torch.einsum("kand,komd->konm", sx, sz)
        xx = sx.pow(2).sum(3)
        zz = sz.pow(2).sum(3)
        dist2 = xx.unsqueeze(3) + zz.unsqueeze(2) - 2.0 * xz
        return torch.exp(log_sf * 2.0) * torch.exp(-0.5 * dist2)

    def compute_kff_diag(self, kern_samples):
        log_sf = kern_samples[..., -1:].unsqueeze(1)
        return torch.exp(log_sf * 2.0)

    def sample_hypers(self, n_hypers):
        log_dist = dist.Normal(self.log_mean, self.log_logvar.exp().sqrt())
        log_hypers = log_dist.rsample(torch.Size([n_hypers]))
        return log_hypers

    def kl_hypers(self):
        var_dist = dist.Normal(self.log_mean, self.log_logvar.exp().sqrt())
        prior_dist = dist.Normal(self.prior_log_mean, self.prior_logvar.exp().sqrt())
        total_kl = kl_divergence(var_dist, prior_dist).sum(dim=0)
        return total_kl
