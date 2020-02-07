import numpy as np
import torch
import torch.nn as nn
from torch import triangular_solve
from ..utils import kl_diagonal_gaussian
import pdb


class RBFKernel(nn.Module):
    def __init__(self, input_dim):
        super(RBFKernel, self).__init__()
        log_ls = (
            np.log(0.5) * np.ones([input_dim])
            + np.random.randn(input_dim) * 0.05
        )
        self.input_dim = input_dim
        self.log_ls_mean = nn.Parameter(torch.from_numpy(log_ls).float())
        self.log_sf_mean = nn.Parameter(np.log(0.5) * torch.ones([1]))
        self.log_ls_logvar = nn.Parameter(-2 * torch.ones([input_dim]))
        self.log_sf_logvar = nn.Parameter(-2 * torch.ones([1]))

        self.register_buffer('prior_mean', torch.Tensor([0.0]))
        self.register_buffer('prior_var', torch.Tensor([1.0]))

    def compute_kuu(self, x, kern_samples):
        x = x.unsqueeze(0)
        log_ls = kern_samples[0].unsqueeze(1).unsqueeze(1)
        log_sf = kern_samples[1].unsqueeze(1).unsqueeze(1)
        sx = x / torch.exp(log_ls)
        xx = torch.einsum("koad,kobd->koab", sx, sx)
        rx = xx.diagonal(offset=0, dim1=-2, dim2=-1)
        dist2 = -2.0 * xx + rx.unsqueeze(2) + rx.unsqueeze(3)
        return torch.exp(log_sf * 2.0) * torch.exp(-0.5 * dist2)

    def compute_kfu(self, x, z, kern_samples):
        x = x.unsqueeze(0).unsqueeze(0)
        z = z.unsqueeze(0)
        log_ls = kern_samples[0].unsqueeze(1).unsqueeze(1)
        log_sf = kern_samples[1].unsqueeze(1).unsqueeze(1)
        sx = x / torch.exp(log_ls)
        sz = z / torch.exp(log_ls)
        xz = torch.einsum("kand,komd->konm", sx, sz)
        xx = sx.pow(2).sum(3)
        zz = sz.pow(2).sum(3)
        dist2 = xx.unsqueeze(3) + zz.unsqueeze(2) - 2.0 * xz
        return torch.exp(log_sf * 2.0) * torch.exp(-0.5 * dist2)

    def compute_kff_diag(self, kern_samples):
        log_sf = kern_samples[1].unsqueeze(1)
        return torch.exp(log_sf * 2.0)

    def sample_hypers(self, no_kern_samples):
        log_sf_std = self.log_sf_logvar.exp().sqrt()
        log_ls_std = self.log_ls_logvar.exp().sqrt()
        eps_log_ls = torch.randn(no_kern_samples, self.input_dim, device=log_ls_std.device)
        log_ls = eps_log_ls * log_ls_std + self.log_ls_mean
        eps_log_sf = torch.randn(no_kern_samples, 1, device=log_sf_std.device)
        log_sf = eps_log_sf * log_sf_std + self.log_sf_mean
        return log_ls, log_sf

    def compute_kl(self):
        kl_sf = kl_diagonal_gaussian(
            self.log_sf_mean,
            torch.exp(self.log_sf_logvar),
            self.prior_mean,
            self.prior_var,
        )
        kl_ls = kl_diagonal_gaussian(
            self.log_ls_mean,
            torch.exp(self.log_ls_logvar),
            self.prior_mean,
            self.prior_var,
        )
        return kl_sf.sum() + kl_ls.sum()