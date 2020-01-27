import numpy as np
import torch
import torch.nn as nn
from torch import triangular_solve
import pdb


class RBFKernel(nn.Module):
    def __init__(self, input_dim):
        super(RBFKernel, self).__init__()
        log_ls = (
            np.log(0.5) * np.ones([input_dim])
            + np.random.randn(input_dim) * 0.05
        )
        self.log_ls = nn.Parameter(torch.from_numpy(log_ls).float())
        self.log_sf = nn.Parameter(np.log(0.5) * torch.ones([1]))

    def compute_kuu(self, x):
        sx = x / torch.exp(self.log_ls)
        sxt = sx.permute([0, 2, 1])
        xx = torch.bmm(sx, sxt)
        rx = xx.diagonal(offset=0, dim1=-2, dim2=-1)
        dist2 = -2.0 * xx + rx.unsqueeze(1) + rx.unsqueeze(2)
        return torch.exp(self.log_sf * 2.0) * torch.exp(-0.5 * dist2)

    def compute_kfu(self, x, z):
        log_ls = self.log_ls
        log_sf = self.log_sf
        sx = x / torch.exp(log_ls)
        sz = z / torch.exp(log_ls)
        xz = torch.einsum("nd,kmd->knm", sx, sz)
        xx = sx.pow(2).sum(1)
        zz = sz.pow(2).sum(2)
        dist2 = xx.unsqueeze(1) + zz.unsqueeze(1) - 2.0 * xz
        return torch.exp(log_sf * 2.0) * torch.exp(-0.5 * dist2)

    def compute_kff_diag(self, x=None):
        return torch.exp(self.log_sf * 2.0)
