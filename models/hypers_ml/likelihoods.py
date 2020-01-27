import numpy as np
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class Gaussian(nn.Module):
    def __init__(self, noise_log_var_init=-4):
        super(Gaussian, self).__init__()
        noise_log_var = nn.Parameter(
            noise_log_var_init * torch.ones([1]), requires_grad=True
        )
        self.noise_log_var = noise_log_var

    def loss_fn(self, ymean, yvar, y):
        y = y.permute([1, 0])
        const_term = 0.5 * np.log(2 * np.pi) + 0.5 * torch.log(yvar)
        diff_term = 0.5 * (ymean - y) ** 2 / yvar
        return torch.mean(const_term + diff_term, dim=1).sum()

    def log_output_fn(self, x):
        return x

    def forward(self, fmean, fvar):
        noise_variance = torch.exp(self.noise_log_var)
        return fmean, fvar + noise_variance


class MulticlassSoftmax(nn.Module):
    def __init__(self):
        super(MulticlassSoftmax, self).__init__()

    def loss_fn(self, fmean, fvar, y, no_samples):
        # draw samples
        n, dout = fmean.shape
        eps = torch.randn([n, dout, no_samples])
        fsamples = fmean.unsqueeze(-1) + fvar.unsqueeze(-1) * eps
        logsoftmax = F.log_softmax(fsamples, dim=1)
        y = y.unsqueeze(-1).repeat([1, no_samples])
        loss = F.nll_loss(logsoftmax, y, reduction="none")
        avg_loss = loss.mean(-1)
        return avg_loss.sum()

    def log_output_fn(self, x):
        return x

    def forward(self, fmean, fvar, no_samples):
        n, dout = fmean.shape
        eps = torch.randn([n, dout, no_samples])
        fsamples = fmean.unsqueeze(-1) + fvar.unsqueeze(-1) * eps
        logsoftmax = F.log_softmax(fsamples, dim=1)
        return logsoftmax
