import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.kl import kl_divergence


class RBFKernel(nn.Module):
    def __init__(self, in_size, prior_log_mean=None, prior_log_logvar=None, map_est=False):
        super().__init__()

        self.map_est = map_est

        # Variational Parameters (log lengthscale and log scale factor)
        log_init = torch.tensor(.5).log() * torch.ones(in_size + 1) \
                   + .05 * torch.randn(in_size + 1)
        self.log_mean = nn.Parameter(log_init)
        self.log_logvar = nn.Parameter(-2 * torch.ones(in_size + 1))

        self.register_buffer('prior_log_mean', prior_log_mean if prior_log_mean is not None \
                                               else torch.zeros_like(self.log_mean))
        self.register_buffer('prior_log_logvar', prior_log_logvar if prior_log_logvar is not None \
                                                 else torch.zeros_like(self.log_logvar))

    def compute(self, kern_samples, x, y=None):
        '''
        Generic batch kernel evaluation. Send
        y = None for efficient re-use of computations
        in case x = y.

        Arguments:
            kern_samples: n_hypers x 3
            x: ...batches x M x D
            y: ...batches x N x D, if None, assumed equals x

        Returns:
            n_hypers x ...batches x M x N
        '''
        n_hypers = kern_samples.size(0)
        kern_samples = kern_samples.view(n_hypers, 1, *([1] * len(x.shape[:-2])), -1)

        sigma = kern_samples[..., :-1].exp()
        gamma2 = (kern_samples[..., -1:] * 2.).exp()

        sx = x.unsqueeze(0) / sigma
        xx = torch.einsum('...ji,...ki->...jk', sx, sx)

        if y is None:
            yy = xy = xx
        else:
            sy = y.unsqueeze(0) / sigma
            yy = torch.einsum('...ji,...ki->...jk', sy, sy)
            xy = torch.einsum('...ji,...ki->...jk', sx, sy)

        dnorm2 = - 2. * xy + xx.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) + yy.diagonal(dim1=-2, dim2=-1).unsqueeze(-2)

        return gamma2 * (-.5 * dnorm2).exp()

    def compute_diag(self, kern_samples):
        gamma2 = (kern_samples[..., -1:] * 2.).exp().unsqueeze(-2)
        return gamma2

    def sample_hypers(self, n_hypers):
        if self.map_est:
            return self.log_mean.unsqueeze(0)

        log_dist = dist.Normal(self.log_mean, self.log_logvar.exp().sqrt())
        log_hypers = log_dist.rsample(torch.Size([n_hypers]))
        return log_hypers

    def kl_hypers(self):
        if self.map_est:
            return torch.tensor(0.0, device=self.log_mean.device)

        var_dist = dist.Normal(self.log_mean, self.log_logvar.exp().sqrt())
        prior_dist = dist.Normal(self.prior_log_mean, self.prior_log_logvar.exp().sqrt())
        total_kl = kl_divergence(var_dist, prior_dist).sum(dim=0)
        return total_kl