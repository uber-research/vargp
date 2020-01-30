import sys
import numpy as np
import torch
import torch.nn as nn
from ..utils import kl_full_gaussian
from torch import triangular_solve
import pdb


class SparseGP(nn.Module):
    def __init__(
        self, input_dim, output_dim, no_inducing_points, kernel, likelihood
    ):
        super(SparseGP, self).__init__()
        # number of inducing points, assuming same for all outputs
        self.no_inducing_points = self.M = M = no_inducing_points
        # input and output dimensions
        self.input_dim = self.Din = Din = input_dim
        self.output_dim = self.Dout = Dout = output_dim
        # store kernel and likelihood, zero mean for now
        self.kernel = kernel
        self.likelihood = likelihood

        # create mean parameter
        self.u_mean = nn.Parameter(torch.Tensor(Dout, M, 1).normal_(0, 0.5))
        # create covariance parameters
        # use full matrix and multiply by a lower mask for now
        # TODO: need a proper way to handle this
        init = np.zeros((M, M))
        init[np.diag_indices(M)] = -4.0
        init = np.repeat(init[np.newaxis, :, :], Dout, 0)
        self.u_tril = nn.Parameter(torch.from_numpy(init).float())
        # create inducing input parameters
        # assuming separate inducing points for different outputs
        max_val = 3
        xu = np.random.random((Dout, M, Din)) * max_val - max_val / 2.0
        self.xu = nn.Parameter(torch.from_numpy(xu).float())

        # some constants
        self.u_lower_mask = torch.ones([M, M]).tril(0)
        self.jitter = 1e-4 * torch.eye(M)

    def forward(self, x, no_kern_samples, return_info=False):
        xu = self.xu
        M = self.M
        dout = self.output_dim
        u_tril_mat = self.u_tril_mat
        u_mean = self.u_mean
        hyper_samples = self.kernel.sample_hypers(no_kern_samples)
        # compute kuu
        kuu = self.kernel.compute_kuu(self.xu, hyper_samples)
        kuu = kuu + self.jitter
        prior_tril = torch.cholesky(kuu)
        prior_mean = torch.zeros([no_kern_samples, dout, M, 1])
        mean_diff = u_mean - prior_mean
        LKinvMu = triangular_solve(mean_diff, prior_tril, upper=False)[0]
        LKinvLSu = triangular_solve(u_tril_mat, prior_tril, upper=False)[0]
        # compute kfu, kuf
        kfu = self.kernel.compute_kfu(x, xu, hyper_samples)
        kuf = kfu.permute([0, 1, 3, 2])

        # compute marginal means and variances
        # first, v1 = Kff, v2 = Kfu Kuuinv Kuf,
        # v3 = Kfu Kuuinv Su Kuuinv Kuf, and v = v1 - v2 + v3 + sn2
        LKinvKuf = triangular_solve(kuf, prior_tril, upper=False)[0]
        vec1 = LKinvKuf
        vec2 = torch.einsum("koab,koam->kobm", LKinvLSu, LKinvKuf)
        v2 = (vec1 * vec1).sum(2)
        v3 = (vec2 * vec2).sum(2)
        v1 = self.kernel.compute_kff_diag(hyper_samples)
        v = v1 - v2 + v3

        # mean m = Kfu Kuuinv Mu or m = m = Kfu Kuuinv u_sample
        m = (LKinvKuf * LKinvMu).sum(2)
        m = m.permute([0, 2, 1])
        v = v.permute([0, 2, 1])
        if return_info:
            info = [prior_mean, prior_tril]
            return m, v, info
        else:
            return m, v

    def forward_prep(self):
        # get q(u) mean and covariance, and gp hypers
        tril_mat = self.u_tril.mul(self.u_lower_mask)
        tril_diag = torch.diagonal(tril_mat, offset=0, dim1=-2, dim2=-1)
        diag_mat = torch.diag_embed(tril_diag, offset=0, dim1=-2, dim2=-1)
        tril_diag_exp = torch.exp(tril_diag)
        exp_diag_mat = torch.diag_embed(
            tril_diag_exp, offset=0, dim1=-2, dim2=-1
        )
        self.u_tril_mat = tril_mat = tril_mat - diag_mat + exp_diag_mat

    def loss(
        self, batch_state, batch_target, no_kern_samples=5, no_func_samples=10
    ):
        # propagate and compute the expected log likelihood
        self.forward_prep()
        f_means, f_vars, info = self.forward(
            batch_state, no_kern_samples, return_info=True
        )
        lik = self.likelihood.loss_fn(
            f_means, f_vars, batch_target, no_func_samples
        )
        # next compute the kl divergence for inducing points
        prior_mean, prior_tril = info[0], info[1]
        s = prior_mean.shape
        prior_mean = prior_mean.reshape([s[0] * s[1], s[2], s[3]])
        s = prior_tril.shape
        prior_tril = prior_tril.reshape([s[0] * s[1], s[2], s[3]])
        kl_u = kl_full_gaussian(
            self.u_mean.repeat([no_kern_samples, 1, 1]),
            self.u_tril_mat.repeat([no_kern_samples, 1, 1]),
            prior_mean,
            prior_tril,
        )
        kl_u = kl_u / no_kern_samples

        # next compute the kl divergence for kernel hypers
        kl_hypers = self.kernel.compute_kl()
        kl_term = kl_hypers + kl_u

        return lik, kl_term

    def predict(
        self, x, pred_y=False, no_kern_samples=10, no_func_samples=100
    ):
        res = self.forward(x, no_kern_samples)
        if not pred_y:
            return res
        else:
            return self.likelihood.forward(*res, no_func_samples)
