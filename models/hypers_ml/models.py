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

    def forward(self, x):
        mean = self.u_mean
        xu = self.xu
        M = self.M
        u_tril_mat = self.u_tril_mat
        u_mean = self.u_mean
        prior_tril = self.prior_tril
        prior_mean = self.prior_mean

        # compute kfu, kuf
        kfu = self.kernel.compute_kfu(x, xu)
        kuf = kfu.permute([0, 2, 1])

        # compute marginal means and variances
        # first, v1 = Kff, v2 = Kfu Kuuinv Kuf,
        # v3 = Kfu Kuuinv Su Kuuinv Kuf, and v = v1 - v2 + v3 + sn2
        LKinvKuf = triangular_solve(kuf, prior_tril, upper=False)[0]
        LKinvLSu = self.LKinvLSu
        vec1 = LKinvKuf
        vec2 = torch.bmm(LKinvLSu.permute([0, 2, 1]), LKinvKuf)
        v2 = (vec1 * vec1).sum(1)
        v3 = (vec2 * vec2).sum(1)
        v1 = self.kernel.compute_kff_diag()
        v = v1 - v2 + v3

        # mean m = Kfu Kuuinv Mu or m = m = Kfu Kuuinv u_sample
        LKinvMu = self.LKinvMu
        m = (LKinvKuf * LKinvMu).sum(1)
        m = m.permute([1, 0])
        v = v.permute([1, 0])
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

        # compute kuu
        kuu = self.kernel.compute_kuu(self.xu)
        kuu = kuu + self.jitter
        self.prior_tril = torch.cholesky(kuu)
        self.prior_mean = torch.zeros([self.output_dim, self.M, 1])
        mean_diff = self.u_mean - self.prior_mean
        self.LKinvMu = triangular_solve(
            mean_diff, self.prior_tril, upper=False
        )[0]
        self.LKinvLSu = triangular_solve(
            tril_mat, self.prior_tril, upper=False
        )[0]

    def loss(self, batch_state, batch_target, no_samples=10):
        batch_state = batch_state
        batch_target = batch_target
        # propagate and compute the expected log likelihood
        self.forward_prep()
        f_means, f_vars = self.forward(batch_state)
        lik = self.likelihood.loss_fn(
            f_means, f_vars, batch_target, no_samples
        )
        # next compute the kl divergence
        kl_term = kl_full_gaussian(
            self.u_mean, self.u_tril_mat, self.prior_mean, self.prior_tril
        )
        return lik, kl_term

    def predict(self, x, pred_y=False, no_samples=100):
        res = self.forward(x)
        if not pred_y:
            return res
        else:
            return self.likelihood.forward(*res, no_samples)
