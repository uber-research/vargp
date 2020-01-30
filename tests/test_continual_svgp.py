'''
NOTE(sanyam): Will gradually modularize this code.
'''
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.hypers_vi.kernels import RBFKernel
from data_utils import get_toy_cla_four


def vec2tril(vec, m):
  batch_shape = vec.shape[:-1]

  idx = torch.tril_indices(m, m)

  tril = torch.zeros(*batch_shape, m, m)
  tril[..., idx[0], idx[1]] = vec

  return tril


class ContinualSVGP(nn.Module):
  '''
  Arguments:
    z: Initial inducing points (M x in_size)
  '''
  def __init__(self, z, kernel, likelihood, n_hypers=1):
    super().__init__()

    self.M = z.size(-2)
    self.in_size = z.size(-1)
    self.out_size = z.size(0)

    self.kernel = kernel
    self.n_hypers = n_hypers
    self.likelihood = likelihood

    ## TODO(sanyam): Previous inducing points for continual learning

    # New inducing points
    self.z = nn.Parameter(z.detach())

    # Variational parameters for q(u)
    self.u_mean = nn.Parameter(torch.Tensor(self.out_size, self.M, 1).normal_(0., .5))
    self.u_tril_vec = nn.Parameter(torch.ones(self.out_size, (self.M * (self.M + 1)) // 2))

    self.jitter = 1e-4 * torch.eye(self.M)

  def forward(self, x):
    '''
    effectively the marginal p(f|X,α) of the joint p(f,u|X,α)
    approximated by q(f,u|X,α) = p(f|X,u,α)q(u|α) 
    for k samples of α. also decompose across x so just
    variance instead of covariance
    '''
    u_tril = vec2tril(self.u_tril_vec, self.M)
    hyper_samples = self.kernel.sample_hypers(self.n_hypers)

    kfu = self.kernel.compute_kfu(x, self.z, hyper_samples)
    kuf = kfu.permute(0, 1, 3, 2)
    kuu = self.kernel.compute_kuu(self.z, hyper_samples)
    Lkuu = torch.cholesky(kuu + self.jitter, upper=False)
    LKinvu, _ = torch.triangular_solve(self.u_mean, Lkuu, upper=False)
    LKinvKuf, _ = torch.triangular_solve(kuf, Lkuu, upper=False)

    kff_diag = self.kernel.compute_kff_diag(hyper_samples)
    diag1 = (LKinvKuf**2).sum(dim=-2)
    LKinvLs, _ = torch.triangular_solve(u_tril, Lkuu)
    vec2 = torch.einsum('...ij,...ik->...jk', LKinvLs, LKinvKuf)
    diag2 = (vec2**2).sum(dim=-2)

    pred_mu = torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvu).squeeze(-1)  # k x out_size x B
    pred_var = kff_diag - diag1 + diag2

    return pred_mu, pred_var

  def loss(self, x, y):
    pred_mu, pred_var = self(x)
    pred_y = self.likelihood(pred_mu, pred_var)

    pred_y = pred_y.permute(3, 2, 0, 1)
    target_y = y.unsqueeze(-1).unsqueeze(-1).expand(-1, *pred_y.shape[-2:])

    lik_loss = F.nll_loss(pred_y, target_y, reduction='none').mean(dim=-1).mean(dim=-1).sum(dim=0)

    return lik_loss


class MulticlassSoftmax(nn.Module):
  def __init__(self, n_f=1):
    super().__init__()

    self.n_f = n_f

  def forward(self, mu, var):
    n_hypers, out_size, B = mu.shape

    eps = torch.randn(n_hypers, self.n_f, out_size, B)
    f_samples = mu.unsqueeze(1) + var.sqrt().unsqueeze(1) * eps

    y_f = F.log_softmax(f_samples, dim=-2)
    return y_f


def main():
  x_train, y_train, *_ = get_toy_cla_four()
  x_train = torch.from_numpy(x_train).float()
  y_train_one_hot = torch.from_numpy(y_train).float()
  y_train = y_train_one_hot.argmax(dim=-1)

  z = (2. * torch.rand(4, 5, x_train.size(-1)) - 1.) * 3.
  kernel = RBFKernel(x_train.size(-1))
  likelihood = MulticlassSoftmax(n_f=3)
  gp = ContinualSVGP(z, kernel, likelihood, n_hypers=5)


if __name__ == "__main__":
  main()
