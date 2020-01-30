'''
NOTE(sanyam): Will gradually modularize this code.
'''
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from models.hypers_vi.kernels import RBFKernel
from data_utils import get_toy_cla_four


def vec2tril(vec, m):
  '''
  Arguments:
    vec: K x ((m * (m + 1)) // 2)
    m: integer

  Returns:
    Batch of lower triangular matrices

    tril: K x m x m
  '''
  batch_shape = vec.shape[:-1]

  idx = torch.tril_indices(m, m)

  tril = torch.zeros(*batch_shape, m, m, device=vec.device)
  tril[..., idx[0], idx[1]] = vec

  return tril


class MulticlassSoftmax(nn.Module):
  def __init__(self, n_f=1):
    super().__init__()

    self.n_f = n_f

  def forward(self, mu, var):
    '''
    Arguments:
      mu: n_hypers x out_size x B
      var: n_hypers x out_size x B

    Returns:
      Predictions for function samples.

      y_f: n_hypers x n_f x out_size x B
    '''
    n_hypers, out_size, B = mu.shape

    eps = torch.randn(n_hypers, self.n_f, out_size, B)
    f_samples = mu.unsqueeze(1) + var.sqrt().unsqueeze(1) * eps

    y_f = F.log_softmax(f_samples, dim=-2)

    return y_f

  def loss(self, pred_mu, pred_var, y):
    '''
    Arguments:
      pred_mu: n_hypers x out_size x B
      pred_var: n_hypers x out_size x B
      y: B

    Returns:
      Total loss scalar value.
    '''
    pred_y = self(pred_mu, pred_var).permute(3, 2, 0, 1)
    target_y = y.unsqueeze(-1).unsqueeze(-1).expand(-1, *pred_y.shape[-2:])

    lik_loss = F.nll_loss(pred_y, target_y,
                          reduction='none').mean(dim=-1).mean(dim=-1).sum(dim=0)
    return lik_loss


class ContinualSVGP(nn.Module):
  '''
  Arguments:
    z: Initial inducing points out_size x M x in_size
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

  def forward(self, x, info=False):
    '''
    effectively the marginal p(f|X,α) of the joint p(f,u|X,α)
    approximated by q(f,u|X,α) = p(f|X,u,α)q(u|α) 
    for k samples of α.
    
    Arguments:
      x: B x in_size

    Returns:
      Output distributions for n_hypers samples of hyperparameters.
      The output contains only diagonal of the full covariance.
 
      pred_mu: n_hypers x out_size x B
      pred_var: n_hypers x out_size x B
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

    if info:
      info = dict(hyper_samples=hyper_samples, u_tril=u_tril, Lkuu=Lkuu)
      return pred_mu, pred_var, info

    return pred_mu, pred_var

  def loss(self, x, y):
    ## Expected Negative Log Likelihood
    pred_mu, pred_var, info = self(x, info=True)
    nll = self.likelihood.loss(pred_mu, pred_var, y)

    ## TODO(sanyam): Add continual variational/prior distribution
    var_mean = self.u_mean.squeeze(-1).unsqueeze(0).expand(self.n_hypers, -1, -1)
    var_tril = info.get('u_tril').unsqueeze(0).expand(self.n_hypers, -1, -1, -1)
    var_dist = dist.MultivariateNormal(var_mean, scale_tril=var_tril)

    prior_mean = torch.zeros_like(self.u_mean).squeeze(-1).unsqueeze(0).expand(self.n_hypers, -1, -1)
    prior_tril = info.get('Lkuu')
    prior_dist = dist.MultivariateNormal(prior_mean, scale_tril=prior_tril)

    kl_u = kl_divergence(var_dist, prior_dist).sum(dim=-1).mean(dim=0)

    ## TODO(sanyam): add non-standard hyperprior
    kl_hypers = self.kernel.compute_kl()

    return nll + kl_u + kl_hypers 


def create_gp(in_size, out_size, M=10, n_f=3, n_hypers=5, max_val=3.):
  z = (2. * torch.rand(out_size, M, in_size) - 1.) * max_val
  kernel = RBFKernel(in_size)
  likelihood = MulticlassSoftmax(n_f=n_f)
  gp = ContinualSVGP(z, kernel, likelihood, n_hypers=n_hypers)
  return gp


def main():
  x_train, y_train, *_ = get_toy_cla_four()
  x_train = torch.from_numpy(x_train).float()
  y_train_one_hot = torch.from_numpy(y_train).float()
  y_train = y_train_one_hot.argmax(dim=-1)

  gp = create_gp(x_train.size(-1), y_train_one_hot.size(-1))
  optim = torch.optim.Adam(gp.parameters(), lr=1e-2)

  for e in tqdm(range(10000)):
    optim.zero_grad()

    lik_loss = gp.loss(x_train, y_train)

    loss = lik_loss
    loss.backward()
    optim.step()

    if (e + 1) % 500 == 0:
      print(f'Epoch {e}/10000: {loss.detach().item()}')


if __name__ == "__main__":
  main()
