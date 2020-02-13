import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from .utils import vec2tril


class ContinualSVGP(nn.Module):
  '''
  Arguments:
    z_init: Initial inducing points out_size x M x in_size
    prev_params: List of old inducing points
  '''
  def __init__(self, z_init, kernel, likelihood, n_hypers=1, jitter=1e-4,
               prev_params=None):
    super().__init__()

    self.prev_params = prev_params or []

    self.M = z_init.size(-2)

    self.kernel = kernel
    self.n_hypers = n_hypers
    self.likelihood = likelihood

    # New inducing points
    self.z = nn.Parameter(z_init.detach())

    # Variational parameters for q(u_t | u_{<t})
    out_size = self.z.size(0)
    self.u_mean = nn.Parameter(torch.Tensor(out_size, self.M, 1).normal_(0., .5))
    self.u_tril_vec = nn.Parameter(torch.ones(out_size, (self.M * (self.M + 1)) // 2))

    self.jitter = jitter

  def linear_gauss_joint(self, m_old, L_old, z_old, m_cur, L_cur, z_cur, theta):
    '''
    Utility function, linear Gaussian systems style,
    gets the full joint p(x,y) = N(x, m, S)N(y; Ax+b, C)

    In our setting this is given by
    mu = [m; Am + b]
    cov = [S, SA^t; AS^T, C + ASA^T]

    Arguments:
      m_old, m_cur: [n_hypers x] out_size x M x 1
      L_old, L_cur: [n_hypers x] out_size x M x M
      z_old, z_cur: out_size x M x in_size
      theta: n_hypers x theta_size

    Returns:
      m_joint: n_hypers x out_size x (M + M) x 1
      L_cov_joint: n_hypers x out_size x (M + M) x (M + M)
      z_joint: out_size x (M + M) x in_size
    '''
    n_hypers = theta.size(0)

    if m_old.dim() == 3:
      m_old = m_old.unsqueeze(0).expand(n_hypers, -1, -1, -1)

    if L_old.dim() == 3:
      L_old = L_old.unsqueeze(0).expand(n_hypers, -1, -1, -1)

    if m_cur.dim() == 3:
      m_cur = m_cur.unsqueeze(0)

    if L_cur.dim() == 3:
      L_cur = L_cur.unsqueeze(0)

    kuf = self.kernel.compute(theta, z_old, z_cur)
    kuu = self.kernel.compute(theta, z_old)
    Lkuu = torch.cholesky(kuu + self.jitter * torch.eye(kuu.size(-1), device=kuu.device), upper=False)

    LKinvKuf, _ = torch.triangular_solve(kuf, Lkuu, upper=False)
    LKinvm, _ = torch.triangular_solve(m_old, Lkuu, upper=False)
    LKinvKuf_LKinvm = torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvm)

    m_new = LKinvKuf_LKinvm + m_cur
    m_joint = torch.cat([m_old, m_new], dim=-2)

    ## Block 00
    S_old = torch.einsum('...ji,...ki->...jk', L_old, L_old)

    ## Block 01
    LKinvLS, _ = torch.triangular_solve(L_old, Lkuu, upper=False)
    LKinvLS_LKinvKuf = torch.einsum('...ij,...ik->...jk', LKinvLS, LKinvKuf)
    SoldAt = torch.einsum('...ji,...ik->...jk', L_old, LKinvLS_LKinvKuf)

    ## Block 10
    ASoldt = torch.einsum('...ij->...ji', SoldAt)

    ## Block 11
    S_cur = torch.einsum('...ji,...ki->...jk', L_cur, L_cur)
    ASoldAt = torch.einsum('...ij,...ik->...jk', LKinvLS_LKinvKuf, LKinvLS_LKinvKuf)

    # Combine all blocks
    cov_joint = torch.cat([
      torch.cat([S_old, SoldAt], dim=-1),
      torch.cat([ASoldt, S_cur + ASoldAt], dim=-1)
    ], dim=-2)

    L_cov_joint = torch.cholesky(cov_joint + self.jitter * torch.eye(cov_joint.size(-1), device=cov_joint.device), upper=False)

    z_joint = torch.cat([z_old, z_cur], dim=-2)

    # only for the conditional
    prior_cov = self.kernel.compute(theta, z_cur, z_cur) - torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvKuf)
    L_prior_cov = torch.cholesky(prior_cov + self.jitter * torch.eye(prior_cov.size(-1), device=prior_cov.device), upper=False)

    cache = dict(LKinvKuf_LKinvm=LKinvKuf_LKinvm, L_prior_cov=L_prior_cov)

    return m_joint, L_cov_joint, z_joint, cache

  def linear_gauss_conditional_diag(self, x, z, u_mean, u_tril, theta):
    '''
    Utility function, linear Gaussian systems style,
    gets the marginal p(f|x,θ) of the joint
    p(f,u|x,z,θ) = p(f|x,u,θ)p(u|z,θ)

    Marginals are given by:

    mu = K(x,z|θ)Kinv(z,z|θ)u
    cov = K(x,x|θ) - K(x,z|θ)Kinv(z,z|θ)K(z,x|θ) + K(x,z|θ)Kinv(z,z|θ)(Su)Kinv(z,z|θ)K(z,x|θ)

    The code below only keeps diagonal covariance.

    Arguments:
      x: data, B x in_size
      z: inducing points, out_size x M x in_size
      u_mean: variational dist. mean, out_size x M x 1
      u_tril: variational dist. scale_tril, out_size x M x M
      theta: kernel hyperparameters n_hypers x theta_size

    Returns:
      mu: out means, n_hypers x out_size x B
      var: out variances, n_hypers x out_size x B
    '''
    kuf = self.kernel.compute(theta, z, x.unsqueeze(0).expand(z.size(0), -1, -1))
    kuu = self.kernel.compute(theta, z)
    Lkuu = torch.cholesky(kuu + self.jitter * torch.eye(kuu.size(-1), device=kuu.device), upper=False)
    LKinvu, _ = torch.triangular_solve(u_mean, Lkuu, upper=False)
    LKinvKuf, _ = torch.triangular_solve(kuf, Lkuu, upper=False)

    kff_diag = self.kernel.compute_diag(theta)
    diag1 = (LKinvKuf**2).sum(dim=-2)
    LKinvLs, _ = torch.triangular_solve(u_tril, Lkuu, upper=False)
    vec2 = torch.einsum('...ij,...ik->...jk', LKinvLs, LKinvKuf)
    diag2 = (vec2**2).sum(dim=-2)

    mu = torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvu).squeeze(-1)
    var = kff_diag - diag1 + diag2

    cache = dict(Lkuu=Lkuu)

    return mu, var, cache

  def forward(self, x):
    '''
    Arguments:
      x: B x in_size

    Returns:
      Output distributions for n_hypers samples of hyperparameters.
      The output contains only diagonal of the full covariance.

      pred_mu: n_hypers x out_size x B
      pred_var: n_hypers x out_size x B
    '''
    theta = self.kernel.sample_hypers(self.n_hypers)

    if self.prev_params:
      # Statistics for q(u_{< t} | θ)
      mu_lt = self.prev_params[0].get('u_mean')
      L_cov_lt = self.prev_params[0].get('u_tril')
      z_lt = self.prev_params[0].get('z')
      for t, params in enumerate(self.prev_params[1:]):
        mu_lt, L_cov_lt, z_lt, _ = self.linear_gauss_joint(mu_lt, L_cov_lt, z_lt,
                                                           params.get('u_mean'),
                                                           params.get('u_tril'), params.get('z'),
                                                           theta)

      # Statistics for q(u_{<= t} | θ)
      u_tril = vec2tril(self.u_tril_vec, self.M)
      mu_leq_t, L_cov_leq_t, z_leq_t, cache_leq_t = self.linear_gauss_joint(mu_lt, L_cov_lt, z_lt,
                                                                            self.u_mean, u_tril, self.z,
                                                                            theta)

      # Statistics for q(u_t | u_{< t}, θ)
      mu_t = (cache_leq_t.get('LKinvKuf_LKinvm') + self.u_mean.unsqueeze(0)).squeeze(-1)
      L_cov_t = u_tril.unsqueeze(0)

      # Statistics for p(u_t| u_{< t}, θ)
      prior_mu_t = cache_leq_t.get('LKinvKuf_LKinvm').squeeze(-1)
      prior_L_cov_t = cache_leq_t.get('L_prior_cov')

      pred_mu, pred_var, _ = self.linear_gauss_conditional_diag(x, z_leq_t, mu_leq_t, L_cov_leq_t, theta)
    else:
      # Statistics for q(u_{<= 1} | θ) = q(u_1)
      mu_leq_t, L_cov_leq_t, z_leq_t = self.u_mean, vec2tril(self.u_tril_vec, self.M), self.z

      # Statistics for q(u_1 | u_{< 1}, θ) = q(u_1)
      mu_t = mu_leq_t.squeeze(-1).unsqueeze(0)
      L_cov_t = L_cov_leq_t.unsqueeze(0)

      pred_mu, pred_var, pred_cache = self.linear_gauss_conditional_diag(x, z_leq_t, mu_leq_t, L_cov_leq_t, theta)

      # Statistics for p(u_1 | u_{< 1}, θ) = p(u_1)
      prior_mu_t = torch.zeros_like(mu_t)
      prior_L_cov_t = pred_cache.get('Lkuu')

    loss_cache = dict(var_mu_t=mu_t, var_L_cov_t=L_cov_t, prior_mu_t=prior_mu_t, prior_L_cov_t=prior_L_cov_t)

    return pred_mu, pred_var, loss_cache

  def loss(self, x, y):
    pred_mu, pred_var, loss_cache = self(x)
    nll = self.likelihood.loss(pred_mu, pred_var, y)

    var_dist = dist.MultivariateNormal(
      loss_cache.get('var_mu_t'),
      scale_tril=loss_cache.get('var_L_cov_t'))

    prior_dist = dist.MultivariateNormal(
      loss_cache.get('prior_mu_t'),
      scale_tril=loss_cache.get('prior_L_cov_t'))

    kl_u = kl_divergence(var_dist, prior_dist).sum(dim=-1).mean(dim=0)

    kl_hypers = self.kernel.kl_hypers()

    return kl_hypers, kl_u, nll

  def predict(self, x):
    pred_mu, pred_var, _ = self(x)
    return self.likelihood.predict(pred_mu, pred_var)
