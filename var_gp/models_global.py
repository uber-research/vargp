import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from .utils import vec2tril


class GlobalContinualSVGP(nn.Module):
  def __init__(self, z_init, kernel, likelihood, n_var_samples=1, jitter=1e-4,
               prev_params=None):
    super().__init__()

    self.prev_params = prev_params or []

    self.M = z_init.size(-2)

    self.kernel = kernel
    self.n_var_samples = n_var_samples
    self.likelihood = likelihood

    # New inducing points
    self.z = nn.Parameter(z_init.detach())

    # Variational parameters for q(u_t)
    out_size = self.z.size(0)
    self.u_mean = nn.Parameter(torch.Tensor(out_size, self.M, 1).normal_(0., .5))
    self.u_tril_vec = nn.Parameter(torch.ones(out_size, (self.M * (self.M + 1)) // 2))

    self.jitter = jitter

  def gp_cond_diag(self, theta, x, z, u_mean, u_tril):
    '''
    Utility function, linear Gaussian systems style,
    gets the marginal p(f|x,θ) of the joint
    p(f,u|x,z,θ) = p(f|x,u,θ)p(u|z,θ)

    Marginals are given by:

    mu = K(x,z|θ)Kinv(z,z|θ)u
    cov = K(x,x|θ) - K(x,z|θ)Kinv(z,z|θ)K(z,x|θ) + K(x,z|θ)Kinv(z,z|θ)(Su)Kinv(z,z|θ)K(z,x|θ)

    The code below only keeps diagonal covariance.

    Arguments:
      theta: kernel hyperparameters, n_hypers x theta_size
      x: data, B x in_size
      z: inducing points, out_size x M x in_size
      u_mean: variational dist. mean, out_size x M x 1
      u_tril: variational dist. scale_tril, out_size x M x M

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
  
  def gp_cond_full(self, theta, x, z, u_mean, u_tril):
    '''
    Utility function, linear Gaussian systems style,
    gets the marginal p(f|x,θ) of the joint
    p(f,u|x,z,θ) = p(f|x,u,θ)p(u|z,θ)

    Marginals are given by:

    mu = K(x,z|θ)Kinv(z,z|θ)u
    cov = K(x,x|θ) - K(x,z|θ)Kinv(z,z|θ)K(z,x|θ) + K(x,z|θ)Kinv(z,z|θ)(Su)Kinv(z,z|θ)K(z,x|θ)

    The code below returns the full covariance.

    Arguments:
      theta: kernel hyperparameters, n_hypers x theta_size
      x: data, out_size x B x in_size
      z: inducing points, out_size x M x in_size
      u_mean: variational dist. mean, out_size x M x 1
      u_tril: variational dist. scale_tril, out_size x M x M

    Returns:
      mu: out means, n_hypers x out_size x B
      var: out variances, n_hypers x out_size x B
    '''
    kuf = self.kernel.compute(theta, z, x)
    kuu = self.kernel.compute(theta, z)
    Lkuu = torch.cholesky(kuu + self.jitter * torch.eye(kuu.size(-1), device=kuu.device), upper=False)
    LKinvu, _ = torch.triangular_solve(u_mean, Lkuu, upper=False)
    LKinvKuf, _ = torch.triangular_solve(kuf, Lkuu, upper=False)

    kff = self.kernel.compute(theta, x)
    cov1 = torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvKuf)
    LKinvLs, _ = torch.triangular_solve(u_tril, Lkuu, upper=False)
    vec2 = torch.einsum('...ij,...ik->...jk', LKinvLs, LKinvKuf)
    cov2 = torch.einsum('...ij,...ik->...jk', vec2, vec2)

    mu = torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvu).squeeze(-1)
    var = kff - cov1 + cov2

    Lkff = torch.cholesky(kff + self.jitter * torch.eye(kff.size(-1), device=kff.device), upper=False)
    cache = dict(Lkff=Lkff)

    return mu, var, cache

  def forward(self, x, loss_cache=False):
    '''
    Arguments:
      x: B x in_size

    Returns:
      Output distributions for n_hypers samples of hyperparameters.
      The output contains only diagonal of the full covariance.

      pred_mu: n_hypers x out_size x B
      pred_var: n_hypers x out_size x B
    '''
    theta = self.kernel.sample_hypers(self.n_var_samples)

    loss_cache = dict() if loss_cache else None

    if self.prev_params:
      # Statistics for q(u_{t-1} | θ)
      mu_tm1 = self.prev_params[0].get('u_mean')
      L_cov_tm1 = self.prev_params[0].get('u_tril')
      z_tm1 = self.prev_params[0].get('z')

      # Statistics for q(u_t | θ)
      mu_t, L_cov_t, z_t = self.u_mean, vec2tril(self.u_tril_vec, self.M), self.z
      pred_mu, pred_var, pred_cache = self.gp_cond_diag(theta, x, z_t, mu_t, L_cov_t)
      pred_mu_tm1, pred_cov_tm1, pred_cache_tm1 = self.gp_cond_full(theta, z_tm1, z_t, mu_t, L_cov_t)

      if loss_cache is not None:
        # Statistics for q(u_t)
        mu_t = mu_t.squeeze(-1).unsqueeze(0).unsqueeze(0)
        L_cov_t = L_cov_t.unsqueeze(0).unsqueeze(0)

        # Statistics for p(u_t | theta)
        prior_mu_t = torch.zeros_like(mu_t)
        prior_L_cov_t = pred_cache.get('Lkuu').unsqueeze(0)

        # statistics for q_t(u_{t-1} | theta)
        pred_mu_tm1 = pred_mu_tm1.squeeze(-1)
        pred_L_cov_tm1 = torch.cholesky(pred_cov_tm1 + self.jitter * torch.eye(pred_cov_tm1.size(-1), device=pred_cov_tm1.device), upper=False)

        # Statistics for q(u_{t-1})
        mu_tm1 = mu_tm1.squeeze(-1).unsqueeze(0).unsqueeze(0)
        L_cov_tm1 = L_cov_tm1.unsqueeze(0).unsqueeze(0)

        # Statistics for p(u_{t-1} | theta)
        prior_mu_tm1 = torch.zeros_like(mu_tm1)
        prior_L_cov_tm1 = pred_cache_tm1.get('Lkff').unsqueeze(0)

        loss_cache = dict(var_mu_t=mu_t, var_L_cov_t=L_cov_t,
                          prior_mu_t=prior_mu_t, prior_L_cov_t=prior_L_cov_t,
                          var_mu_tm1=mu_tm1, var_L_cov_tm1=L_cov_tm1,
                          prior_mu_tm1=prior_mu_tm1, prior_L_cov_tm1=prior_L_cov_tm1,
                          pred_mu_tm1=pred_mu_tm1, pred_L_cov_tm1=pred_L_cov_tm1)
    else:
      # Statistics for q(u_1)
      mu_t, L_cov_t, z_t = self.u_mean, vec2tril(self.u_tril_vec, self.M), self.z
      pred_mu, pred_var, pred_cache = self.gp_cond_diag(theta, x, z_t, mu_t, L_cov_t)

      if loss_cache is not None:
        # Statistics for q(u_1)
        mu_t = mu_t.squeeze(-1).unsqueeze(0).unsqueeze(0)
        L_cov_t = L_cov_t.unsqueeze(0).unsqueeze(0)

        # Statistics for p(u_1 | theta)
        prior_mu_t = torch.zeros_like(mu_t)
        prior_L_cov_t = pred_cache.get('Lkuu').unsqueeze(0)

        loss_cache = dict(var_mu_t=mu_t, var_L_cov_t=L_cov_t, prior_mu_t=prior_mu_t, prior_L_cov_t=prior_L_cov_t)

    return pred_mu, pred_var, loss_cache

  def loss(self, x, y):
    pred_mu, pred_var, loss_cache = self(x, loss_cache=True)
    nll = self.likelihood.loss(pred_mu, pred_var, y)

    var_dist = dist.MultivariateNormal(
      loss_cache.get('var_mu_t'),
      scale_tril=loss_cache.get('var_L_cov_t'))

    prior_dist = dist.MultivariateNormal(
      loss_cache.get('prior_mu_t'),
      scale_tril=loss_cache.get('prior_L_cov_t'))

    kl_u = kl_divergence(var_dist, prior_dist).sum(dim=-1).mean(dim=0).mean(dim=0)

    kl_hypers = self.kernel.kl_hypers()

    u_prev_reg = 0

    if "var_mu_tm1" in loss_cache.keys():
      # q(u_{t-1})
      var_dist_tm1 = dist.MultivariateNormal(loss_cache.get('var_mu_tm1'), scale_tril=loss_cache.get('var_L_cov_tm1'))
      
      # p(u_{t-1} | theta)
      prior_dist_tm1 = dist.MultivariateNormal(loss_cache.get('prior_mu_tm1'), scale_tril=loss_cache.get('prior_L_cov_tm1'))
      
      # q_t{u_{t-1} | theta}
      pred_mu_tm1 = loss_cache.get('pred_mu_tm1')
      pred_L_cov_tm1 = loss_cache.get('pred_L_cov_tm1')
      q_t_u_tm1 = dist.MultivariateNormal(pred_mu_tm1, scale_tril=pred_L_cov_tm1)
      u_tm1_samples = q_t_u_tm1.rsample(torch.Size([self.n_var_samples]))
      
      # compute log densities
      log_var_dist = var_dist_tm1.log_prob(u_tm1_samples)
      log_prior_dist = prior_dist_tm1.log_prob(u_tm1_samples)
      u_prev_reg = (log_var_dist - log_prior_dist).sum(dim=-1).mean(dim=0).mean(dim=0)

    return kl_hypers, kl_u, u_prev_reg, nll

  def predict(self, x):
    pred_mu, pred_var, _ = self(x)
    return self.likelihood.predict(pred_mu, pred_var)
