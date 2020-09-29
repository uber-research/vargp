import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.kl import kl_divergence

from .gp_utils import vec2tril, mat2trilvec, cholesky, rev_cholesky, gp_cond, linear_joint, linear_marginal_diag
from .kernels import RBFKernel, DeepRBFKernel
from .likelihoods import MulticlassSoftmax


class VARGP(nn.Module):
  def __init__(self, z_init, kernel, likelihood, n_var_samples=1, ep_var_mean=True, prev_params=None):
    super().__init__()

    self.var_mean_mask = float(ep_var_mean)

    self.prev_params = [
      dict(z=p['z'], u_mean=p['u_mean'], u_tril=vec2tril(p['u_tril_vec']))
      for p in (prev_params or [])
    ]

    self.M = z_init.size(-2)

    self.kernel = kernel
    self.n_v = n_var_samples
    self.likelihood = likelihood

    self.z = nn.Parameter(z_init.detach())

    out_size = self.z.size(0)
    self.u_mean = nn.Parameter(torch.Tensor(out_size, self.M, 1).normal_(0., .5))
    self.u_tril_vec = nn.Parameter(
      mat2trilvec(torch.eye(self.M).unsqueeze(0).expand(out_size, -1, -1)))

  def compute_q(self, theta, cache=None):
    '''
    Compute variational auto-regressive distributions.

    Arguments:
      theta: n_hypers x (D + 1)

    Returns
      mu_lt: n_hypers x out_size x (\sum M_t - M_T) x 1
      S_lt: n_hypers x out_size x (\sum M_t - M_T) x (\sum M_t - M_T)
      mu_leq_t: n_hypers x out_size x (\sum M_t) x 1
      S_leq_t: n_hypers x out_size x (\sum M_t) x (\sum M_t)
      z_leq_t: out_size x (\sum M_t) x D
    '''
    n_hypers = theta.size(0)

    ## Compute q(u_{<t} | \theta)
    z_lt = self.prev_params[0]['z']
    mu_lt = self.prev_params[0]['u_mean']
    S_lt = rev_cholesky(self.prev_params[0]['u_tril'])

    if mu_lt.dim() == 3:
      mu_lt = mu_lt.unsqueeze(0).expand(n_hypers, -1, -1, -1)
    if S_lt.dim() == 3:
      S_lt = S_lt.unsqueeze(0).expand(n_hypers, -1, -1, -1)

    for params in self.prev_params[1:]:
      Kzx = self.kernel.compute(theta, z_lt, params['z'])
      Kzz = self.kernel.compute(theta, z_lt)

      V = rev_cholesky(params['u_tril']).unsqueeze(0).expand(n_hypers, -1, -1, -1)
      b = params['u_mean'].unsqueeze(0).expand(n_hypers, -1, -1, -1)

      mu_lt, S_lt = linear_joint(mu_lt, S_lt, Kzx, Kzz, V, b)
      z_lt = torch.cat([z_lt, params['z']], dim=-2)

    ## Compute q(u_{\leq t} | \theta)
    Kzx = self.kernel.compute(theta, z_lt, self.z)
    Kzz = self.kernel.compute(theta, z_lt)

    V = rev_cholesky(vec2tril(self.u_tril_vec)).unsqueeze(0).expand(n_hypers, -1, -1, -1)
    b = self.u_mean.unsqueeze(0).expand(n_hypers, -1, -1, -1)

    cache_leq_t = dict()
    mu_leq_t, S_leq_t = linear_joint(mu_lt, S_lt, Kzx, Kzz, V, b, cache=cache_leq_t)
    z_leq_t = torch.cat([z_lt, self.z], dim=-2)

    if isinstance(cache, dict):
      cache['Lz_lt'] = cache_leq_t['Lz']
      cache['Lz_lt_Kz_lt_z_t'] = cache_leq_t['Lz_Kzx']

    return mu_lt, S_lt, \
           mu_leq_t, S_leq_t, \
           z_leq_t

  def compute_pf_diag(self, theta, x, mu_leq_t, S_leq_t, z_leq_t, cache=None):
    '''
    Compute p(f) = \int p(f|u_{\leq t})q(u_{\leq t}).
    Only diagonal of covariance for p(f) is used.

    Arguments:
      theta: n_hypers x (D + 1)
      x: B x D
      mu_leq_t: [n_hypers] x out_size x (\sum M_t) x 1
      S_leq_t: [n_hypers] x out_size x (\sum M_t) x (\sum M_t)
      z_leq_t: out_size x (\sum M_t) x D

    Returns:
      f_mean: n_hypers x out_size x B
      f_var: n_hypers x out_size x B
    '''
    xf = x.unsqueeze(0).expand(z_leq_t.size(0), -1, -1)

    Kzz = self.kernel.compute(theta, z_leq_t)
    Kzx = self.kernel.compute(theta, z_leq_t, xf)
    Kxx_diag = self.kernel.compute_diag(theta)

    f_mean, f_var = linear_marginal_diag(mu_leq_t, S_leq_t, Kzz, Kzx, Kxx_diag, cache=cache)
    return f_mean, f_var

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
    theta = self.kernel.sample_hypers(self.n_v)

    if self.prev_params:
      cache_q = dict()

      mu_lt, S_lt, mu_leq_t, S_leq_t, z_leq_t = self.compute_q(theta, cache=cache_q)

      pred_mu, pred_var = self.compute_pf_diag(theta, x, mu_leq_t, S_leq_t, z_leq_t)

      if isinstance(loss_cache, dict):
        q_lt = dist.MultivariateNormal(mu_lt.squeeze(-1), covariance_matrix=S_lt)
        u_lt = q_lt.rsample(torch.Size([self.n_v])).unsqueeze(-1)
        # u_lt = mu_lt.unsqueeze(0)

        if u_lt.dim() == 4:
          u_lt = u_lt.unsqueeze(1)

        ## Compute p(u_t | u_{<t}, \theta)
        Lz = cache_q.pop('Lz_lt').unsqueeze(0)
        Lz_Kzx = cache_q.pop('Lz_lt_Kz_lt_z_t').unsqueeze(0).expand(self.n_v, *([-1] * (Lz.dim() - 1)))
        Kzz = self.kernel.compute(theta, self.z).unsqueeze(0)
        prior_mu_t, prior_cov_t = gp_cond(u_lt, None, None, Kzz, Lz=Lz, Lz_Kzx=Lz_Kzx)

        ## Compute q(u_t | u_{<t}, \theta)
        var_mu_t = prior_mu_t * self.var_mean_mask + self.u_mean.unsqueeze(0).unsqueeze(0)
        var_L_cov_t = vec2tril(self.u_tril_vec, self.M).unsqueeze(0).unsqueeze(0)

        loss_cache.update(dict(var_mu_t=var_mu_t.squeeze(-1), var_L_cov_t=var_L_cov_t,
                               prior_mu_t=prior_mu_t.squeeze(-1), prior_L_cov_t=cholesky(prior_cov_t)))
    else:
      cache_pf = dict()

      mu_leq_t = self.u_mean
      L_cov_leq_t = vec2tril(self.u_tril_vec, self.M)

      pred_mu, pred_var = self.compute_pf_diag(theta, x, mu_leq_t, rev_cholesky(L_cov_leq_t), self.z, cache=cache_pf)

      if isinstance(loss_cache, dict):
        # Compute q(u_1)
        mu_t = mu_leq_t.squeeze(-1).unsqueeze(0).unsqueeze(0)
        L_cov_t = L_cov_leq_t.unsqueeze(0).unsqueeze(0)

        # Compute p(u_1)
        prior_mu_t = torch.zeros_like(mu_t)
        prior_L_cov_t = cache_pf.pop('Lz').unsqueeze(0)

        loss_cache.update(dict(var_mu_t=mu_t, var_L_cov_t=L_cov_t, prior_mu_t=prior_mu_t, prior_L_cov_t=prior_L_cov_t))

    return pred_mu, pred_var

  def loss(self, x, y):
    loss_cache = dict()
    pred_mu, pred_var = self(x, loss_cache=loss_cache)
    nll = self.likelihood.loss(pred_mu, pred_var, y)

    var_dist = dist.MultivariateNormal(
      loss_cache.pop('var_mu_t'),
      scale_tril=loss_cache.pop('var_L_cov_t'))

    prior_dist = dist.MultivariateNormal(
      loss_cache.pop('prior_mu_t'),
      scale_tril=loss_cache.pop('prior_L_cov_t'))

    kl_u = kl_divergence(var_dist, prior_dist).sum(dim=-1).mean(dim=0).mean(dim=0)

    kl_hypers = self.kernel.kl_hypers()

    return kl_hypers, kl_u, nll

  def predict(self, x):
    pred_mu, pred_var = self(x)
    return self.likelihood.predict(pred_mu, pred_var)

  @staticmethod
  def create_clf(dataset, M=20, n_f=10, n_var_samples=3, prev_params=None,
                 ep_var_mean=True, map_est_hypers=False, dkl=False):
    N = len(dataset)
    out_size = torch.unique(dataset.targets).size(0)

    ## init inducing points at random data points.
    z = torch.stack([
      dataset[torch.randperm(N)[:M]][0]
      for _ in range(out_size)])

    prior_log_mean, prior_log_logvar = None, None
    phi_params = None

    if prev_params:
      ## Init kernel hyperprior to last timestep.
      prior_log_mean = prev_params[-1].get('kernel.log_mean')
      prior_log_logvar = prev_params[-1].get('kernel.log_logvar')

      ## Init kernel NN to last timestep if available.
      if dkl:
        phi_params = {k[11:]: v for k, v in prev_params[-1].items() if k.startswith('kernel.phi.')}

      def process(p):
        for k in list(p.keys()):
          if k.startswith('kernel'):
            p.pop(k)
        return p

      prev_params = [process(p) for p in prev_params]

    if dkl:
      kernel = DeepRBFKernel(z.size(-1), prior_log_mean=prior_log_mean,
                             prior_log_logvar=prior_log_logvar, map_est=map_est_hypers)
      if phi_params is not None:
        kernel.phi.load_state_dict(phi_params)
    else:
      kernel = RBFKernel(z.size(-1), prior_log_mean=prior_log_mean,
                         prior_log_logvar=prior_log_logvar, map_est=map_est_hypers)

    likelihood = MulticlassSoftmax(n_f=n_f)
    gp = VARGP(z, kernel, likelihood, n_var_samples=n_var_samples,
               ep_var_mean=ep_var_mean, prev_params=prev_params)
    return gp
