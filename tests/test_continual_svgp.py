from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.kl import kl_divergence


class RBFKernel(nn.Module):
    def __init__(self, in_size, prior_log_mean=None, prior_log_logvar=None):
        super().__init__()

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
        log_dist = dist.Normal(self.log_mean, self.log_logvar.exp().sqrt())
        log_hypers = log_dist.rsample(torch.Size([n_hypers]))
        return log_hypers

    def kl_hypers(self):
        var_dist = dist.Normal(self.log_mean, self.log_logvar.exp().sqrt())
        prior_dist = dist.Normal(self.prior_log_mean, self.prior_log_logvar.exp().sqrt())
        total_kl = kl_divergence(var_dist, prior_dist).sum(dim=0)
        return total_kl


def vec2tril(vec, m=None):
  '''
  Arguments:
    vec: K x ((m * (m + 1)) // 2)
    m: integer, if None, inferred from last dimension.

  Returns:
    Batch of lower triangular matrices

    tril: K x m x m
  '''
  if m is None:
    D = vec.size(-1)
    m = (torch.tensor(8. * D + 1).sqrt() - 1.) / 2.
    m = m.long().item()

  batch_shape = vec.shape[:-1]

  idx = torch.tril_indices(m, m)

  tril = torch.zeros(*batch_shape, m, m, device=vec.device)
  tril[..., idx[0], idx[1]] = vec

  # ensure positivity constraint of cholesky diagonals
  tril[..., range(m), range(m)] = F.softplus(tril[..., range(m), range(m)])

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

    eps = torch.randn(n_hypers, self.n_f, out_size, B, device=mu.device)
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
    target_y = y.view(-1, 1, 1).expand(-1, *pred_y.shape[-2:])
    lik_loss = F.nll_loss(pred_y, target_y,
                          reduction='none').mean(dim=-1).mean(dim=-1).sum(dim=0)
    return lik_loss

  def predict(self, mu, var):
    '''
    Arguments:
      mu: n_hypers x out_size x B
      var: n_hypers x out_size x B

    Returns
      output probabilities B x out_size
    '''
    y_f = self(mu, var)

    y = y_f.view(-1, *mu.shape[-2:])

    probs = y.logsumexp(dim=0).exp() / y.size(0)
    return probs.T


class GaussianLikelihood(nn.Module):
  '''
  Independent multi-output Gaussian likelihood.
  '''
  def __init__(self, out_size, init_log_var=-4.):
    super().__init__()

    self.obs_log_var = nn.Parameter(init_log_var * torch.ones(out_size))

  def forward(self, mu, var):
    '''
    Arguments:
      mu: n_hypers x out_size x B
      var: n_hypers x out_size x B

    Returns:
      observation mean and variance

      obs_mu: n_hypers x out_size x B x 1
      obs_var: n_hypers x out_size x B x 1
    '''
    obs_mu = mu.unsqueeze(-1)
    obs_var = var.unsqueeze(-1) + self.obs_log_var.exp().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    return obs_mu, obs_var

  def loss(self, pred_mu, pred_var, y):
    '''
    Arguments:
      pred_mu: n_hypers x out_size x B
      pred_var: n_hypers x out_size x B
      y: out_size x B

    Returns:
      Total loss scalar value.
    '''
    obs_mean, obs_var = self(pred_mu, pred_var)

    y_dist = dist.Independent(dist.Normal(obs_mean, obs_var.sqrt()), 1)
    log_prob = y_dist.log_prob(y.unsqueeze(0).unsqueeze(-1))

    nll = - log_prob.mean(dim=0).mean(dim=0).sum(dim=0)
    return nll

  def predict(self, mu, var):
    return mu


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
      z_old: out_size x M x in_size
      z_cur: out_size x M x in_size
      theta: n_hypers x theta_size

    Returns:
      m_joint: n_hypers x out_size x (M + M) x 1
    '''
    n_hypers = theta.size(0)

    if m_old.dim() == 3:
      m_old = m_old.unsqueeze(0)

    if L_old.dim() == 3:
      L_old = L_old.unsqueeze(0)

    if m_cur.dim() == 3:
      m_cur = m_cur.unsqueeze(0)

    if L_cur.dim() == 3:
      L_cur = L_cur.unsqueeze(0)

    kuf = self.kernel.compute(theta, z_old, z_cur)
    kuu = self.kernel.compute(theta, z_old)
    Lkuu = torch.cholesky(kuu + self.jitter * torch.eye(kuu.size(-1), device=kuu.device), upper=False)

    LKinvKuf, _ = torch.triangular_solve(kuf, Lkuu, upper=False)
    LKinvm, _ = torch.triangular_solve(m_old, Lkuu, upper=False)

    m_new = torch.einsum('...ij,...ik->...jk', LKinvKuf, LKinvm) + m_cur
    m_joint = torch.cat([m_old.expand_as(m_new), m_new], dim=-2)

    ## Block 00
    S_old = torch.einsum('...ji,...ki->...jk', L_old, L_old)

    ## Block 01
    LKinvLS, _ = torch.triangular_solve(L_old, Lkuu, upper=False)
    LKinvLS_LKinvKuf = torch.einsum('...ij,...ik->...jk', LKinvLS, LKinvKuf)
    SoldAt = torch.einsum('...ji,...ik->...jk', L_old.expand_as(LKinvLS_LKinvKuf), LKinvLS_LKinvKuf)

    ## Block 10
    ASoldt = torch.einsum('...ij->...ji', SoldAt)

    ## Block 11
    S_cur = torch.einsum('...ji,...ki->...jk', L_cur, L_cur)
    ASoldAt = torch.einsum('...ij,...ik->...jk', LKinvLS_LKinvKuf, LKinvLS_LKinvKuf)

    # Combine all blocks
    cov_joint = torch.cat([
      torch.cat([S_old.expand_as(SoldAt), SoldAt], dim=-1),
      torch.cat([ASoldt, S_cur + ASoldAt], dim=-1)
    ], dim=-2)

    L_cov_joint = torch.cholesky(cov_joint + self.jitter * torch.eye(cov_joint.size(-1), device=cov_joint.device), upper=False)

    z_joint = torch.cat([z_old, z_cur], dim=-2)

    return m_joint, L_cov_joint, z_joint

  def linear_gauss_conditional(self, x, z, u_mean, u_tril, theta):
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
    LKinvLs, _ = torch.triangular_solve(u_tril, Lkuu)
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
        mu_lt, L_cov_lt, z_lt = self.linear_gauss_joint(mu_lt, L_cov_lt, z_lt,
                                                        params.get('u_mean'), params.get('u_tril'), params.get('z'),
                                                        theta)

      # Statistics for q(u_{<= t} | θ)
      mu_leq_t, L_cov_leq_t, z_leq_t = self.linear_gauss_joint(mu_lt, L_cov_lt, z_lt,
                                                               self.u_mean, vec2tril(self.u_tril_vec, self.M), self.z,
                                                               theta)
    else:
      mu_leq_t, L_cov_leq_t, z_leq_t = self.u_mean, vec2tril(self.u_tril_vec, self.M), self.z

    pred_mu, pred_var, cache = self.linear_gauss_conditional(x, z_leq_t, mu_leq_t, L_cov_leq_t, theta)
    cache = dict(u_tril=L_cov_leq_t, **cache)

    return pred_mu, pred_var, cache

  def loss(self, x, y):
    ## Expected Negative Log Likelihood
    pred_mu, pred_var, cache = self(x)
    nll = self.likelihood.loss(pred_mu, pred_var, y)

    ## TODO(sanyam): Fix these terms to account for prev_params
    var_mean = self.u_mean.squeeze(-1).unsqueeze(0)
    var_tril = cache.get('u_tril').unsqueeze(0)
    var_dist = dist.MultivariateNormal(var_mean, scale_tril=var_tril)

    prior_mean = torch.zeros_like(self.u_mean).squeeze(-1).unsqueeze(0)
    prior_tril = cache.get('Lkuu')
    prior_dist = dist.MultivariateNormal(prior_mean, scale_tril=prior_tril)

    kl_u = kl_divergence(var_dist, prior_dist).sum(dim=-1).mean(dim=0)

    kl_hypers = self.kernel.kl_hypers()

    return nll + kl_u + kl_hypers

  def predict(self, x):
    pred_mu, pred_var, _ = self(x)
    return self.likelihood.predict(pred_mu, pred_var)


def process_params(params):
  if params is None:
    return None

  def process(p):
    if 'u_tril_vec' in p:
      p['u_tril'] = vec2tril(p.pop('u_tril_vec'))
    return p

  return [process(p) for p in params]


def create_class_gp(x_train, out_size, M=20, n_f=10, n_hypers=3, prev_params=None):
  prev_params = process_params(prev_params)

  z = torch.stack([
    x_train[torch.randperm(x_train.size(0))[:M]]
    for _ in range(out_size)])

  prior_log_mean, prior_log_logvar = None, None
  if prev_params is not None:
    prior_log_mean = prev_params[-1].get('kernel.log_mean')
    prior_log_logvar = prev_params[-1].get('kernel.log_logvar')

  kernel = RBFKernel(x_train.size(-1), prior_log_mean=prior_log_mean, prior_log_logvar=prior_log_logvar)
  likelihood = MulticlassSoftmax(n_f=n_f)
  gp = ContinualSVGP(z, kernel, likelihood, n_hypers=n_hypers,
                     prev_params=prev_params)
  return gp


def train_gp(x_train, y_train, n_classes, epochs=int(1e4), prev_params=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  gp = create_class_gp(x_train, n_classes,
                       M=20, n_f=10, n_hypers=3,
                       prev_params=prev_params).to(device)
  optim = torch.optim.Adam(gp.parameters(), lr=1e-2)

  for e in tqdm(range(epochs)):
    optim.zero_grad()

    lik_loss = gp.loss(x_train, y_train)

    loss = lik_loss
    loss.backward()
    optim.step()

    if (e + 1) % 500 == 0:
      print(f'Loss: {loss.detach().item()}')

  return gp.state_dict()


def test_gp(gp_state_dict, x_train, y_train, x_test, X1, X2, n_classes, prev_params=None):
  x_test = torch.from_numpy(x_test).float().to(device)
  
  with torch.no_grad():
    test_gp = create_class_gp(x_train, n_classes,
                              M=20, n_f=100, n_hypers=10,
                              prev_params=prev_params).to(device)
    test_gp.load_state_dict(gp_state_dict)

    y_pred = test_gp.predict(x_test)

    from test_cla_batch_ml import plot_prediction_four
    plot_prediction_four(
        y_pred.cpu(),
        x_train.cpu(), y_train.cpu(),
        X1, X2,
        test_gp.z.cpu().detach(), "four_batch_cgp"
    )


def main():
  from data_utils import get_toy_cla_four
  x_train, y_train, *test_data = get_toy_cla_four()

  x_train = torch.from_numpy(x_train).float().to(device)
  y_train = torch.from_numpy(y_train).float().argmax(dim=-1).to(device)
  n_classes = y_train.unique().size(0)

  prev_params = []

  # Train GP on only classes 0,1
  c01_idx = torch.masked_select(torch.arange(y_train.size(0)).to(device), (y_train == 0) | (y_train == 1))
  c01_gp_state_dict = train_gp(x_train[c01_idx], y_train[c01_idx], n_classes)
  test_gp(c01_gp_state_dict, x_train[c01_idx], y_train[c01_idx], *test_data, n_classes)

  prev_params.append(c01_gp_state_dict)

  # Train GP on only classes 2,3
  c23_idx = torch.masked_select(torch.arange(y_train.size(0)).to(device), (y_train == 2) | (y_train == 3))
  c23_gp_state_dict = train_gp(x_train[c23_idx], y_train[c23_idx], n_classes, prev_params=prev_params)
  test_gp(c23_gp_state_dict, x_train[c23_idx], y_train[c23_idx], *test_data, n_classes, prev_params=prev_params)


if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  main()
