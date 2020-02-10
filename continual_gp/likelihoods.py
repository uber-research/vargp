import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


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
