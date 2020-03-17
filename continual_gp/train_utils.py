import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .kernels import RBFKernel
from .likelihoods import MulticlassSoftmax
from .models import ContinualSVGP
from .utils import vec2tril


def set_seeds(seed=None):
  if seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def process_params(params):
  if params is None:
    return None

  def process(p):
    if 'u_tril_vec' in p:
      p['u_tril'] = vec2tril(p.pop('u_tril_vec'))
    return p

  return [process(p) for p in params]


def compute_accuracy(dataset, gp, batch_size=512, device=None):
  loader = DataLoader(dataset, batch_size=batch_size)

  with torch.no_grad():
    count = 0
    for x, y in tqdm(loader, leave=False):
      preds = gp.predict(x.to(device))
      count += (preds.argmax(dim=-1) == y.to(device)).sum().item()

    acc = count / len(dataset)

  return acc


def create_class_gp(dataset, M=20, n_f=10, n_var_samples=3,
                    ep_var_mean=True, map_est_hypers=False,
                    prev_params=None):
  prev_params = process_params(prev_params)

  N = len(dataset)
  out_size = torch.unique(dataset.targets).size(0)

  # init inducing points at random data points.
  z = torch.stack([
    dataset[torch.randperm(N)[:M]][0]
    for _ in range(out_size)])

  prior_log_mean, prior_log_logvar = None, None
  if prev_params:
    prior_log_mean = prev_params[-1].get('kernel.log_mean')
    prior_log_logvar = prev_params[-1].get('kernel.log_logvar')

  kernel = RBFKernel(z.size(-1), prior_log_mean=prior_log_mean, prior_log_logvar=prior_log_logvar,
                     map_est=map_est_hypers)
  likelihood = MulticlassSoftmax(n_f=n_f)
  gp = ContinualSVGP(z, kernel, likelihood, n_var_samples=n_var_samples,
                     ep_var_mean=ep_var_mean, prev_params=prev_params)
  return gp


# Reference: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopper:
  def __init__(self, patience=10, delta=1e-4):
    self.patience = patience
    self.delta = delta

    self._counter = 0

    self._best_info = None
    self._best_score = None

  def is_done(self):
    return self._counter >= self.patience

  def info(self):
    return self._best_info

  def __call__(self, score, info):
    assert not self.is_done()

    if self._best_score is None:
      self._best_score = score
      self._best_info = info
    elif score < self._best_score + self.delta:
      self._counter += 1
    else:
      self._best_score = score
      self._best_info = info
      self._counter = 0
