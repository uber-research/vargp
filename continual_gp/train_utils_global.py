import random
import numpy as np
import torch
import torch_optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from .kernels import RBFKernel
from .likelihoods import MulticlassSoftmax
from .models_global import GlobalContinualSVGP
from .utils import vec2tril
from .train_utils import compute_accuracy, process_params, EarlyStopper


def create_class_gp(dataset, M=20, n_f=10, n_var_samples=3,
                    map_est_hypers=False, prev_params=None):
  prev_params = process_params(prev_params)

  N = len(dataset)
  out_size = torch.unique(dataset.targets).size(0)

  if prev_params:
    z_prev = prev_params[-1].get('z')
    z_new = []
    for i in range(out_size):
      z_prev_i = z_prev[i]
      M_prev_i = z_prev_i.shape[0]
      # this silently assumes more inducing points for additional tasks
      M_add_i = M - M_prev_i
      z_add_i = dataset[torch.randperm(N)[:M_add_i]][0]
      z_i = torch.cat([z_prev_i, z_add_i], 0)
      z_new.append(z_i)
    z = torch.stack(z_new)
  else:
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
  gp = GlobalContinualSVGP(z, kernel, likelihood, n_var_samples=n_var_samples,
                     prev_params=prev_params)
  return gp


def train(task_id, train_set, val_set, test_set, map_est_hypers=False,
          epochs=1, M=20, n_f=10, n_var_samples=3, batch_size=512, lr=1e-2, beta=1.0,
          eval_interval=10, patience=20, prev_params=None, logger=None, device=None):
  gp = create_class_gp(train_set, M=M, n_f=n_f, n_var_samples=n_var_samples,
                       map_est_hypers=map_est_hypers,
                       prev_params=prev_params).to(device)

  stopper = EarlyStopper(patience=patience)

  # optim = torch.optim.Adam(gp.parameters(), lr=lr)
  optim = torch_optimizer.Yogi(gp.parameters(), lr=lr)

  N = len(train_set)
  loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  for e in tqdm(range(epochs)):
    for x, y in tqdm(loader, leave=False):
      optim.zero_grad()

      kl_hypers, kl_u, u_prev_reg, lik = gp.loss(x.to(device), y.to(device))

      loss = beta * kl_hypers + kl_u - u_prev_reg + (N / x.size(0)) * lik
      loss.backward()

      optim.step()

    if (e + 1) % eval_interval == 0:
      train_acc = compute_accuracy(train_set, gp, device=device)
      val_acc = compute_accuracy(val_set, gp, device=device)
      test_acc = compute_accuracy(test_set, gp, device=device)

      loss_summary = {
        f'task{task_id}/loss/kl_hypers': kl_hypers.detach().item(),
        f'task{task_id}/loss/kl_u': kl_u.detach().item(),
        f'task{task_id}/loss/lik': lik.detach().item()
      }

      acc_summary = {
        f'task{task_id}/train/acc': train_acc,
        f'task{task_id}/val/acc': val_acc,
        f'task{task_id}/test/acc': test_acc,
      }

      if logger is not None:
        for k, v in (dict(**loss_summary, **acc_summary)).items():
          logger.add_scalar(k, v, global_step=e + 1)

      stopper(val_acc, dict(state_dict=gp.state_dict(), acc_summary=acc_summary, step=e + 1))
      if stopper.is_done():
        break

  info = stopper.info()
  if logger is not None:
    for k, v in info.get('acc_summary').items():
      logger.add_scalar(f'{k}_best', v, global_step=info.get('step'))

    with open(f'{logger.log_dir}/ckpt{task_id}.pt', 'wb') as f:
      torch.save(info.get('state_dict'), f)
    wandb.save(f'{logger.log_dir}/ckpt{task_id}.pt')

  return info.get('state_dict')
