import os
from tqdm.auto import tqdm
import wandb
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from continual_gp.datasets import SplitMNIST
from continual_gp.train_utils import create_class_gp, compute_accuracy, set_seeds, EarlyStopper


def train(task_id, train_set, val_set, test_set,
          epochs=1, M=20, n_f=10, n_hypers=3, batch_size=512, lr=1e-2,
          beta=1.0, eval_interval=10,
          prev_params=None, logger=None, device=None):
  gp = create_class_gp(train_set, M=M, n_f=n_f, n_hypers=n_hypers,
                       prev_params=prev_params).to(device)

  stopper = EarlyStopper(patience=20)

  optim = torch.optim.Adam(gp.parameters(), lr=lr)

  N = len(train_set)
  loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  for e in tqdm(range(epochs)):
    for x, y in tqdm(loader, leave=False):
      optim.zero_grad()

      kl_hypers, kl_u, lik = gp.loss(x.to(device), y.to(device))

      loss = beta * kl_hypers + kl_u + (N / x.size(0)) * lik
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
  for k, v in info.get('acc_summary').items():
    logger.add_scalar(f'{k}_best', v, global_step=info.get('step'))

  viz_ind_pts = info.get('state_dict').get('z')[2*task_id:2*task_id+2][:, torch.randperm(M)[:8], :].view(16, 1, 28, 28)
  logger.add_images(f'task{task_id}/inducing', viz_ind_pts, global_step=e + 1)

  with open(f'{logger.log_dir}/ckpt{task_id}.pt', 'wb') as f:
    torch.save(info.get('state_dict'), f)
  wandb.save(f'{logger.log_dir}/ckpt{task_id}.pt')

  return info.get('state_dict')


def main(data_dir='/tmp', epochs=10, M=20, lr=1e-2, batch_size=512, beta=1.0, seed=42):
  set_seeds(seed)

  wandb.init(tensorboard=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger = SummaryWriter(log_dir=wandb.run.dir)

  mnist_train = SplitMNIST(f'{data_dir}', train=True)
  mnist_val = SplitMNIST(f'{data_dir}', train=True)
  mnist_test = SplitMNIST(f'{data_dir}', train=False)

  idx = torch.randperm(len(mnist_train))
  train_idx, val_idx = idx[:-10000], idx[-10000:]
  mnist_train.filter_by_idx(train_idx)
  mnist_val.filter_by_idx(val_idx)

  prev_params = []
  for t in range(5):
    mnist_train.filter_by_class([2 * t, 2 * t + 1])
    mnist_val.filter_by_class(range(2 * t + 2))
    mnist_test.filter_by_class(range(2 * t + 2))
    
    state_dict = train(t, mnist_train, mnist_val, mnist_test,
                       epochs=epochs, M=M, lr=lr, beta=beta, batch_size=batch_size,
                       prev_params=prev_params, logger=logger, device=device)

    prev_params.append(state_dict)

  logger.close()


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = 'run' if os.environ.get('IS_UBUILD') else 'dryrun'

  import fire
  fire.Fire(main)
