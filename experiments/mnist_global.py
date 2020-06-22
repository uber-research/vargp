import os
from tqdm.auto import tqdm
import wandb
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from var_gp.datasets import SplitMNIST, PermutedMNIST
from var_gp.train_utils import set_seeds
from var_gp.train_utils_global import train


def split_mnist(data_dir='/tmp', epochs=500, M=60, lr=3e-3,
                batch_size=512, beta=10.0, map_est_hypers=False,
                seed=None):
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
                       map_est_hypers=bool(map_est_hypers),
                       prev_params=prev_params, logger=logger, device=device)

    prev_params.append(state_dict)

  logger.close()


def permuted_mnist(data_dir='/tmp', n_tasks=10, epochs=1000, M=100, lr=3.7e-3,
                   batch_size=512, beta=1.64, seed=None):
  set_seeds(seed)

  wandb.init(tensorboard=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger = SummaryWriter(log_dir=wandb.run.dir)

  ## NOTE: First task is unpermuted MNIST.
  tasks = [torch.arange(784)] + PermutedMNIST.create_tasks(n=n_tasks - 1)

  mnist_train = PermutedMNIST(f'{data_dir}', train=True)

  idx = torch.randperm(len(mnist_train))
  train_idx, val_idx = idx[:-10000], idx[-10000:]
  mnist_train.filter_by_idx(train_idx)

  mnist_val = []
  mnist_test = []

  prev_params = []
  for t in range(len(tasks)):
    mnist_train = PermutedMNIST(f'{data_dir}', train=True)
    mnist_train.filter_by_idx(train_idx)
    mnist_train.set_task(tasks[t])

    mnist_val.append(PermutedMNIST(f'{data_dir}', train=True))
    mnist_val[-1].filter_by_idx(val_idx)
    mnist_val[-1].set_task(tasks[t])

    mnist_test.append(PermutedMNIST(f'{data_dir}', train=False))
    mnist_test[-1].set_task(tasks[t])

    state_dict = train(t, mnist_train, ConcatDataset(mnist_val), ConcatDataset(mnist_test),
                       epochs=epochs, M=M, lr=lr, beta=beta, batch_size=batch_size,
                       prev_params=prev_params, logger=logger, device=device)

    prev_params.append(state_dict)

  logger.close()


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = 'run' if os.environ.get('IS_UBUILD') else 'dryrun'

  import fire
  fire.Fire(dict(s_mnist=split_mnist, p_mnist=permuted_mnist))
