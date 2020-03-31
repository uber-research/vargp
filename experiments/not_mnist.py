import os
from tqdm.auto import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from continual_gp.datasets import SplitNotMNIST
from continual_gp.train_utils import set_seeds, train


def main(data_dir='/tmp', epochs=500, M=60, lr=3e-3,
         batch_size=512, beta=10.0, ep_var_mean=True, map_est_hypers=False,
         seed=None):
  set_seeds(seed)

  wandb.init(tensorboard=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger = SummaryWriter(log_dir=wandb.run.dir)

  not_mnist_train = SplitNotMNIST(f'{data_dir}', train=True)
  not_mnist_val = SplitNotMNIST(f'{data_dir}', train=True)
  not_mnist_test = SplitNotMNIST(f'{data_dir}', train=False)

  idx = torch.randperm(len(not_mnist_train))
  train_idx, val_idx = idx[:-4000], idx[-4000:]
  not_mnist_train.filter_by_idx(train_idx)
  not_mnist_val.filter_by_idx(val_idx)

  prev_params = []
  for t in range(5):
    not_mnist_train.filter_by_class([2 * t, 2 * t + 1])
    not_mnist_val.filter_by_class(range(2 * t + 2))
    not_mnist_test.filter_by_class(range(2 * t + 2))

    state_dict = train(t, not_mnist_train, not_mnist_val, not_mnist_test,
                       epochs=epochs, M=M, lr=lr, beta=beta, batch_size=batch_size,
                       ep_var_mean=bool(ep_var_mean), map_est_hypers=bool(map_est_hypers),
                       prev_params=prev_params, logger=logger, device=device)

    prev_params.append(state_dict)

  logger.close()


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = 'run' if os.environ.get('IS_UBUILD') else 'dryrun'

  import fire
  fire.Fire(main)