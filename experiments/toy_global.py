import os
from tqdm.auto import tqdm
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter

from var_gp.datasets import ToyDataset
from var_gp.train_utils import set_seeds
from var_gp.train_utils_global import train


def main(data_dir=None, epochs=10000, M=20, lr=1e-2,
         batch_size=512, beta=1.0, map_est_hypers=False,
         seed=None):
  data_dir = data_dir or os.environ.get('USER_DATADIR', default='/tmp')
  set_seeds(seed)

  wandb.init(tensorboard=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger = SummaryWriter(log_dir=wandb.run.dir)

  toy_train = ToyDataset()
  toy_val = ToyDataset(X=toy_train.data.clone(), Y=toy_train.targets.clone())
  toy_test = ToyDataset(X=toy_train.data.clone(), Y=toy_train.targets.clone())

  prev_params = []
  for t in range(2):
    toy_train.filter_by_class([2 * t, 2 * t + 1])
    toy_val.filter_by_class(range(2 * t + 2))
    toy_test.filter_by_class(range(2 * t + 2))

    state_dict = train(t, toy_train, toy_val, toy_test,
                       epochs=epochs, M=M*(t+1), lr=lr, beta=beta, batch_size=batch_size,
                       map_est_hypers=bool(map_est_hypers),
                       prev_params=prev_params, logger=logger, device=device)

    prev_params = [state_dict]

  logger.close()


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')

  import fire
  fire.Fire(main)
