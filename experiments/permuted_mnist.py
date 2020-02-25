import os
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import wandb

from continual_gp.datasets import PermutedMNIST
from continual_gp.train_utils import set_seeds, create_class_gp, compute_accuracy


def train(task_id, train_dataset, eval_dataset,
          epochs=1, M=20, n_f=10, n_hypers=3, batch_size=512, lr=1e-2,
          beta=1.0, prev_params=None, logger=None, device=None):
  gp = create_class_gp(train_dataset, M=M, n_f=n_f, n_hypers=n_hypers,
                       prev_params=prev_params).to(device)

  optim = torch.optim.Adam(gp.parameters(), lr=lr)

  N = len(train_dataset)
  loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  for e in tqdm(range(epochs)):
    for x, y in tqdm(loader, leave=False):
      optim.zero_grad()

      kl_hypers, kl_u, lik = gp.loss(x.to(device), y.to(device))

      loss = beta * kl_hypers + kl_u + (N / x.size(0)) * lik
      loss.backward()

      optim.step()

    if (e + 1) % 50 == 0:
      print(len(train_dataset), len(eval_dataset))
      acc = compute_accuracy(train_dataset, gp, device=device)
      eval_acc = compute_accuracy(eval_dataset, gp, device=device)

      summary = {
        f'task{task_id}/loss/kl_hypers': kl_hypers.detach().item(),
        f'task{task_id}/loss/kl_u': kl_u.detach().item(),
        f'task{task_id}/loss/lik': lik.detach().item(),
        f'task{task_id}/train/acc': acc,
        f'task{task_id}/eval/acc': eval_acc,
      }

      if logger is not None:
        for k, v in summary.items():
          logger.add_scalar(k, v, global_step=e + 1)

        with open(f'{logger.log_dir}/ckpt{task_id}.pt', 'wb') as f:
          torch.save(gp.state_dict(), f)

        wandb.save(f'{logger.log_dir}/ckpt{task_id}.pt')

  return gp.state_dict()


def main(data_dir='/tmp', n_tasks=4, epochs=100, M=200, lr=1e-2, batch_size=512, beta=1.0, seed=None):
  set_seeds(seed)

  wandb.init(tensorboard=True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger = SummaryWriter(log_dir=wandb.run.dir)

  prev_params = []
  ## @NOTE: the first task is unpermuted MNIST for now.
  tasks = [None] + PermutedMNIST.create_tasks(n=n_tasks)

  for t in range(len(tasks)):
    train_dataset = PermutedMNIST(f'{data_dir}/mnist', task=tasks[t], train=True)
    
    test_dataset = ConcatDataset([
      PermutedMNIST(f'{data_dir}/mnist', task=tasks[j], train=True)
      for j in range(t + 1)
    ])

    state_dict = train(t, train_dataset, test_dataset,
                       epochs=epochs, M=M, lr=lr, beta=beta, batch_size=batch_size,
                       prev_params=prev_params, logger=logger, device=device)

    prev_params.append(state_dict)

  logger.close()


if __name__ == "__main__":
  os.environ['WANDB_MODE'] = 'run' if os.environ.get('IS_UBUILD') else 'dryrun'

  import fire
  fire.Fire(main)
