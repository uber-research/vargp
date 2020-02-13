import os
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from continual_gp.kernels import RBFKernel
from continual_gp.likelihoods import MulticlassSoftmax
from continual_gp.models import ContinualSVGP
from continual_gp.utils import vec2tril, process_params
from continual_gp.datasets import SplitMNIST


def compute_accuracy(dataset, gp, device=None):
  loader = DataLoader(dataset, batch_size=512)

  with torch.no_grad():
    count = 0
    for x, y in tqdm(loader, leave=False):
      preds = gp.predict(x.to(device))
      # print(torch.distributions.Categorical(preds).entropy().mean(dim=0))
      count += (preds.argmax(dim=-1) == y.to(device)).sum().item()

    acc = count / len(dataset)
  
  return acc


def create_class_gp(dataset, M=20, n_f=10, n_hypers=3, prev_params=None):
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

  kernel = RBFKernel(z.size(-1), prior_log_mean=prior_log_mean, prior_log_logvar=prior_log_logvar)
  likelihood = MulticlassSoftmax(n_f=n_f)
  gp = ContinualSVGP(z, kernel, likelihood, n_hypers=n_hypers,
                     prev_params=prev_params)
  return gp


def train_gp(train_dataset, eval_dataset,
             epochs=1, M=20, n_f=10, n_hypers=3, batch_size=512, lr=1e-2,
             prev_params=None, logger=None, device=None):  
  gp = create_class_gp(train_dataset, M=M, n_f=n_f, n_hypers=n_hypers,
                       prev_params=prev_params).to(device)
  
  # with open('logs/mnist/ckpt.pt', 'rb') as f:
  #   state_dict = torch.load(f)
  #   gp.kernel.log_mean.data = state_dict.get('kernel.log_mean')
  #   gp.kernel.log_logvar.data = state_dict.get('kernel.log_logvar')

  optim = torch.optim.Adam(gp.parameters(), lr=lr)

  N = len(train_dataset)
  loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  for e in tqdm(range(epochs)):
    for x, y in tqdm(loader, leave=False):
      optim.zero_grad()

      kl_hypers, kl_u, lik = gp.loss(x.to(device), y.to(device)) 

      loss = kl_hypers + kl_u + (N / x.size(0)) * lik
      loss.backward()
      
      optim.step()

    if (e + 1) % 10 == 0:
      acc = compute_accuracy(train_dataset, gp, device=device)
      eval_acc = compute_accuracy(eval_dataset, gp, device=device)

      if logger is not None:
        logger.add_scalar('loss/kl_hypers', kl_hypers, global_step=e + 1)
        logger.add_scalar('loss/kl_u', kl_u, global_step=e + 1)
        logger.add_scalar('loss/lik', lik, global_step=e + 1)
        
        logger.add_scalar('train/acc', acc, global_step=e + 1)
        logger.add_scalar('eval/acc', eval_acc, global_step=e + 1)

        with open(f'{logger.log_dir}/ckpt.pt', 'wb') as f:
          torch.save(gp.state_dict(), f)

  return gp.state_dict()


def main(data_dir='/tmp', epochs=500, n_inducing_points=20, lr=5e-3, log_dir=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  prev_params = []

  for t in range(5):
    if log_dir and os.path.isfile(f'{log_dir}/{t}/ckpt.pt'):
      with open(f'{log_dir}/{t}/ckpt.pt', 'rb') as f:
        prev_params.append(torch.load(f))
      print(f'Loaded {log_dir}/{t}/ckpt.pt')
      continue

    train_dataset = SplitMNIST(f'{data_dir}/mnist', train=True)
    train_dataset.filter_classes([2 * t, 2 * t + 1])

    eval_dataset = SplitMNIST(f'{data_dir}/mnist', train=True)
    eval_dataset.filter_classes(range(2 * t + 2))

    logger = SummaryWriter(log_dir=f'{log_dir}/{t}') if log_dir is not None else None
    state_dict = train_gp(train_dataset, eval_dataset,
                          epochs=epochs, M=n_inducing_points, lr=lr,
                          prev_params=prev_params, logger=logger, device=device)
    if logger is not None:
      logger.close()

    ## Comment this line to disable continual learning. 
    prev_params.append(state_dict)


if __name__ == "__main__":
  import fire
  fire.Fire(main)
