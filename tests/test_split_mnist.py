from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from continual_gp.kernels import RBFKernel
from continual_gp.likelihoods import MulticlassSoftmax
from continual_gp.models import ContinualSVGP
from continual_gp.utils import vec2tril, process_params
from continual_gp.datasets import SplitMNIST


def create_class_gp(dataset, M=20, n_f=10, n_hypers=3, prev_params=None):
  prev_params = process_params(prev_params)

  N = len(dataset)
  out_size = torch.unique(dataset.targets).size(0)

  # init inducing points at random data points.
  z = torch.stack([
    dataset[torch.randperm(N)[:M]][0]
    for _ in range(out_size)])

  prior_log_mean, prior_log_logvar = None, None
  if prev_params is not None:
    prior_log_mean = prev_params[-1].get('kernel.log_mean')
    prior_log_logvar = prev_params[-1].get('kernel.log_logvar')

  kernel = RBFKernel(z.size(-1), prior_log_mean=prior_log_mean, prior_log_logvar=prior_log_logvar)
  likelihood = MulticlassSoftmax(n_f=n_f)
  gp = ContinualSVGP(z, kernel, likelihood, n_hypers=n_hypers,
                     prev_params=prev_params)
  return gp


def train_gp(dataset, task_id=-1, epochs=int(1e4), batch_size=512, prev_params=None, logger=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  
  dataset.set_task(task_id)

  gp = create_class_gp(dataset, M=200, n_f=10, n_hypers=3,
                       prev_params=prev_params).to(device)
  optim = torch.optim.Adam(gp.parameters(), lr=1e-2)

  N = len(dataset)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  for e in tqdm(range(epochs)):
    for x, y in tqdm(loader, leave=False):
      optim.zero_grad()

      kl_hypers, kl_u, lik = gp.loss(x.to(device), y.to(device)) 

      loss = kl_hypers + kl_u + (N / x.size(0)) * lik
      loss.backward()
      
      optim.step()

    if (e + 1) % 10 == 0:
      with torch.no_grad():
        count = 0
        for x, y in tqdm(loader, leave=False):
          preds = gp.predict(x.to(device))
          # print(torch.distributions.Categorical(preds).entropy().mean(dim=0))
          count += (preds.argmax(dim=-1) == y.to(device)).sum().item()

        acc = count / len(dataset)

      if logger is not None:
        logger.add_scalar('loss/kl_hypers', kl_hypers, global_step=e + 1)
        logger.add_scalar('loss/kl_u', kl_u, global_step=e + 1)
        logger.add_scalar('loss/lik', lik, global_step=e + 1)
        
        logger.add_scalar('train/acc', acc, global_step=e + 1)

  return gp.state_dict()


def main(data_dir='/tmp', task_id=-1, log_dir=None):
  logger = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

  train_dataset = SplitMNIST(f'{data_dir}/mnist_train', train=True)

  train_gp(train_dataset, task_id=task_id,
           logger=logger)


if __name__ == "__main__":
  import fire
  fire.Fire(main)
