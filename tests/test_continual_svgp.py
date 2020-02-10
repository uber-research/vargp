from tqdm.auto import tqdm
import torch
from continual_gp.kernels import RBFKernel
from continual_gp.likelihoods import MulticlassSoftmax
from continual_gp.models import ContinualSVGP
from continual_gp.utils import vec2tril, process_params


def create_class_gp(x_train, out_size, M=20, n_f=10, n_hypers=3, prev_params=None):
  prev_params = process_params(prev_params)

  z = torch.stack([
    x_train[torch.randperm(x_train.size(0))[:M]]
    for _ in range(out_size)])

  prior_log_mean, prior_log_logvar = None, None
  if prev_params is not None:
    prior_log_mean = prev_params[-1].get('kernel.log_mean')
    prior_log_logvar = prev_params[-1].get('kernel.log_logvar')

  kernel = RBFKernel(x_train.size(-1), prior_log_mean=prior_log_mean, prior_log_logvar=prior_log_logvar)
  likelihood = MulticlassSoftmax(n_f=n_f)
  gp = ContinualSVGP(z, kernel, likelihood, n_hypers=n_hypers,
                     prev_params=prev_params)
  return gp


def train_gp(x_train, y_train, n_classes, epochs=int(1e4), prev_params=None):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  gp = create_class_gp(x_train, n_classes,
                       M=20, n_f=10, n_hypers=3,
                       prev_params=prev_params).to(device)
  optim = torch.optim.Adam(gp.parameters(), lr=1e-2)

  for e in tqdm(range(epochs)):
    optim.zero_grad()

    lik_loss = gp.loss(x_train, y_train)

    loss = lik_loss
    loss.backward()
    optim.step()

    if (e + 1) % 500 == 0:
      print(f'Loss: {loss.detach().item()}')

  return gp.state_dict()


def test_gp(gp_state_dict, x_train, y_train, x_test, X1, X2, n_classes, prev_params=None):
  x_test = torch.from_numpy(x_test).float().to(device)

  with torch.no_grad():
    test_gp = create_class_gp(x_train, n_classes,
                              M=20, n_f=100, n_hypers=10,
                              prev_params=prev_params).to(device)
    test_gp.load_state_dict(gp_state_dict)

    y_pred = test_gp.predict(x_test)

    from test_cla_batch_ml import plot_prediction_four
    plot_prediction_four(
        y_pred.cpu(),
        x_train.cpu(), y_train.cpu(),
        X1, X2,
        test_gp.z.cpu().detach(), "four_batch_cgp"
    )


def main():
  from data_utils import get_toy_cla_four
  x_train, y_train, *test_data = get_toy_cla_four()

  x_train = torch.from_numpy(x_train).float().to(device)
  y_train = torch.from_numpy(y_train).float().argmax(dim=-1).to(device)
  n_classes = y_train.unique().size(0)

  prev_params = []

  # Train GP on only classes 0,1
  c01_idx = torch.masked_select(torch.arange(y_train.size(0)).to(device), (y_train == 0) | (y_train == 1))
  c01_gp_state_dict = train_gp(x_train[c01_idx], y_train[c01_idx], n_classes)
  test_gp(c01_gp_state_dict, x_train[c01_idx], y_train[c01_idx], *test_data, n_classes)

  prev_params.append(c01_gp_state_dict)

  # Train GP on only classes 2,3
  c23_idx = torch.masked_select(torch.arange(y_train.size(0)).to(device), (y_train == 2) | (y_train == 3))
  c23_gp_state_dict = train_gp(x_train[c23_idx], y_train[c23_idx], n_classes, prev_params=prev_params)
  test_gp(c23_gp_state_dict, x_train[c23_idx], y_train[c23_idx], *test_data, n_classes, prev_params=prev_params)


if __name__ == "__main__":
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  main()
