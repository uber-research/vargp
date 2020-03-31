import os
import glob
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, VisionDataset
from torchvision.datasets.utils import makedir_exist_ok, download_and_extract_archive


class ToyDataset(Dataset):
  def __init__(self, N_K=50, K=4, X=None, Y=None):
    super().__init__()

    if X is not None:
      self.data, self.targets = X, Y
    else:
      self.data, self.targets = self._init_data(N_K, K)

    self.task_ids = torch.arange(self.targets.size(0))

  def _init_data(self, N_K, K):
    X1 = torch.cat([
      0.8 + 0.4 * torch.randn(N_K, 1),
      1.5 + 0.4 * torch.randn(N_K, 1),
    ], dim=-1)
    Y1 = 0 * torch.ones(X1.size(0)).long()

    X2 = torch.cat([
      0.5 + 0.6 * torch.randn(N_K, 1),
      -0.2 - 0.1 * torch.randn(N_K, 1),
    ], dim=-1)
    Y2 = 1 * torch.ones(X2.size(0)).long()

    X3 = torch.cat([
      2.5 - 0.1 * torch.randn(N_K, 1),
      1.0 + 0.6 * torch.randn(N_K, 1),
    ], dim=-1)
    Y3 = 2 * torch.ones(X3.size(0)).long()

    X4 = torch.distributions.MultivariateNormal(
            torch.Tensor([-0.5, 1.5]),
            covariance_matrix=torch.Tensor([[0.2, 0.1], [0.1, 0.1]])).sample(torch.Size([N_K]))
    Y4 = 3 * torch.ones(X4.size(0)).long()

    X = torch.cat([X1, X2, X3, X4], dim=0)
    X[:, 1] -= 1
    X[:, 0] -= 0.5

    Y = torch.cat([Y1, Y2, Y3, Y4])

    return X, Y

  def filter_by_class(self, class_list=None):
    if class_list:
      mask = torch.zeros_like(self.targets).bool()
      for c in class_list:
        mask |= self.targets == c
    else:
      mask = torch.ones_like(self.targets).bool()

    self.task_ids = torch.masked_select(torch.arange(self.targets.size(0)), mask)

  def __getitem__(self, index):
    return self.data[self.task_ids[index]], self.targets[self.task_ids[index]]

  def __len__(self):
    return self.task_ids.size(0)


class SplitMNIST(MNIST):
  def __init__(self, *args, **kwargs):
    kwargs['download'] = True
    super().__init__(*args, **kwargs)

    self.data = self.data.reshape(self.data.size(0), -1).float() / 255.

    self.task_ids = torch.arange(self.targets.size(0))

  def filter_by_class(self, class_list=None):
    if class_list:
      mask = torch.zeros_like(self.targets).bool()
      for c in class_list:
        mask |= self.targets == c
    else:
      mask = torch.ones_like(self.targets).bool()

    self.task_ids = torch.masked_select(torch.arange(self.targets.size(0)), mask)

  def filter_by_idx(self, idx):
    self.data = self.data[idx]
    self.targets = self.targets[idx]
    self.task_ids = torch.arange(self.targets.size(0))

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    return self.data[self.task_ids[index]], self.targets[self.task_ids[index]]

  def __len__(self):
    return self.task_ids.size(0)


class PermutedMNIST(MNIST):
  @staticmethod
  def create_tasks(n=1):
    return [torch.randperm(784) for _ in range(n)]

  def __init__(self, *args, **kwargs):
    kwargs['download'] = True
    super().__init__(*args, **kwargs)

    self.data = self.data.reshape(self.data.size(0), -1).float() / 255.
    self.perm = None

  def set_task(self, perm):
    assert self.perm is None, 'Cannot set task again.'

    self.data = self.data[:, perm]
    self.perm = perm

  def filter_by_idx(self, idx):
    self.data = self.data[idx]
    self.targets = self.targets[idx]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    return self.data[index], self.targets[index]


class SplitNotMNIST(VisionDataset):
  resources = [
      ("http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz", "c9890a473a9769fda4bdf314aaf500dd"),
  ]

  classes = ['0 - A', '1 - B', '2 - C', '3 - D', '4 - E',
             '5 - F', '6 - G', '7 - H', '8 - I', '9 - J']

  def __init__(self, root, train=True, transform=None, target_transform=None,
               download=True):
    super().__init__(root, transform=transform, target_transform=target_transform)
    self.train = train

    if download:
      self.download()

    self.data = []
    self.targets = []

    for l in range(10):
      d = os.path.join(self.raw_folder, 'notMNIST_small', chr(65 + l))
      for p in glob.glob(f'{d}/*.png'):
        try:
          im = np.expand_dims(np.array(Image.open(p).getdata()), axis=0)
          self.data.append(im)
          self.targets.append(l)
        except UnidentifiedImageError:
          pass

    self.data = torch.from_numpy(np.vstack(self.data)).float() / 255.
    self.targets = torch.from_numpy(np.array(self.targets)).long()

    self.task_ids = torch.arange(self.targets.size(0))

  def filter_by_class(self, class_list=None):
    if class_list:
      mask = torch.zeros_like(self.targets).bool()
      for c in class_list:
        mask |= self.targets == c
    else:
      mask = torch.ones_like(self.targets).bool()

    self.task_ids = torch.masked_select(torch.arange(self.targets.size(0)), mask)

  def __getitem__(self, index):
    return self.data[self.task_ids[index]], self.targets[self.task_ids[index]]

  def __len__(self):
    return len(self.task_ids)

  def filter_by_idx(self, idx):
    self.data = self.data[idx]
    self.targets = self.targets[idx]
    self.task_ids = torch.arange(self.targets.size(0))

  @property
  def raw_folder(self):
      return os.path.join(self.root, self.__class__.__name__, 'raw')

  @property
  def processed_folder(self):
      return os.path.join(self.root, self.__class__.__name__, 'processed')

  def _check_exists(self):
      return os.path.exists(self.processed_folder)

  def download(self):
      """Download the notMNIST data if it doesn't exist in processed_folder already."""

      if self._check_exists():
          return

      makedir_exist_ok(self.raw_folder)
      makedir_exist_ok(self.processed_folder)

      # download files
      for url, md5 in self.resources:
          filename = url.rpartition('/')[2]
          download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
