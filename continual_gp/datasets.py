import torch
from torchvision.datasets import MNIST


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
