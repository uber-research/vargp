import torch
from torchvision.datasets import MNIST


class SplitMNIST(MNIST):
  def __init__(self, *args, **kwargs):
    kwargs['download'] = True
    super().__init__(*args, **kwargs)

    self.data = self.data.reshape(self.data.size(0), -1).float() / 255.

    self.task_ids = torch.arange(self.targets.size(0))

  def set_task(self, i):
    '''
    There are five tasks implemented as a binary classifications
    and each i refers to the classification of (2i) / (2i+1) classes.
    e.g. task i = 4 will give out data for classes 8/9.
    '''
    assert -1 <= i <= 4, 'task IDs can be -1, 0, 1, 2, 3, 4'

    if i == -1:
      self.task_ids = torch.arange(self.targets.size(0))
      return
    
    self.task_ids = torch.masked_select(torch.arange(self.targets.size(0)),
                                       (self.targets == 2 * i) | (self.targets == 2 * i + 1))

    ## Only keep desired data and remap 2i -> 0/ 2i + 1 -> 1
    # self.data = self.data[self.task_ids]
    # self.targets = self.targets[self.task_ids] % 2
    # self.task_ids = torch.arange(self.targets.size(0))

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
