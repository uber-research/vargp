import torch
import torch.nn.functional as F


def vec2tril(vec, m=None):
  '''
  Arguments:
    vec: K x ((m * (m + 1)) // 2)
    m: integer, if None, inferred from last dimension.

  Returns:
    Batch of lower triangular matrices

    tril: K x m x m
  '''
  if m is None:
    D = vec.size(-1)
    m = (torch.tensor(8. * D + 1).sqrt() - 1.) / 2.
    m = m.long().item()

  batch_shape = vec.shape[:-1]

  idx = torch.tril_indices(m, m)

  tril = torch.zeros(*batch_shape, m, m, device=vec.device)
  tril[..., idx[0], idx[1]] = vec

  # ensure positivity constraint of cholesky diagonals
  mask = torch.eye(m, device=vec.device).bool()
  tril = torch.where(mask, F.softplus(tril), tril)

  return tril


def process_params(params):
  if params is None:
    return None

  def process(p):
    if 'u_tril_vec' in p:
      p['u_tril'] = vec2tril(p.pop('u_tril_vec'))
    return p

  return [process(p) for p in params]
