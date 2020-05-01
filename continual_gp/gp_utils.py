import torch
import torch.nn.functional as F


def cholesky(M, eps=1e-4):
  '''
  Compute L, where M = LL^T.
  '''
  I = torch.eye(M.size(-1), device=M.device)
  L = torch.cholesky(M + eps * I , upper=False)
  return L


def rev_cholesky(L):
  '''
  Compute M = LL^T.
  '''
  M = torch.einsum('...ij,...kj->...ik', L, L)
  return M


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


def gp_cond(u, Kzz, Kzx, Kxx, Lz=None, Lz_Kzx=None):
  '''
  Compute the GP predictive conditional
  p(f|u,x,z)

  μ = Kxz Kzz_inv u
  Σ = Kxx - Kxz Kzz_inv Kzx

  Arguments:
    m: ... x M x 1
    Kzz: ... x M x M
    Kzx: ... x M x N
    Kxx: ... x N x N

  Returns:
    μ: ... x N x 1
    Σ: ... x N x N
  '''
  if Lz is None:
    Lz = cholesky(Kzz)

  Lz_u, _ = torch.triangular_solve(u, Lz, upper=False)

  if Lz_Kzx is None:
    Lz_Kzx, _ = torch.triangular_solve(Kzx, Lz, upper=False)

  μ = torch.einsum('...ij,...ik->...jk', Lz_Kzx, Lz_u)

  Σ = Kxx - torch.einsum('...ij,...ik->...jk', Lz_Kzx, Lz_Kzx)

  return μ, Σ


def linear_joint(m, S, Kzx, Kzz, V, b, cache=None):
  '''
  Compute product of Gaussian densities of the form
  p(x,y) = N(x; m, S)N(y; Ax + b, V) where A = Kxz Kzz_inv.

  Results in N([x,y]; μ, Σ) where
  μ = [m, Am + b]
  Σ = [S, SA^T; AS, V + ASA^T]

  Arguments:
    m: ... x M x 1
    S: ... x M x M
    Kzx: ... x M x N
    Kzz: ... x M x M
    b: ... x N x 1
    V: ... x N x N

  Returns:
    μ: ... x (M + N) x 1
    Σ: ... x (M + N) x (M + N)
  '''

  Lz = cholesky(Kzz)
  Lz_m, _ = torch.triangular_solve(m, Lz, upper=False)
  Lz_Kzx, _ = torch.triangular_solve(Kzx, Lz, upper=False)

  Am = torch.einsum('...ij,...ik->...jk', Lz_Kzx, Lz_m)

  Lz_S, _ = torch.triangular_solve(S, Lz, upper=False)

  AS = torch.einsum('...ij,...ik->...jk', Lz_Kzx, Lz_S)
  SAt = torch.einsum('...ij->...ji', AS)

  Lz_SAt, _ = torch.triangular_solve(SAt, Lz, upper=False)

  ASAt = torch.einsum('...ij,...ik->...jk', Lz_SAt, Lz_Kzx)

  μ = torch.cat([m, Am + b], dim=-2)
  Σ = torch.cat([
    torch.cat([S, SAt], dim=-1),
    torch.cat([AS, V + ASAt], dim=-1)
  ], dim=-2)

  if isinstance(cache, dict):
    cache.update(dict(Lz_Kzx=Lz_Kzx, Lz=Lz))

  return μ, Σ


def linear_marginal_diag(m, S, Kzz, Kzx, Kxx_diag, cache=None):
  '''
  Compute the diagonal of the marginal of
  product of Gaussian densities of the form
  p(x,y) = N(x; m, S)N(y; Ax, V) where A = Kxz Kzz_inv.

  This function combines `gp_cond`, `linear_joint`
  and diagonalization into a single efficient function.

  μ = Am + b
  Σ = V + ASA^T

  Arguments:
    m: ... x M x 1
    S: ... x M x M
    Kzx: ... x M x N
    Kzz: ... x M x M
    Kxx_diag: ... x N x N

  Returns:
    μ: ... x N
    Σ: ... x N
  '''

  Lz = cholesky(Kzz)
  Lz_m, _ = torch.triangular_solve(m, Lz, upper=False)
  Lz_Kzx, _ = torch.triangular_solve(Kzx, Lz, upper=False)

  μ = torch.einsum('...ij,...ik->...jk', Lz_Kzx, Lz_m).squeeze(-1)

  diag1 = Lz_Kzx.pow(2).sum(dim=-2)

  Lz_LS, _ = torch.triangular_solve(cholesky(S), Lz, upper=False)

  diag2 = torch.einsum('...ij,...ik->...jk', Lz_LS, Lz_Kzx).pow(2).sum(dim=-2)

  Σ = Kxx_diag - diag1 + diag2

  if isinstance(cache, dict):
    cache.update(dict(Lz=Lz, Lz_Kzx=Lz_Kzx))

  return μ, Σ
