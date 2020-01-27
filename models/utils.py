import numpy as np
import torch
import torch.nn as nn
from torch import triangular_solve
import pdb


def kl_full_gaussian(mean1, tril1, mean2, tril2):
    const_term = -0.5 * mean1.shape[0] * mean1.shape[1]
    logdet_prior = torch.sum(
        torch.log(torch.diagonal(tril2, offset=0, dim1=-2, dim2=-1))
    )
    logdet_post = torch.sum(
        torch.log(torch.diagonal(tril1, offset=0, dim1=-2, dim2=-1))
    )
    logdet_term = logdet_prior - logdet_post
    # in the latest version of pytorch you can do this
    # LpiLq = torch.triangular_solve(tril1, tril2, upper=False)
    LpiLq = torch.triangular_solve(tril1, tril2, upper=False)[0]
    trace_term = 0.5 * torch.sum(LpiLq ** 2)
    mu_diff = mean1 - mean2
    # in the latest version of pytorch you can do this
    # quad_solve = torch.triangular_solve(mu_diff, tril2, upper=False)
    quad_solve = torch.triangular_solve(mu_diff, tril2, upper=False)[0]
    quad_term = 0.5 * torch.sum(quad_solve ** 2)
    kl = const_term + logdet_term + trace_term + quad_term
    return kl


def kl_diagonal_gaussian(mean1, var1, mean2, var2):
    no_params = mean1.shape[0]
    const_term = -0.5 * no_params
    log_var_diff = torch.log(var2) - torch.log(var1)
    log_std_diff = 0.5 * torch.sum(log_var_diff)
    mu_diff_term = (var1 + (mean1 - mean2) ** 2) / var2
    mu_diff_term = 0.5 * torch.sum(mu_diff_term)
    kl = const_term + log_std_diff + mu_diff_term
    return kl
