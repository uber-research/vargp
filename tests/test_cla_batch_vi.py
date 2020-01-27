from data_utils import get_toy_cla_four
import sys

sys.path.append("../")
from models.hypers_vi.models import SparseGP
from models.hypers_vi.kernels import RBFKernel
from models.hypers_vi.likelihoods import MulticlassSoftmax
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pylab as plt
import random
from plot_utils import colors, number_formatter
import pdb
from test_cla_batch_ml import plot_prediction_four


x_train, y_train, x_plot, X1, X2 = get_toy_cla_four()
x_train = torch.from_numpy(x_train).to(torch.float)
y_train_one_hot = torch.from_numpy(y_train)
y_train_class = torch.from_numpy(np.argmax(y_train, 1))
x_plot = torch.from_numpy(x_plot).to(torch.float)

Din = x_train.shape[1]
Dout = 4
kern = RBFKernel(Din)
lik = MulticlassSoftmax()
M = 20
model = SparseGP(Din, Dout, M, kern, lik)
optimizer = Adam(model.parameters(), lr=0.01)
no_epochs = 5000
k_kern_train = 3
k_func_train = 10
N_total = x_train.shape[0]
for e in range(no_epochs):
    optimizer.zero_grad()
    losses = model.loss(
        x_train,
        y_train_class,
        no_kern_samples=k_kern_train,
        no_func_samples=k_func_train,
    )
    # in case when we need to do minibatching
    scale = x_train.shape[0] / N_total
    loss = losses[0] + losses[1]
    loss.backward()
    optimizer.step()
    if e % 10 == 0 or e == no_epochs - 1:
        print("epoch {} / {}, loss {}".format(e, no_epochs, loss.item()))

k_kern_test = 10
k_func_test = 100
with torch.no_grad():
    pred_logprobs = model.predict(
        x_plot,
        pred_y=True,
        no_kern_samples=k_kern_test,
        no_func_samples=k_func_test,
    )
    s = pred_logprobs.shape
    pred_logprobs = pred_logprobs.reshape([s[0], s[1], s[2] * s[3]])
    loglik = torch.logsumexp(pred_logprobs, dim=-1) - np.log(s[2] * s[3])
    pred_probs = torch.exp(loglik)
zu = model.xu.detach()
plot_prediction_four(
    pred_probs, x_train, y_train_class, X1, X2, zu, "four_batch_vi"
)
