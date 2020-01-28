from data_utils import get_toy_cla_four
import sys

sys.path.append("../")
from models.hypers_ml.models import SparseGP
from models.hypers_ml.kernels import RBFKernel
from models.hypers_ml.likelihoods import MulticlassSoftmax
from torch.optim import Adam
import torch
import numpy as np
import matplotlib.pylab as plt
import random
from plot_utils import colors, number_formatter
import pdb


def plot_prediction_four(pred_prob, x_train, y_train, X1, X2, zu, title):
    n_classes = pred_prob.shape[1]
    fig, axs = plt.subplots(1, n_classes, figsize=(10, 3.5))
    levels = np.array([0.3, 0.5, 0.9])
    if plt.rcParams["text.usetex"]:
        fmt = r"%r"
    else:
        fmt = "%r"
    for k in range(n_classes):
        pred_prob_k = np.reshape(pred_prob[:, k], (X1.shape[0], X1.shape[1]))
        imshow_handle = axs[k].imshow(
            pred_prob_k,
            extent=(X1.min(), X1.max(), X2.min(), X2.max()),
            origin="lower",
        )
        ind = np.where(y_train == k)
        axs[k].scatter(
            x_train[ind, 0],
            x_train[ind, 1],
            marker="o",
            s=25,
            c="w",
            edgecolors="k",
            alpha=0.8,
        )
        axs[k].scatter(
            zu[k, :, 0],
            zu[k, :, 1],
            marker="x",
            s=15,
            c="w",
            edgecolors="k",
            alpha=0.6,
        )

    ax = plt.axes([0.15, 0.04, 0.7, 0.02])
    plt.title("Probability")
    plt.colorbar(imshow_handle, cax=ax, orientation="horizontal")
    for k in range(n_classes):
        axs[k].set_xlabel("$x_1$")
        if k == 0:
            axs[k].set_ylabel("$x_2$")
        axs[k].set_xlim(np.min(X1), np.max(X1))
        axs[k].set_ylim(np.min(X2), np.max(X2))
    # plt.title(title)
    plt.savefig(
        "/tmp/" + "prediction_cla_" + title + ".pdf",
        bbox_inches="tight",
        pad_inches=0,
    )

if __name__ == "__main__":
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
    k_train = 10
    N_total = x_train.shape[0]
    for e in range(no_epochs):
        optimizer.zero_grad()
        losses = model.loss(x_train, y_train_class, no_samples=k_train)
        # in case when we need to do minibatching
        scale = x_train.shape[0] / N_total
        loss = losses[0] + losses[1]
        loss.backward()
        optimizer.step()
        if e % 10 == 0 or e == no_epochs - 1:
            print("epoch {} / {}, loss {}".format(e, no_epochs, loss.item()))

    k_test = 100
    with torch.no_grad():
        pred_logprobs = model.predict(x_plot, pred_y=True, no_samples=k_test)
        loglik = torch.logsumexp(pred_logprobs, dim=-1) - np.log(k_test)
        pred_probs = torch.exp(loglik)
    zu = model.xu.detach()
    plot_prediction_four(
        pred_probs, x_train, y_train_class, X1, X2, zu, "four_batch_ml"
    )
