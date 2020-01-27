import numpy as np


def get_toy_cla_four(N_train=50):
    np.random.seed(0)
    # creating a multiclass dataset
    n_classes = 4
    N_perclass = int(N_train * 1.0 / n_classes)
    N_train = N_perclass * n_classes

    X1 = np.hstack(
        (
            0.8 + 0.4 * np.random.randn(N_perclass, 1),
            1.5 + 0.4 * np.random.randn(N_perclass, 1),
        )
    )
    X2 = np.hstack(
        (
            0.5 + 0.6 * np.random.randn(N_perclass, 1),
            -0.2 - 0.1 * np.random.randn(N_perclass, 1),
        )
    )
    X3 = np.hstack(
        (
            2.5 - 0.1 * np.random.randn(N_perclass, 1),
            1.0 + 0.6 * np.random.randn(N_perclass, 1),
        )
    )
    X4 = np.random.multivariate_normal(
        [-0.5, 1.5], [[0.2, 0.1], [0.1, 0.1]], N_perclass
    )
    x = np.vstack((X1, X2, X3, X4))
    x[:, 1] -= 1
    x[:, 0] -= 0.5
    y = np.zeros((N_train, n_classes))
    for i in range(n_classes):
        y[i * N_perclass : (i + 1) * N_perclass, i] = 1
    D = np.hstack((x, y))
    np.random.shuffle(D)
    x = D[:, :-n_classes]
    y = D[:, -n_classes:]
    merged = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
    np.random.shuffle(merged)
    x = merged[:, : x.shape[1]]
    y = merged[:, x.shape[1] :]
    delta = 0.2
    x1 = np.arange(-4, 4, delta)
    x2 = np.arange(-4, 4, delta)
    X1, X2 = np.meshgrid(x1, x2)
    X11 = X1.reshape([X1.shape[0] * X1.shape[1], 1])
    X22 = X2.reshape([X2.shape[0] * X2.shape[1], 1])
    X_plot = np.concatenate((X11, X22), axis=1)
    return x, y, X_plot, X1, X2


def get_data_toy_cl_1(index):
    # creating a multiclass dataset
    N_perclass = 100
    n_classes = 4
    X1 = np.hstack(
        (
            0.8 + 0.2 * np.random.randn(N_perclass, 1),
            1.5 + 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X3 = np.hstack(
        (
            0.2 + 0.3 * np.random.randn(N_perclass, 1),
            -0.2 - 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X2 = np.hstack(
        (
            1.5 - 0.1 * np.random.randn(N_perclass, 1),
            1.0 + 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X4 = np.random.multivariate_normal(
        [-0.5, 1.5], [[0.2, 0.1], [0.1, 0.1]], N_perclass
    )
    X5 = np.hstack(
        (
            0.2 + 0.3 * np.random.randn(N_perclass, 1),
            -0.2 - 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X6 = np.hstack(
        (
            1.5 - 0.1 * np.random.randn(N_perclass, 1),
            1.0 + 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    x = np.vstack((X1, X2, X3, X4, X5, X6))
    x[:, 1] -= 1
    y = np.zeros((x.shape[0]))
    for i in range(n_classes):
        y[i * N_perclass : (i + 1) * N_perclass] = i
    y[-2 * N_perclass : -N_perclass] = 1
    y[-N_perclass:] = 2
    x, y = x.astype(np_ftype), y.astype(np_ftype)

    ind_0 = np.arange(N_perclass)
    ind_1 = np.arange(N_perclass, 2 * N_perclass)
    ind_2 = np.arange(2 * N_perclass, 3 * N_perclass)
    ind_3 = np.arange(3 * N_perclass, 4 * N_perclass)
    ind_4 = np.arange(4 * N_perclass, 5 * N_perclass)
    ind_5 = np.arange(5 * N_perclass, 6 * N_perclass)
    #
    # ind = [np.hstack([ind_0, ind_2]), np.hstack([ind_1, ind_3]), np.hstack([ind_4, ind_5]), np.hstack([ind_0, ind_2, ind_1, ind_3])]
    ind = [ind_0, ind_1, ind_2, ind_3, np.hstack([ind_0, ind_2, ind_1, ind_3])]
    return x[ind[index], :].astype(np_ftype), y[ind[index]].astype(np_itype)


def get_data_toy_cl_2(index):
    # creating a multiclass dataset
    N_perclass = 100
    n_classes = 4
    X1 = np.hstack(
        (
            0.8 + 0.2 * np.random.randn(N_perclass, 1),
            1.5 + 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X2 = np.hstack(
        (
            0.2 + 0.4 * np.random.randn(N_perclass, 1),
            -0.2 - 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X3 = np.hstack(
        (
            1.5 - 0.1 * np.random.randn(N_perclass, 1),
            0.5 + 0.2 * np.random.randn(N_perclass, 1),
        )
    )
    X4 = np.random.multivariate_normal(
        [-0.6, 1.5], [[0.2, 0.1], [0.1, 0.1]], N_perclass
    )
    x = np.vstack((X1, X2, X3, X4))
    x[:, 1] -= 1
    y = np.zeros((x.shape[0]))
    for i in range(n_classes):
        y[i * N_perclass : (i + 1) * N_perclass] = i
    x, y = x.astype(np_ftype), y.astype(np_ftype)

    ind_0 = np.where(x[:, 0] < -0.5)[0]
    ind_1 = np.where(np.logical_and(x[:, 0] > -0.5, x[:, 0] < 0))[0]
    ind_2 = np.where(np.logical_and(x[:, 0] > 0, x[:, 0] < 0.8))[0]
    ind_3 = np.where(x[:, 0] > 0.8)[0]

    ind = [ind_0, ind_1, ind_2, ind_3, np.hstack([ind_0, ind_1, ind_2, ind_3])]
    return x[ind[index], :].astype(np_ftype), y[ind[index]].astype(np_itype)
