# Continual Learning with GPs

This repository hosts code for
[Variational Auto-Regressive Gaussian Processes for Continual Learning](https://arxiv.org/abs/2006.05468) by [_Sanyam Kapoor_](https://www.sanyamkapoor.com), [_Theofanis Karaletsos_](http://karaletsos.com), [_Thang D. Bui_](https://thangbui.github.io).

## Setup

From the root of the directory

```shell
conda env create
```

This creates the environment from [environment.yml](./environment.yml).

In addition, ensure availability of the this directory in `PYTHONPATH`
for module imports. As an example in Bash,

```shell
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

## Experiments

All experiment scripts utilize [Fire](https://github.com/google/python-fire)
and the arguments can directly be converted to CLI arguments. Appropriate
functions are mentioned to look for CLI arguments to change. The default arguments are enough to reproduce results in the paper.

### Toy Dataset

CLI Arguments: See `toy` method.

```shell
python experiments/vargp.py toy
```

### Split MNIST

CLI Arguments: See `split_mnist` method.

```shell
python experiments/vargp.py s-mnist
```

### Permuted MNIST

CLI Arguments: See `permuted_mnist` method.

```shell
python experiments/vargp.py p-mnist
```

### Checkpoints and Graphs

The checkpoints files can be downloaded [here](https://bit.ly/var-gp-results). Extract
contents of the zip file into `notebooks/results`.

All graphs in the paper can now be generated via code in the notebooks.
