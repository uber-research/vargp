# Continual Learning with GPs

This repository hosts experiments for
*Variational Auto-Regressive Gaussian Processes for Continual Learning*.

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

CLI Arguments: See `main` method.

```shell
python experiments/toy.py
```

### Split MNIST

CLI Arguments: See `split_mnist` method.

```shell
python experiments/mnist.py s-mnist
```

### Permuted MNIST

CLI Arguments: See `permuted_mnist` method.

```shell
python experiments/mnist.py p-mnist
```

### Checkpoints and Graphs

The checkpoints files can be downloaded [here](#). Extract
contents of the zip file into `notebooks/results`.

All graphs in the paper can now be generated via
code in the notebooks.
