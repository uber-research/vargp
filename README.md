# Continual Learning with GPs

[![](https://img.shields.io/badge/arXiv-2006.05468-red)](https://u.perhapsbay.es/vargp-arxiv)
[![](https://img.shields.io/badge/ICML-2021-brightgreen)](https://u.perhapsbay.es/vargp)

This repository hosts the code for 
[_Variational Auto-Regressive Gaussian Processes for Continual Learning_](https://u.perhapsbay.es/vargp) (**VAR-GPs**)
by [Sanyam Kapoor](https://im.perhapsbay.es), 
[Theofanis Karaletsos](https://karaletsos.com),
[Thang D. Bui](https://thangbui.github.io).

## What can VAR-GPs do?

By first, only training on classes 0/1 (*Task 0*), and next, only training on classes 2/3 (*Task 1*), we show below the posterior predictive density surface for a classifier on a synthetic classification dataset with four classes. The marks in red are the training data for each class. Qualitatively, brighter regions represent a higher predictive probability that the classifier assigns to the corresponding class in each figure.

| ![](https://i.imgur.com/J8EDeEK.png) | ![](https://i.imgur.com/nT4ka2a.png) |
|---|---|
| **VAR-GPs** preserve information from *Task 0* even after training on *Task 1*, i.e. prevent catastrophic forgetting (seen as bright regions preserved around data from *Task 0*).  | VCL (our main baseline), on the other hand, is overconfident in its predictions, often in the wrong regions, and is structurally less stable (seen as large variations in predictive probabilities between *Task 0* and *Task 1*)  | 

## Setup

From the root of the directory

```shell
conda env create -f environment.yml
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

The checkpoints files are available under [notesbooks/results](./notebooks/results).
Use [Git LFS](https://git-lfs.github.com) to make sure these are pulled
alongside the repository.

**TIP**: Use `git lfs fetch` if Git LFS was installed after the first clone.

All graphs in the paper can now be generated via code in the notebooks.

## License

Apache 2.0

_For research purpose only. Support and/or new releases may be limited._
