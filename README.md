# FigBO

This repository is the official implementation of FigBO: A Generalized Acquisition Function Framework with Look-Ahead Capability for Bayesian Optimization.

FigBO is a Bayesian optimization method based on a gamma-designed acquisition strategy, enabling a flexible and principled trade-off between exploration and exploitation.


# Installation

Clone the repository:

```bash
git clone https://github.com/HuiChen3313/FigBO.git
cd FigBO
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```
The code is implemented in Python and relies on PyTorch and BoTorch for Bayesian optimization.


# Requirements

The main dependencies are listed in requirements.txt, including:

PyTorch

BoTorch

GPyTorch

Ax-platform

NumPy

SciPy

Matplotlib

Hydra

# Implemented Acquisition Functions

This repository implements the following Bayesian optimization acquisition functions:

FigBO (proposed method)

Expected Improvement (EI)

Noisy Expected Improvement (NEI)

qLog Noisy Expected Improvement (qLogNEI)

Probability of Improvement (PI)

Upper Confidence Bound (UCB)


# FigBO (Proposed)

Key: FigBO

Implementation: GammaExpectedImprovement

FigBO is the proposed acquisition function described in the paper. In the codebase, it is implemented using GammaExpectedImprovement, which realizes the gamma-based design introduced in the method. All experimental results reported as FigBO in the paper are generated using this implementation.

Note: The name GammaExpectedImprovement reflects an internal implementation detail related to the gamma-based formulation and does not indicate a separate baseline or variant.
