# FigBO

This repository is the official implementation of FigBO: A Generalized Acquisition Function Framework with Look-Ahead Capability for Bayesian Optimization.

FigBO is a Bayesian optimization method based on a gamma-designed acquisition strategy, enabling a flexible and principled trade-off between exploration and exploitation.


## Installation

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


## Requirements

The main dependencies are listed in requirements.txt, including:

- torch==2.2.0
- botorch==0.12.0
- gpytorch==1.13
- ax-platform==0.4.3
- numpy==1.26.4
- scipy==1.13.1
- matplotlib==3.7.2
- hydra-core==1.3.2


## FigBO Algorithm

Key: FigBO

Implementation: GammaExpectedImprovement (internal)

FigBO is the proposed acquisition framework described in the paper. In the codebase, it is implemented using an internal module referred to as GammaExpectedImprovement, which realizes the Γ(x)-based look-ahead design introduced in the method. All experimental results reported as FigBO in the paper are generated using this implementation.

Note that GammaExpectedImprovement is an internal implementation detail of the FigBO framework, rather than an official acquisition function provided by BoTorch. The implementation is built on top of BoTorch’s GP posterior interfaces without modifying the BoTorch source code.


## Usage
### Running FigBO

To run the proposed FigBO method:
```bash
python main.py acq=FigBO
```

### Running Baselines

To run a baseline acquisition function (e.g., Expected Improvement):
```bash
python main.py acq=EI
```

## Result
All figures and tables in the paper are generated from the CSV and JSON files produced by running `main.py` with the corresponding configuration files and random seeds. These files record the optimization trajectories, evaluations, and model hyperparameters used for analysis

## Folder Structure
- `main.py`: Main entry point for running Bayesian optimization experiments.
- `configs/`: Hydra configuration files specifying benchmarks, acquisition functions, models, and random seeds.
- `benchmarking/`: Benchmark functions, acquisition implementations, and evaluation utilities.
- `results/`: Directory where CSV and JSON result files are saved after running experiments.



