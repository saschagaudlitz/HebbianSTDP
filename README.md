# Hebbian STDP Simulation Code

This repository contains simulation code for the paper "[Spike-timing-dependent Hebbian learning as noisy gradient descent](https://www.arxiv.org/abs/2505.10272)". The code implements and visualizes Hebbian Spike-Timing-Dependent Plasticity (STDP) learning dynamics in Julia.

## Overview

The simulation explores how Hebbian STDP can be interpreted as a form of noisy gradient descent. It provides comprehensive visualizations of loss landscapes and learning trajectories in both single and multiple neuron scenarios.

## Features

- Visualization of loss landscapes on probability simplexes
- Gradient flow visualizations
- Single weight vector learning trajectories on the probability simplex
- Multiple weight vector learning with orthogonalization
- Allows for correlated inputs via specification of a correlation matrix
- Various plotting utilities for trajectory visualization
- Conversion utilities between cartesian and barycentric coordinates

## Requirements

- Julia (1.0 or higher)
- Required packages:
  - PyPlot
  - LinearAlgebra
  - Distributions
  - Random

## Installation

1. Install Julia from [julialang.org](https://julialang.org/downloads/)
2. Install required packages:
```julia
using Pkg
Pkg.add(["PyPlot", "LinearAlgebra", "Distributions", "Random"])
```


The script will generate several visualizations:
1. Loss landscape on the probability simplex with and without gradient flow
2. Single neuron weight vector trajectories
3. Multiple neuron weight vector dynamics with orthogonalization

## Configuration

The script includes configurable parameters at the top:

- Probability simplex configuration (vertices, grid resolution)
- Learning rates for single and multiple weight vectors
- Correlation matrix for input neurons
- Number of simulation time steps
- Starting points for trajectories
- Input neuron intensities
- Number of simulation runs

## Key Functions

- `LossLandscape`: Visualizes the loss function across the probability simplex
- `indiv_trajectory`: Simulates individual STDP trajectories
- `plot_trajectories`: Plots multiple trajectories on the loss landscape
- `learning_dynamics`: Implements learning for multiple weight vectors with orthogonalization
- `plot_trajectories_multiple_weights`: Visualizes trajectories of multiple weight vectors
- Various helper functions for coordinate conversion and visualization

## Citation

If you use this code in your research, please cite:

```
@article{hebbianstdp2025,
  title={Spike-timing-dependent Hebbian learning as noisy gradient descent},
  author={[Niklas Dexheimer, Sascha Gaudlitz, Johannes Schmidt-Hieber]},
  journal={arXiv preprint arXiv:2505.10272},
  year={2025}
}
```

