---
id: canonical-framework
title: The Canonical PINN Framework
sidebar_position: 2
---

# The Canonical PINN Framework

This document details the foundational PINN architecture introduced by **Raissi, Perdikaris, and Karniadakis (2019)** — the most cited and influential PINN work.

## Overview

The canonical PINN framework consists of:
1. **Neural Network Architecture**: Multi-layer feed-forward network
2. **Forward Problem**: Predicting solution u(x,t) from PDE
3. **Inverse Problem**: Discovering PDE coefficients from sparse data
4. **Loss Function**: Combining data and physics residuals

---

## Architecture

### Network Design

```
Input Layer (d_in: spatial + temporal dimensions)
    ↓
Hidden Layers (typically 4-8 layers, 20-50 neurons each)
    ├── Activation: tanh or ReLU
    └── No batch norm (typically)
    ↓
Output Layer (d_out: usually 1 for scalar PDEs)
```

**Typical specifications**:
- **Depth**: 4–8 layers
- **Width**: 20–50 neurons per layer
- **Activation**: tanh (preferred), ReLU
- **Initialization**: Xavier uniform
- **Optimizer**: Adam (early), L-BFGS (fine-tuning)

### Why tanh?

$$\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Advantages:
- Smooth, infinitely differentiable
- Zero-centered (helps gradient flow)
- Bounds output ∈ [-1, 1]
- Better for high-order derivatives (needed for PDEs)

---

## Forward Problem Formulation

### Problem Statement

Given a PDE:
$$u_t + \nabla \cdot \mathbf{f}(u) = 0 \quad \text{(conservation law)}$$

And boundary/initial conditions, find $u(x,t)$.

### PINN Solution

Train neural network $u_\theta(x,t)$ to minimize:

$$\mathcal{L}_{\text{forward}} = \mathcal{L}_{\text{data}} + \mathcal{L}_{\text{physics}}$$

#### Data Loss

$$\mathcal{L}_{\text{data}} = \frac{1}{N_d} \sum_{i=1}^{N_d} |u_\theta(x_i, t_i) - u_i^{\text{obs}}|^2$$

- Enforces match to **measured or reference data**
- Can incorporate measurement noise
- Typically sparse (tens to thousands of points)

#### Physics Loss

$$\mathcal{L}_{\text{physics}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \left| \mathcal{F}\left(x_j, u_\theta, \frac{\partial u_\theta}{\partial x}, \frac{\partial^2 u_\theta}{\partial x^2}, \ldots\right) \right|^2$$

Where $\mathcal{F}$ is the **residual operator**:
$$\mathcal{F}(u) = u_t + u \cdot u_x - \nu u_{xx} \quad \text{(Burgers)}$$

**Key insight**: Derivatives computed via **automatic differentiation**!

---

## Inverse Problem Formulation

### Problem Statement

Given:
- Sparse observations of $u$ at $(x_i, t_i)$
- Known PDE structure: $u_t + u \cdot u_x = \nu u_{xx}$
- Unknown parameter: $\nu$

Find: The value of $\nu$.

### PINN Solution

Augment the network to predict both **solution** and **parameters**:

```python
# Pseudo-code
def forward(x, t):
    u = neural_network(x, t)          # Solution
    nu = learnable_parameter           # PDE coefficient
    return u, nu

def loss(x_data, t_data, u_data, x_colloc, t_colloc):
    u_pred, nu_pred = forward(x_data, t_data)
    
    # Data loss: match observations
    L_data = ||u_pred - u_data||^2
    
    # Physics loss: residual at collocation points
    residual = du/dt + u*du/dx - nu_pred*d²u/dx²
    L_physics = ||residual||^2
    
    # Regularization on parameter (optional)
    L_reg = (nu_pred - nu_true)^2
    
    return L_data + L_physics + lambda_reg * L_reg
```

---

## Key Benchmark: Burgers Equation

### PDE Definition

$$u_t + u \cdot u_x = \nu u_{xx}, \quad x \in [-1,1], t \in [0,1]$$

With:
- Initial condition: $u(x,0) = -\sin(\pi x)$
- Boundary: $u(-1,t) = u(1,t) = 0$
- Viscosity: $\nu = 0.01/\pi$

### Canonical Results

| Metric | PINN | Classical FDM |
|--------|------|---------------|
| **Training data** | 100 points | Dense grid (10,000+) |
| **Relative error** | 10^-3 | 10^-4 |
| **Inference time** | <1 ms | 1-10 s |
| **Coefficient discovery** | Yes (ν) | No |

### Success Criteria

✅ **Forward problem**: Relative L² error < 10^-3  
✅ **Inverse problem**: Coefficient discovered within 1% error  

---

## Automatic Differentiation

### How It Works

1. **Build computation graph**: x → network → u
2. **Forward pass**: Compute u at collocation points
3. **Backward pass**: Chain rule to compute du/dx, d²u/dx²
4. **Physics residual**: Evaluate F(u) using derivatives
5. **Backprop**: Gradient of loss w.r.t. network weights

### Example: Computing du/dx

```python
import tensorflow as tf

@tf.function
def compute_residual(x_colloc, u_pred):
    with tf.GradientTape() as tape:
        tape.watch(x_colloc)
        with tf.GradientTape() as tape2:
            tape2.watch(x_colloc)
            u = neural_network(x_colloc)  # Forward pass
        du_dx = tape2.gradient(u, x_colloc)  # 1st deriv
    d2u_dx2 = tape.gradient(du_dx, x_colloc)  # 2nd deriv
    
    # PDE residual: u_t + u*u_x - nu*u_xx = 0
    residual = du_dt + u_pred * du_dx - nu * d2u_dx2
    return residual
```

### Key Insight

**Second-order autodiff** computes Hessians efficiently. This is non-trivial but essential for PDEs.

---

## Training Details

### Optimizer Strategy

**Phase 1: Adam** (0–5000 epochs)
```python
optimizer = Adam(learning_rate=1e-3)
for epoch in range(5000):
    loss = compute_loss(x_data, u_data, x_colloc)
    optimizer.minimize(loss)
```

**Phase 2: L-BFGS** (5000+ epochs)
```python
optimizer = LBFGS(learning_rate=0.8)
for epoch in range(5000):
    loss = compute_loss(...)
    optimizer.minimize(loss)
```

**Rationale**:
- Adam: Fast convergence initially, robust to learning rate
- L-BFGS: Better final convergence, handles ill-conditioning

### Hyperparameters

| Parameter | Typical Value | Impact |
|-----------|---------------|--------|
| **Hidden layers** | 4 | Deeper → better approximation, more overfitting |
| **Neurons/layer** | 20 | Wider → higher capacity, slower training |
| **Batch size** | 32–256 | Larger → smoother gradients, slower per-step |
| **Learning rate** | 1e-3 (Adam) | Too high → divergence; too low → slow |
| **λ_p / λ_d** | 1 | Critical! Imbalance → poor training |

---

## Limitations of the Canonical Framework

❌ **Spectral bias**: Struggles with high-frequency components  
❌ **Loss imbalance**: Physics loss often dominates, drowning data signal  
❌ **Poor long-time integration**: Errors accumulate over time  
❌ **Hyperparameter sensitivity**: Requires careful tuning  
❌ **No theoretical guarantees**: Convergence analysis lacking  

These limitations motivate the **variants and extensions** covered in later sections.

---

## References

1. **Raissi, Perdikaris, Karniadakis (2019)**  
   *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems*  
   [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

2. **Raissi, Perdikaris, Karniadakis (2017 Part II)**  
   *Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear PDEs*  
   [arXiv:1711.10566](https://arxiv.org/abs/1711.10566)

3. **Lagaris, Likas, Fotiadis (1998)**  
   *Artificial neural networks for solving ordinary and partial differential equations*  
   [DOI:10.1109/72.712178](https://doi.org/10.1109/72.712178)
