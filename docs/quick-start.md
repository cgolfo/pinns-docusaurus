---
id: quick-start
title: Quick Start
sidebar_position: 3
---

# Quick Start: Your First PINN

In this guide, we'll build and train a simple PINN to solve **Burgers equation**, a classic benchmark PDE.

## The Problem

We want to solve the 1D Burgers equation:

$$u_t + u \cdot u_x = \nu \cdot u_{xx}$$

With boundary and initial conditions:
- $u(x, 0) = -\sin(\pi x)$ (initial)
- $u(-1, t) = u(1, t) = 0$ (boundary)
- $\nu = 0.01/\pi$ (viscosity)

## Installation

### Using DeepXDE (Recommended)

```bash
pip install deepxde tensorflow  # or torch
```

### Setup
```python
import deepxde as dde
import numpy as np
from deepxde.backend import tf
```

## Step 1: Define the PDE

```python
# Define the governing PDE
def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)  # ∂u/∂x
    u_t = dde.grad.jacobian(u, x, i=0, j=1)  # ∂u/∂t
    u_xx = dde.grad.hessian(u, x, i=0, j=0)  # ∂²u/∂x²
    
    nu = 0.01 / np.pi
    return u_t + u * u_x - nu * u_xx
```

## Step 2: Specify Boundary/Initial Conditions

```python
# Spatial domain
spatial_domain = dde.geometry.Interval(-1, 1)

# Temporal domain
temporal_domain = dde.geometry.TimeDomain(0, 1)

# Combined spatio-temporal domain
geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

# Initial condition
def initial_condition(x):
    return -np.sin(np.pi * x[:, 0:1])

ic = dde.IC(
    geomtime,
    initial_condition,
    lambda x, u: u,
    on_initial=True  # Enforced at t=0
)

# Boundary conditions (Dirichlet)
bc = dde.BC(
    geomtime,
    lambda x, u: u,
    lambda x: np.logical_or(x[:, 0:1] == -1, x[:, 0:1] == 1),
)
```

## Step 3: Create the PINN Model

```python
# Neural network architecture
net = dde.nn.FeedForward(
    input_size=2,           # (x, t)
    output_size=1,          # u(x, t)
    hidden_layers=[20, 20, 20, 20],
    activation="tanh"
)

# Define the PDE problem
model = dde.Model(
    data,  # Geometry, BCs, ICs
    net,
    model_data
)

model.compile("adam", lr=1e-3)
```

## Step 4: Train

```python
# Train on collocation points
history = model.train(epochs=10000, batch_size=128)

# Plot training history
dde.plot.plot_loss_history(history)
```

## Step 5: Evaluate

```python
# Create test points
x_test = np.linspace(-1, 1, 100)
t_test = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_test, t_test)
X_test = np.column_stack((X.ravel(), T.ravel()))

# Predict
u_pred = model.predict(X_test)

# Compare with reference solution (optional)
import scipy.io
data = scipy.io.loadmat('burgers_shock.mat')
u_exact = data['usol'].T
error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
print(f"Relative error: {error:.2e}")
```

---

## Complete Code

```python
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# ============ Step 1: Define PDE ============
def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    
    nu = 0.01 / np.pi
    return u_t + u * u_x - nu * u_xx

# ============ Step 2: Define Geometry ============
spatial_domain = dde.geometry.Interval(-1, 1)
temporal_domain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

# ============ Step 3: Initial/Boundary Conditions ============
def initial_condition(x):
    return -np.sin(np.pi * x[:, 0:1])

ic = dde.IC(geomtime, initial_condition, lambda x, u: u, on_initial=True)
bc = dde.BC(
    geomtime,
    lambda x, u: u,
    lambda x: np.logical_or(x[:, 0:1] == -1, x[:, 0:1] == 1)
)

# ============ Step 4: Create Problem ============
data = dde.data.PDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2500,
    num_boundary=100,
    num_initial=160
)

# ============ Step 5: Build Network ============
net = dde.nn.FeedForward(
    input_size=2,
    output_size=1,
    hidden_layers=[20, 20, 20, 20],
    activation="tanh"
)

# ============ Step 6: Compile Model ============
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)

# ============ Step 7: Train ============
history = model.train(epochs=10000, batch_size=128)

# ============ Step 8: Evaluate ============
x_test = np.linspace(-1, 1, 100)
t_test = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_test, t_test)
X_test = np.column_stack((X.ravel(), T.ravel()))
u_pred = model.predict(X_test)

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Predicted Solution")
plt.contourf(X, T, u_pred.reshape(100, 100))
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")

plt.tight_layout()
plt.show()
```

---

## Key Takeaways

| Concept | Meaning |
|---------|---------|
| **PDE function** | Encodes the governing equation |
| **Geometry** | Defines spatial and temporal domains |
| **Boundary/Initial Conditions** | Enforced via constraint functions |
| **Neural network** | Approximates the solution u(x, t) |
| **Loss function** | Combines data + physics residuals |
| **Training** | Minimizes loss via gradient descent |

---

## Next Steps

- Explore **[Variants](../methods/variants.md)** for advanced techniques (VPINNs, cPINNs, etc.)
- Learn about **[Loss Balancing](../training/loss-balancing.md)** for stable training
- See **[Applications](../applications/fluids.md)** in different domains
- Check out **[Tools](../tools/deepxde.md)** (DeepXDE, Modulus, SciANN)

## Common Pitfalls

⚠️ **Spectral bias**: NNs struggle with high-frequency features → use Fourier features  
⚠️ **Loss imbalance**: PDE loss dominates → use weighted/adaptive loss balancing  
⚠️ **Poor initialization**: Random weights may fail → use causal training for time-dependent PDEs  
⚠️ **Insufficient collocation**: Too few interior points → increase `num_domain`  

---

## References

- DeepXDE Docs: [deepxde.readthedocs.io](https://deepxde.readthedocs.io)
- Raissi et al. (2019): [Physics-informed neural networks (PINNs)](https://doi.org/10.1016/j.jcp.2018.10.045)
- Tutorial Notebooks: [idrl-lab/PINNs](https://github.com/idrl-lab/PINNs)
