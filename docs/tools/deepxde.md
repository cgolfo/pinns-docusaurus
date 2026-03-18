---
id: deepxde
title: DeepXDE Library
sidebar_position: 1
---

# DeepXDE: A Deep Learning Library for Solving Differential Equations

**DeepXDE** is the most widely used open-source library for implementing PINNs, created by Lululxvi (Lu et al., 2021).

## Key Features

✅ **High-level API**: Simple syntax for defining PDEs  
✅ **Multi-backend**: TensorFlow, PyTorch, JAX  
✅ **Built-in geometries**: Interval, Rectangle, Disk, etc.  
✅ **Boundary conditions**: Dirichlet, Neumann, Robin, periodic  
✅ **Advanced features**:
- Residual-adaptive refinement (RAR)
- Uncertainty quantification
- Operator learning (DeepONet)

## Installation

```bash
pip install deepxde
# Choose backend
pip install tensorflow  # or torch, jax
```

## Quick Example: Burgers Equation

```python
import deepxde as dde
import numpy as np

def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    nu = 0.01 / np.pi
    return u_t + u * u_x - nu * u_xx

# Domain and conditions
geomtime = dde.geometry.GeometryXTime(
    dde.geometry.Interval(-1, 1),
    dde.geometry.TimeDomain(0, 1)
)

def ic(x):
    return -np.sin(np.pi * x[:, 0:1])

ic_bc = dde.IC(geomtime, ic, lambda x, u: u)
bc = dde.BC(geomtime, lambda x, u: u, lambda x: np.abs(x[:, 0:1]) > 1 - 1e-6)

# Data and model
data = dde.data.PDE(
    geomtime, pde, [ic_bc, bc],
    num_domain=2500, num_boundary=100, num_initial=160
)

net = dde.nn.FeedForward([2, 20, 20, 20, 20, 1], "tanh")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=10000)
```

## More Resources

- **Documentation**: [deepxde.readthedocs.io](https://deepxde.readthedocs.io)
- **GitHub**: [github.com/lululxvi/deepxde](https://github.com/lululxvi/deepxde)
- **Tutorials**: Comprehensive examples in repository

---

*See also: [Modulus](./modulus.md), [SciANN](./sciann.md)*
