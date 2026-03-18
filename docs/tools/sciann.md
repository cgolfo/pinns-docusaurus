---
id: sciann
title: SciANN
sidebar_position: 3
---

# SciANN: Scientific AI

**SciANN** is a Keras/TensorFlow wrapper for scientific machine learning with high-level abstractions.

## Features

- **Field-based interface**: Define fields (scalars, vectors, tensors)
- **Operator overloading**: Natural PDE syntax
- **Automatic constraints**: IC/BC handling built-in

## Example

```python
import sciann as snn
import numpy as np

# Define fields
x_var = snn.Variable('x')
t_var = snn.Variable('t')
u = snn.Functional('u', [x_var, t_var], activation='tanh')

# Define PDEs via operator overloading
u_t = snn.diff(u, t_var, order=1)
u_x = snn.diff(u, x_var, order=1)
u_xx = snn.diff(u, x_var, order=2)

pde = u_t + u * u_x - 0.01/np.pi * u_xx

# Train
model = snn.SciANN([u], [pde], ...)
model.train(epochs=10000)
```

## Resources

- **Documentation**: [sciann.readthedocs.io](https://sciann.readthedocs.io)
- **GitHub**: [github.com/ehsanhaghighat/sciann](https://github.com/ehsanhaghighat/sciann)

---

*Lightweight alternative to DeepXDE for TensorFlow users*
