---
id: auto-differentiation
title: Automatic Differentiation in PINNs
sidebar_position: 4
---

# Automatic Differentiation: The Heart of PINNs

Automatic differentiation (AD) is the computational technique that makes PINNs possible. Without it, computing PDE residuals would be intractable.

## What Is Automatic Differentiation?

Automatic differentiation computes derivatives of functions by:
1. **Building a computation graph** of all operations
2. **Applying the chain rule** systematically through the graph
3. **Tracking gradients** in either forward or reverse mode

**Key insight**: AD computes derivatives **exactly** (up to machine precision), not via finite differences.

## Why Not Finite Differences?

### Problems with Finite Differences

```python
# Finite difference approximation
du_dx_fd = (u(x + h) - u(x - h)) / (2*h)
```

❌ **Truncation error**: O(h²) for central differences  
❌ **Cancellation error**: h too small → numerical instability  
❌ **Scale sensitivity**: Requires problem-specific h tuning  
❌ **Second derivatives worse**: Errors compound  
❌ **High-dimensional intractable**: 100D PDE = 100 FD evaluations per point  

### Automatic Differentiation Advantages

✅ **Exact**: Machine precision (no approximation)  
✅ **Stable**: No numerical cancellation  
✅ **Scale-invariant**: Works for any magnitude of values  
✅ **High-order**: Arbitrary derivative order via composition  
✅ **Efficient**: Reverse-mode scales to 100+ dimensions  

## Forward vs. Reverse Mode

### Forward Mode (Tangent Linear)

Computes all derivatives **with respect to one input**:

```
Input: x
Output: du/dx, d²u/dx², d³u/dx³, ...
```

**Use when**: Few inputs, many outputs (e.g., single x → multiple derivatives)

**Cost**: O(1) forward passes

### Reverse Mode (Backpropagation)

Computes all derivatives **of one output with respect to all inputs**:

```
Input: all x₁, x₂, ..., xₙ
Output: du/dx₁, du/dx₂, ..., du/dxₙ
```

**Use when**: Many inputs, few outputs (typical for neural networks)

**Cost**: O(1) backward passes + O(1) memory for gradients

## Automatic Differentiation in PINNs

### 1st Derivative (du/dx)

```python
import tensorflow as tf

x = tf.Variable(x_colloc, trainable=False)

with tf.GradientTape() as tape:
    tape.watch(x)
    u = neural_network(x)  # Forward pass

du_dx = tape.gradient(u, x)  # 1st derivative
```

**Mechanism**: TensorFlow builds computation graph, applies chain rule

### 2nd Derivative (d²u/dx²)

Critical for PDEs like heat equation, Burgers equation:

```python
x = tf.Variable(x_colloc, trainable=False)

with tf.GradientTape() as tape2:
    tape2.watch(x)
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = neural_network(x)
    du_dx = tape.gradient(u, x)

d2u_dx2 = tape2.gradient(du_dx, x)  # 2nd derivative
```

**Key point**: Nested GradientTape computes derivatives of derivatives

### Mixed Derivatives (∂²u/∂x∂t)

For time-dependent PDEs:

```python
xt = tf.Variable(xt_colloc, trainable=False)

with tf.GradientTape() as tape_t:
    tape_t.watch(xt)
    with tf.GradientTape() as tape_x:
        tape_x.watch(xt)
        u = neural_network(xt)
    du_dx = tape_x.gradient(u, xt, i=0)  # ∂u/∂x

d2u_dxdt = tape_t.gradient(du_dx, xt, i=1)  # ∂²u/∂x∂t
```

## Implementation Frameworks

### TensorFlow/Keras

```python
import tensorflow as tf

@tf.function  # JIT compilation for speed
def compute_residual(x, t):
    xt = tf.stack([x, t], axis=1)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(xt)
        u = model(xt)
    
    u_xt = tape.gradient(u, xt)
    u_x = u_xt[:, 0]
    u_t = u_xt[:, 1]
    
    with tf.GradientTape() as tape2:
        tape2.watch(xt)
        u_x_temp = tape2.gradient(u, xt)[:, 0]
    u_xx = tape2.gradient(u_x_temp, xt)[:, 0]
    
    # PDE: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    return residual
```

### PyTorch

```python
import torch

def compute_residual(x, t, model):
    xt = torch.cat([x.unsqueeze(1), t.unsqueeze(1)], dim=1)
    xt.requires_grad_(True)
    
    u = model(xt)
    
    # 1st derivatives
    u_xt = torch.autograd.grad(
        u.sum(), xt, create_graph=True, retain_graph=True
    )[0]
    u_x = u_xt[:, 0]
    u_t = u_xt[:, 1]
    
    # 2nd derivatives
    u_xx = torch.autograd.grad(
        u_x.sum(), xt, create_graph=True, retain_graph=True
    )[0][:, 0]
    
    # PDE residual
    residual = u_t + u * u_x - nu * u_xx
    return residual
```

### JAX (Modern Approach)

```python
import jax
from jax import grad, jacobian

def residual_fn(params, x, t):
    u = neural_network(params, x, t)
    u_x = grad(u, argnums=0)
    u_t = grad(u, argnums=1)
    u_xx = grad(grad(u, argnums=0), argnums=0)
    
    return u_t + u * u_x - nu * u_xx

# Vectorize over batch
vmap_residual = jax.vmap(residual_fn, in_axes=(None, 0, 0))
residuals = vmap_residual(params, x_batch, t_batch)
```

## Computational Cost Analysis

### Cost Scaling

| Operation | Cost (# forward evaluations) |
|-----------|------------------------------|
| **1st derivative** | ~2× (one forward, one backward) |
| **2nd derivative** | ~4× (nested derivatives) |
| **Mixed derivatives** | ~4-8× (problem-dependent) |

### Memory Overhead

- **Forward pass**: Store all intermediate values for chain rule
- **Deep networks**: Memory can dominate training cost
- **Solution**: Gradient checkpointing (recompute intermediate values)

## Common Pitfalls

### 1. Forgetting `create_graph=True`

```python
# ❌ WRONG: Can't compute 2nd derivatives
u_x = torch.autograd.grad(u, x)[0]
u_xx = torch.autograd.grad(u_x, x)[0]  # Error!

# ✅ CORRECT: Enable graph building for 2nd order
u_x = torch.autograd.grad(u, x, create_graph=True)[0]
u_xx = torch.autograd.grad(u_x, x)[0]  # Works!
```

### 2. Forgetting `requires_grad_(True)`

```python
# ❌ WRONG: Collocation points not tracked
x = torch.tensor(x_colloc)  # requires_grad=False by default
u = model(x)
du_dx = torch.autograd.grad(u, x)  # Error!

# ✅ CORRECT: Enable gradient tracking
x = torch.tensor(x_colloc, requires_grad=True)
u = model(x)
du_dx = torch.autograd.grad(u, x)[0]  # Works!
```

### 3. Modifying Variables During Differentiation

```python
# ❌ WRONG: Graph broken by in-place operation
x.detach_()  # Breaks gradient flow
u = model(x)
du_dx = torch.autograd.grad(u, x)  # Error!

# ✅ CORRECT: Use out-of-place operations
x_detached = x.detach()  # Doesn't break original graph
```

## Performance Optimization

### Gradient Checkpointing

For deep networks, recompute intermediate values instead of storing:

```python
def checkpoint_forward(module, x):
    """Reduce memory by recomputing during backward pass"""
    return torch.utils.checkpoint.checkpoint(module, x)
```

### JIT Compilation

TensorFlow/JAX can JIT-compile AD operations:

```python
@tf.function  # JIT compile for speed
def batched_residuals(x_batch, t_batch):
    return vmap_compute_residual(x_batch, t_batch)
```

### Vectorization

Process multiple collocation points simultaneously:

```python
# Slow: One point at a time
for x_i, t_i in zip(x_colloc, t_colloc):
    residual_i = compute_residual(x_i, t_i)

# Fast: Vectorized batch
residuals = vmap_compute_residual(x_colloc, t_colloc)
```

## Debugging AD

### Numerical Gradient Checking

Verify AD implementation via finite differences:

```python
def finite_diff_grad(f, x, eps=1e-5):
    """Compute gradient via finite differences"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (f(x_plus) - f(x_minus)) / (2*eps)
    return grad

def check_grad(model, x):
    """Compare AD gradient to finite difference"""
    x_tensor = torch.tensor(x, requires_grad=True)
    u = model(x_tensor)
    grad_ad = torch.autograd.grad(u.sum(), x_tensor)[0].detach().numpy()
    
    grad_fd = finite_diff_grad(lambda xi: model(torch.tensor(xi)).detach().numpy(), x)
    
    error = np.max(np.abs(grad_ad - grad_fd))
    print(f"Max gradient error: {error:.2e}")
    assert error < 1e-4, "Gradient check failed!"
```

---

## Key Takeaways

✅ Automatic differentiation is **essential** for computing physics residuals  
✅ **Reverse-mode** AD (backprop) scales to high dimensions  
✅ **Second-order derivatives** require nested gradient tapes  
✅ **Performance** requires careful implementation (vectorization, checkpointing, JIT)  
✅ **Debugging** via numerical gradient checking validates implementation  

## References

- **Baydin et al. (2018)**: Automatic differentiation in machine learning  
  [arXiv:1502.05767](https://arxiv.org/abs/1502.05767)

- **TensorFlow Autodiff**: [tensorflow.org/guide/autodiff](https://www.tensorflow.org/guide/autodiff)

- **PyTorch Autograd**: [pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

- **JAX Transformations**: [jax.readthedocs.io/en/latest/notebooks/quickstart.html](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
