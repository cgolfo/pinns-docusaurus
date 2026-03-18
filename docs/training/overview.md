---
id: overview
title: Training & Optimization Overview
sidebar_position: 1
---

# Training and Optimization Challenges in PINNs

While PINNs are powerful, training them reliably remains an active research frontier. This section covers major challenges and solutions.

## The Training Challenge

### Why PINNs Are Hard to Train

1. **Multi-objective optimization** (data + physics)
2. **Loss imbalance** (components differ by orders of magnitude)
3. **Spectral bias** (networks prefer low frequencies)
4. **Non-convex landscape** (many local minima)
5. **Ill-conditioned Hessian** (poor gradient flow)
6. **No convergence guarantees** (unlike FEM)

### Impact

- ❌ Requires careful hyperparameter tuning
- ❌ Sensitive to initialization
- ❌ Training can plateau or diverge
- ❌ Not reproducible across runs

---

## Major Training Challenges

### 1. Spectral Bias

Neural networks naturally fit low-frequency components first, struggling with high-frequency features.

**Symptom**: Predicted solution is smooth but misses fine details

**Solution**:
- Fourier feature embeddings
- Positional encoding
- Specialized architectures (Transformers, KANs)

[→ Full discussion: Spectral Bias](./spectral-bias.md)

### 2. Loss Imbalance

Data, PDE, and boundary losses have vastly different magnitudes, causing optimization conflicts.

**Symptom**: One loss dominates; others are ignored

**Solutions**:
- Inverse variance weighting
- NTK-guided balancing
- Causal training
- Multi-task learning

[→ Full discussion: Loss Balancing](./loss-balancing.md)

### 3. Gradient Pathologies

Poor gradient flow due to network architecture or PDE characteristics.

**Symptom**: Slow convergence, stuck optimization

**Solutions**:
- Better activation functions
- Residual connections
- Second-order optimization (L-BFGS)
- Gradient surgery

### 4. Long-Time Integration

Errors accumulate exponentially for time-dependent PDEs, especially chaotic systems.

**Symptom**: Good early predictions, divergence at later times

**Solutions**:
- Causal training
- Ensemble methods
- Statistical/attractor-aware objectives
- Hybrid time-stepping

[→ Full discussion: Long-Time Training](../research/gap-4-long-time.md)

### 5. NTK Regime Limitations

Neural tangent kernel theory, which motivates many training strategies, breaks down for nonlinear PDEs.

**Implication**: Loss balancing methods lack theoretical justification

**Research frontier**: Develop nonlinear NTK theory

[→ Full discussion: NTK Theory](./ntk-theory.md)

---

## Training Workflow

### Phase 1: Problem Setup

1. **Discretize domain**: Define geometry, collocation points
2. **Initialize network**: Choose architecture, activation, weights
3. **Define losses**: Data, PDE, IC, BC terms
4. **Set hyperparameters**: Loss weights, learning rates, batch size

### Phase 2: Adam Training (0–5,000 epochs)

```python
# Fast initial convergence
optimizer = Adam(lr=1e-3)
scheduler = ExponentialLR(gamma=0.99)

for epoch in range(5000):
    loss = compute_loss()
    optimizer.step(loss)
    scheduler.step()
    
    if epoch % 500 == 0:
        print(f"Loss: {loss:.2e}")
```

**Characteristics**:
- ✅ Rapid initial loss decrease
- ✅ Robust to learning rate
- ✅ ❌ Often plateaus

### Phase 3: L-BFGS Fine-Tuning (5,000–10,000 epochs)

```python
# Better final convergence
optimizer = LBFGS(lr=0.8)

for epoch in range(5000):
    def closure():
        loss = compute_loss()
        loss.backward()
        return loss
    
    optimizer.step(closure)
```

**Characteristics**:
- ✅ Continued convergence
- ✅ Better Hessian handling
- ❌ Slow per-iteration

### Phase 4: Validation & Refinement

1. **Evaluate on test data**: Compute relative errors
2. **Diagnostic checks**: Verify loss components, gradient flow
3. **Refine if needed**: Increase network capacity, adjust hyperparameters
4. **Deploy**: Use trained model for inference

---

## Hyperparameter Sensitivity

### Critical Hyperparameters

| Parameter | Range | Impact |
|-----------|-------|--------|
| **Hidden layers** | 4–8 | Deeper = higher capacity |
| **Neurons/layer** | 20–256 | Wider = more flexible |
| **Activation** | tanh, ReLU | tanh for high-order derivatives |
| **λ_p / λ_d** | Variable | Critical! Imbalance = failure |
| **Learning rate (Adam)** | 1e-4 — 1e-2 | Typical: 1e-3 |
| **Learning rate (L-BFGS)** | 0.1–2.0 | Typical: 0.8 |
| **Batch size** | 32–1024 | Larger = smoother gradients |
| **Collocation density** | 10² — 10⁴ | Problem-dependent |

### Where Tuning Helps Most

1. **Loss weights** (λ_p, λ_d): Most critical
2. **Collocation density**: Problem size
3. **Learning rates**: Training speed, stability
4. **Network depth**: Expressiveness
5. **Activation function**: Derivative smoothness

---

## Debugging Guide

### Loss Not Decreasing

**Diagnosis**:
```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"{name}: No gradient!")
    elif param.grad.abs().max() < 1e-8:
        print(f"{name}: Tiny gradients")
```

**Solutions**:
- Reduce learning rate
- Check loss function (NaN?)
- Increase batch size
- Verify collocation points are valid

### One Loss Component Dominates

**Diagnosis**:
```python
# Log individual losses
print(f"Data loss: {L_data:.2e}")
print(f"PDE loss: {L_pde:.2e}")
print(f"IC loss: {L_ic:.2e}")
```

**Solutions**:
- Implement loss balancing
- Adjust λ weights manually
- Use inverse variance weighting

### Divergence (NaN/Inf)

**Diagnosis**:
- Check collocation points for NaN/Inf
- Verify boundary/initial conditions
- Look for numerical instability

**Solutions**:
- Reduce learning rate (e.g., 1e-3 → 1e-4)
- Use gradient clipping
- Normalize input/output

### Overfitting (Train loss ≠ Test error)

**Symptoms**:
- Training loss small
- Test error large

**Solutions**:
- Add more collocation points
- Reduce network size
- Add regularization (L2, dropout)

---

## Validation Strategy

### Train/Test Split

```python
# Split collocation points
split = 0.8
N_train = int(split * N_colloc)

indices = np.random.permutation(N_colloc)
train_indices = indices[:N_train]
test_indices = indices[N_train:]

x_train, x_test = x_colloc[train_indices], x_colloc[test_indices]
```

### Error Metrics

```python
def relative_error(u_pred, u_exact):
    """L² relative error"""
    return np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

def max_error(u_pred, u_exact):
    """Maximum absolute error"""
    return np.max(np.abs(u_pred - u_exact))

# On test set
error_rel = relative_error(u_pred[test_indices], u_exact[test_indices])
error_max = max_error(u_pred[test_indices], u_exact[test_indices])

print(f"Relative L² error: {error_rel:.2e}")
print(f"Max absolute error: {error_max:.2e}")
```

### Reference Solutions

Compare against:
- **Analytical solutions** (if available)
- **Reference numerical solver** (FEM, FD)
- **Other learning methods** (pure NN, other PINNs variants)

---

## Training Time & Computational Cost

### Typical Scaling

| Problem | Parameters | Training Time | Inference |
|---------|-----------|---------------|-----------|
| **1D Burgers** | ~5K | 30 min (CPU) | <1 ms |
| **2D Navier-Stokes** | ~50K | 2–4 hrs (GPU) | 5–10 ms |
| **3D Heat** | ~100K | 8–16 hrs (GPU) | 20–50 ms |
| **High-D (100D)** | ~500K | 24+ hrs (GPU) | 100+ ms |

### Cost Breakdown

**Single forward + backward pass**:
- Forward: ~1x (evaluate network)
- Backward data loss: ~1x (standard backprop)
- Backward PDE loss: ~3–5x (compute derivatives)
- Total: ~5–6x typical NN cost

**Optimization cost**:
- Adam step: O(#params)
- L-BFGS step: O(#params²) (Hessian approximation)

### GPU vs. CPU

- **CPU**: Fine for small problems (<10K parameters)
- **GPU**: Essential for large problems (>100K parameters)
- **TPU**: Emerging option for very large scale

---

## Next Steps

Explore specific training challenges:
- [Spectral Bias](./spectral-bias.md)
- [Loss Balancing](./loss-balancing.md)
- [NTK Theory](./ntk-theory.md)
- [Causal Training](./causal-training.md)
- [Meta-Learning](./meta-learning.md)

---

## References

1. **Raissi et al. (2019)**: Physics-informed neural networks  
   [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

2. **Wang et al. (2023)**: An expert's guide to training physics-informed neural networks  
   [arXiv:2308.08468](https://arxiv.org/abs/2308.08468)

3. **Ranade et al. (2024)**: The PINNacle: Understanding PINNs and why they train  
   [NeurIPS 2024](https://nips.cc)
