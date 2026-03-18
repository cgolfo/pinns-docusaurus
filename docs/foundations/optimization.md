---
id: optimization
title: Optimization & Training Strategies
sidebar_position: 6
---

# Optimization and Training Strategies for PINNs

Effective optimization is crucial for PINN success. This section covers optimizers, learning rate schedules, and training strategies.

## Optimizer Selection

### Adam Optimizer (Early Training)

**Strengths**:
- ✅ Fast convergence initially
- ✅ Robust to learning rate choice
- ✅ Good for non-convex landscapes
- ✅ Industry standard

**Weakness**:
- ❌ May plateau before reaching optimal solution

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    loss = compute_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### L-BFGS Optimizer (Fine-Tuning)

**Strengths**:
- ✅ Better final convergence
- ✅ Handles ill-conditioned problems
- ✅ Quasi-Newton second-order approximation

**Weakness**:
- ❌ Requires full loss computation per step
- ❌ Slow initial convergence

```python
def closure():
    optimizer.zero_grad()
    loss = compute_loss()
    loss.backward()
    return loss

optimizer = optim.LBFGS(model.parameters(), lr=0.8)

for epoch in range(5000):
    loss = optimizer.step(closure)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.2e}")
```

### Two-Phase Training Strategy

**Standard approach** (Raissi et al., 2019):

```
Phase 1: Adam for 5,000-10,000 epochs
    └─ Fast initial convergence
    └─ Learning rate: 1e-3 → 1e-4 (decay)

Phase 2: L-BFGS for 5,000-10,000 epochs
    └─ Fine-tune to optimality
    └─ Learning rate: 0.8
```

**Implementation**:

```python
# Phase 1: Adam
optimizer1 = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(5000):
    loss = compute_loss()
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

# Phase 2: L-BFGS
optimizer2 = optim.LBFGS(model.parameters(), lr=0.8)
for epoch in range(5000):
    def closure():
        optimizer2.zero_grad()
        loss = compute_loss()
        loss.backward()
        return loss
    optimizer2.step(closure)
```

---

## Learning Rate Scheduling

### Exponential Decay

Gradually reduce learning rate:

$$\text{lr}(t) = \text{lr}_0 \cdot \gamma^{t / \text{decay_steps}}$$

```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

for epoch in range(epochs):
    loss = compute_loss()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

### Step Decay

Reduce at fixed milestones:

```python
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, 
    milestones=[1000, 3000, 5000],
    gamma=0.1
)
```

### Cosine Annealing

Smooth decay following cosine schedule:

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6
)
```

### Warm Restart (SGDR)

Restart learning rate periodically:

```python
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100,  # Initial period
    T_mult=2  # Period multiplier
)
```

---

## Loss Tracking and Diagnostics

### Training History

Track loss components to diagnose problems:

```python
history = {
    'total_loss': [],
    'data_loss': [],
    'pde_loss': [],
    'ic_loss': [],
    'bc_loss': []
}

for epoch in range(epochs):
    L_total = compute_total_loss()
    L_data = compute_data_loss()
    L_pde = compute_pde_loss()
    L_ic = compute_ic_loss()
    L_bc = compute_bc_loss()
    
    history['total_loss'].append(L_total.item())
    history['data_loss'].append(L_data.item())
    history['pde_loss'].append(L_pde.item())
    history['ic_loss'].append(L_ic.item())
    history['bc_loss'].append(L_bc.item())
    
    # ... optimize ...

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.semilogy(history['total_loss'], label='Total')
plt.semilogy(history['data_loss'], label='Data')
plt.semilogy(history['pde_loss'], label='PDE')
plt.semilogy(history['ic_loss'], label='IC')
plt.semilogy(history['bc_loss'], label='BC')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### Debugging Training Failures

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| **Loss NaN** | Divergence | Reduce learning rate |
| **Loss stuck** | Spectral bias | Add Fourier features |
| **One loss dominates** | Imbalance | Use adaptive weights |
| **Slow convergence** | Poor initialization | Use better activation |
| **Overfitting early** | Insufficient data | Add more collocation points |

---

## Causal Training (Recent Breakthrough)

**Reference**: Wang et al. (2024)

For time-dependent PDEs, enforce **temporal causality** by gradually increasing the training time window:

### Standard Training (Problematic)

```python
# Sample uniformly over entire [0, T]
t_colloc = torch.rand(N_colloc) * T_final

# Train on all times simultaneously
residual = compute_residual(t_colloc)
loss = residual.mean()
```

**Problem**: Network must satisfy PDE at all times, including future times (acausal!)

### Causal Training (Better)

```python
for epoch in range(epochs):
    # Gradually increase temporal domain
    t_max = (epoch / epochs) * T_final  # [0, t_max]
    
    # Sample collocation points in causal window
    t_colloc = torch.rand(N_colloc) * t_max
    
    # Only optimize for times <= current epoch
    residual = compute_residual(t_colloc)
    loss = residual.mean()
    
    optimizer.step(loss)
```

**Advantages**:
- ✅ Respects temporal causality
- ✅ Better long-time accuracy
- ✅ Faster early convergence
- ✅ No future information leak

### Implementation with Masking

```python
def causal_pde_loss(t_colloc, t_current):
    """Only enforce PDE for t <= t_current"""
    residuals = compute_residual(t_colloc)
    
    # Mask: only t <= t_current
    mask = (t_colloc <= t_current).float()
    
    # Masked loss
    masked_residuals = mask * residuals
    loss = (masked_residuals**2).mean()
    
    return loss

for epoch in range(epochs):
    t_max = (epoch / epochs) * T_final
    
    loss = causal_pde_loss(t_colloc, t_max)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Batch Training and Stochasticity

### Mini-Batch Training

Sample random collocation points each iteration (faster, more stable):

```python
def get_batch(batch_size):
    """Sample random collocation batch"""
    indices = np.random.choice(N_colloc, batch_size, replace=False)
    return x_colloc[indices], t_colloc[indices]

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        x_batch, t_batch = get_batch(batch_size=128)
        
        loss = compute_loss(x_batch, t_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Importance Sampling

Weight collocation points by importance:

```python
def importance_weights(residuals, temperature=1.0):
    """Higher weight to points with larger residuals"""
    # Softmax weighting based on residual magnitude
    weights = np.exp(residuals / temperature)
    weights /= weights.sum()
    return weights

# Resample according to importance
indices = np.random.choice(
    N_colloc,
    size=batch_size,
    p=importance_weights(residuals)
)
```

---

## Adaptive Collocation Point Refinement

### Residual-Adaptive Refinement (RAR)

**DeepXDE feature**: Add collocation points where residuals are large:

```python
def adaptive_refinement(model, x_colloc, threshold=0.1):
    """Add points where PDE residual is large"""
    residuals = compute_residual(x_colloc)
    
    # Points with high residual
    high_residual_mask = residuals > threshold * residuals.max()
    
    # Generate new points near high-residual regions
    x_high = x_colloc[high_residual_mask]
    
    # Perturbation to get new points
    epsilon = 0.01
    x_new = x_high + epsilon * np.random.randn(*x_high.shape)
    
    # Add to collocation set
    x_colloc = np.vstack([x_colloc, x_new])
    
    return x_colloc
```

**Applied periodically during training**:

```python
for epoch in range(epochs):
    loss = compute_loss()
    optimizer.step(loss)
    
    if epoch % 100 == 0 and epoch > 0:
        # Refine collocation points
        x_colloc = adaptive_refinement(model, x_colloc)
```

---

## Training Stability Tips

### 1. Initialize Properly

```python
# Bounded initialization helps
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        # Xavier uniform
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.zero_()

model.apply(init_weights)
```

### 2. Batch Normalization (Careful!)

Generally **not recommended** for PINNs (breaks residual computation), but can help:

```python
# Use layer norm instead
model = nn.Sequential(
    nn.Linear(d_in, 64),
    nn.LayerNorm(64),
    nn.Tanh(),
    # ...
)
```

### 3. Gradient Clipping

Prevent gradient explosion:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. Mixed Precision

Train with float32, but use float16 for memory efficiency:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(epochs):
    with autocast():
        loss = compute_loss()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Checkpointing and Early Stopping

```python
import torch

class CheckpointCallback:
    def __init__(self, save_path, patience=100):
        self.save_path = save_path
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, loss, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False

checkpoint = CheckpointCallback('./best_model.pt', patience=100)

for epoch in range(epochs):
    loss = compute_loss()
    optimizer.step(loss)
    
    if checkpoint(loss, model):
        print("Early stopping triggered")
        break

# Load best model
model.load_state_dict(torch.load('./best_model.pt'))
```

---

## Key Takeaways

✅ **Two-phase training** (Adam then L-BFGS) is standard  
✅ **Learning rate scheduling** improves convergence  
✅ **Causal training** respects temporal causality  
✅ **Adaptive refinement** focuses training where needed  
✅ **Loss tracking** reveals training problems  

## References

1. **Raissi et al. (2019)**: Physics-informed neural networks  
   [DOI:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)

2. **Wang et al. (2024)**: Understanding and mitigating gradient flow pathologies  
   [arXiv:2001.04536](https://arxiv.org/abs/2001.04536)

3. **Kingma & Ba (2014)**: Adam: A method for stochastic optimization  
   [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
