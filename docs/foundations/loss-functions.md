---
id: loss-functions
title: Loss Functions & Physics Regularization
sidebar_position: 5
---

# Loss Functions and Physics Regularization

The loss function is where physics meets learning. This section covers how to design, balance, and optimize loss functions for PINNs.

## The Total Loss

### Basic Formulation

$$\mathcal{L}_{\text{total}} = \lambda_d \mathcal{L}_d + \lambda_p \mathcal{L}_p$$

Where:
- **$\mathcal{L}_d$**: Data loss (supervised)
- **$\mathcal{L}_p$**: Physics loss (unsupervised)
- **$\lambda_d, \lambda_p$**: Hyperparameters controlling contribution ratio

### Multi-Component Problems

For multiple constraints (ICs, BCs, PDE):

$$\mathcal{L} = \lambda_{\text{IC}} \mathcal{L}_{\text{IC}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_d \mathcal{L}_d$$

---

## Loss Components

### 1. Data Loss (Supervised Learning)

Match predictions to **observed or reference data**:

$$\mathcal{L}_d = \frac{1}{N_d} \sum_{i=1}^{N_d} w_i \| u_\theta(x_i) - u_i^{\text{obs}} \|^2$$

**With weights** $w_i$ to handle data importance/uncertainty:

```python
# Example: Weighted data loss with uncertainty
uncertainty = np.sqrt(noise_level)  # Known noise std
weights = 1 / uncertainty**2  # Inverse variance weighting

data_loss = weights * (u_pred - u_obs)**2
L_data = data_loss.mean()
```

**Variants**:
- **MSE**: $\|u - u^{\text{obs}}\|^2$ (default)
- **MAE**: $\|u - u^{\text{obs}}\|$ (robust to outliers)
- **Huber**: Hybrid MSE/MAE (smooth, robust)
- **Weighted**: Account for measurement uncertainty

### 2. Initial Condition (IC) Loss

Enforce initial conditions at $t=0$:

$$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}} \sum_{i=1}^{N_{\text{IC}}} \| u_\theta(x_i, 0) - u_0(x_i) \|^2$$

**Implementation**:

```python
def ic_loss(x_ic, u_init):
    """Enforce initial condition u(x, 0) = u_init(x)"""
    u_pred = model(torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1))
    return ((u_pred - u_init)**2).mean()
```

### 3. Boundary Condition (BC) Loss

Enforce boundary conditions (Dirichlet, Neumann, Robin):

#### Dirichlet BC (value specified)

$$\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}} \sum_{i=1}^{N_{\text{BC}}} \| u_\theta(x_i^{\text{bc}}) - g(x_i^{\text{bc}}) \|^2$$

```python
def dirichlet_bc_loss(x_bc, u_bc):
    """u(x_bc) = u_bc"""
    u_pred = model(x_bc)
    return ((u_pred - u_bc)**2).mean()
```

#### Neumann BC (derivative specified)

$$\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}} \sum_{i=1}^{N_{\text{BC}}} \left\| \frac{\partial u_\theta}{\partial \mathbf{n}}(x_i^{\text{bc}}) - q(x_i^{\text{bc}}) \right\|^2$$

```python
def neumann_bc_loss(x_bc, n, q):
    """du/dn = q (directional derivative)"""
    x_bc.requires_grad_(True)
    u_pred = model(x_bc)
    u_x = torch.autograd.grad(u_pred.sum(), x_bc, create_graph=True)[0]
    du_dn = (u_x * n).sum(dim=1, keepdim=True)  # Directional deriv
    return ((du_dn - q)**2).mean()
```

#### Robin BC (mixed)

$$\frac{\partial u}{\partial \mathbf{n}} + \alpha u = \beta$$

### 4. Physics/PDE Loss (Residual)

Enforce the governing equation **unsupervised**:

$$\mathcal{L}_{\text{PDE}} = \frac{1}{N_c} \sum_{j=1}^{N_c} \| \mathcal{F}(x_j, u_\theta, \nabla u_\theta, \ldots) \|^2$$

**Example: Burgers equation**

$$\mathcal{F}(u) = \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2}$$

```python
def pde_loss(x_colloc, t_colloc, nu=0.01/np.pi):
    """Burgers equation residual"""
    xt = torch.cat([x_colloc, t_colloc], dim=1)
    xt.requires_grad_(True)
    
    u = model(xt)
    
    # Compute derivatives
    u_xt = torch.autograd.grad(
        u.sum(), xt, create_graph=True, retain_graph=True
    )[0]
    u_x = u_xt[:, 0:1]
    u_t = u_xt[:, 1:2]
    
    u_xx = torch.autograd.grad(
        u_x.sum(), xt, create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    # PDE: u_t + u*u_x - nu*u_xx = 0
    residual = u_t + u * u_x - nu * u_xx
    
    return (residual**2).mean()
```

---

## Loss Balancing: The Critical Problem

### The Challenge

Different loss components have vastly different magnitudes:

| Component | Typical Range | Issue |
|-----------|---------------|-------|
| **Data loss** | 10^-2 — 1 | Well-scaled |
| **PDE loss** | 10^-6 — 10^2 | Highly variable |
| **IC/BC loss** | 10^-3 — 1 | Moderate |

**Problem**: PDE loss often dominates, drowning data signal!

### Manual Weighting (Problematic)

```python
# ❌ Fixed weights: Sensitive to problem scale
loss = 1.0 * L_data + 1.0 * L_pde + 1.0 * L_ic
```

**Result**: Trial-and-error hyperparameter tuning

### Solution 1: Inverse Variance Weighting

Weight each component by inverse of its magnitude:

$$\lambda_i = \frac{1}{\mathcal{L}_i}$$

```python
def adaptive_loss(L_data, L_pde, L_ic, L_bc):
    """Normalize by magnitude"""
    # Compute losses
    L_d = data_loss()
    L_p = pde_loss()
    L_i = ic_loss()
    L_b = bc_loss()
    
    # Adaptive weights (inverse variance)
    w_d = 1.0 / (L_d.detach() + 1e-8)
    w_p = 1.0 / (L_p.detach() + 1e-8)
    w_i = 1.0 / (L_i.detach() + 1e-8)
    w_b = 1.0 / (L_b.detach() + 1e-8)
    
    # Normalized loss
    total_loss = (w_d * L_d + w_p * L_p + w_i * L_i + w_b * L_b) / (w_d + w_p + w_i + w_b)
    return total_loss
```

### Solution 2: NTK-Guided Weighting

**Wang et al. (2024)** proposes weighting based on neural tangent kernels:

$$\lambda_i = \frac{1}{\text{NTK}_i}$$

where $\text{NTK}_i$ measures the network's sensitivity to component $i$.

```python
def ntk_weighted_loss(model, x_data, u_data, x_colloc, t_colloc):
    """NTK-based adaptive weighting"""
    
    # Compute NTK (simplified)
    J_data = jacobian(model, x_data)  # Sensitivity to data region
    J_pde = jacobian(model, x_colloc)  # Sensitivity to PDE region
    
    # NTK magnitude
    ntk_data = (J_data @ J_data.T).diagonal().mean()
    ntk_pde = (J_pde @ J_pde.T).diagonal().mean()
    
    # Weights: inverse NTK
    w_data = 1.0 / (ntk_data + 1e-8)
    w_pde = 1.0 / (ntk_pde + 1e-8)
    
    # Normalized total loss
    L_d = data_loss(x_data, u_data)
    L_p = pde_loss(x_colloc, t_colloc)
    
    total_loss = (w_data * L_d + w_pde * L_p) / (w_data + w_pde)
    return total_loss
```

### Solution 3: Causal Training

For time-dependent PDEs, gradually increase temporal domain:

```python
def causal_loss(t_colloc, u_pred, t_final_epoch):
    """Enforce causality: only use t <= current training time"""
    t_max = (epoch / total_epochs) * t_final  # Grow temporal domain
    mask = t_colloc <= t_max
    
    residual = compute_residual(u_pred)
    return (mask * residual**2).mean()
```

---

## Advanced Loss Formulations

### Conservative Formulations

For conservation laws, use **flux form**:

$$\frac{\partial u}{\partial t} + \nabla \cdot \mathbf{f}(u) = 0$$

```python
def conservative_loss(u, flux_fn):
    """Enforce conservation in flux form"""
    u_t = du_dt
    div_f = divergence(flux_fn(u))
    return (u_t + div_f)**2
```

### Entropy-Stable Loss

For hyperbolic systems, enforce entropy condition:

$$\frac{\partial \eta(u)}{\partial t} + \nabla \cdot \psi(u) \leq 0$$

```python
def entropy_loss(u, entropy_fn, flux_fn):
    """Enforce entropy stability"""
    eta_t = d_entropy_dt
    psi_t = divergence(flux_fn(u))
    
    # Penalize entropy growth
    entropy_violation = max(0, eta_t + psi_t)
    return entropy_violation**2
```

### Variational Loss (VPINN)

Integrate by parts to reduce derivative order:

$$\int_\Omega (u_t + uu_x) v \, d\Omega = \int_\Omega \nu u_x v_x \, d\Omega$$

(See [VPINNs](../methods/vpinn.md) section)

---

## Loss Function Selection Guide

| PDE Type | Recommended Loss | Notes |
|----------|-----------------|-------|
| **Parabolic (heat)** | Data + PDE + IC | Standard formulation |
| **Hyperbolic (wave)** | Conservative + IC | Enforce conservation |
| **Elliptic (Laplace)** | Variational (VPINN) | Reduces derivative order |
| **Nonlinear (Burgers)** | Causal + adaptive weights | Important for stability |
| **Chaotic (Lorenz)** | Ensemble + statistics | Pointwise errors fail |

---

## Key Takeaways

✅ **Loss balancing is critical** for training stability  
✅ **Inverse variance weighting** is simple and effective  
✅ **NTK-guided weights** are theoretically motivated  
✅ **Causal training** ensures temporal consistency  
✅ **Problem-specific formulations** (conservative, entropy-stable) improve convergence  

## References

1. **Wang et al. (2024)**: Understanding and mitigating gradient flow pathologies in physics-informed neural networks  
   [arXiv:2001.04536](https://arxiv.org/abs/2001.04536)

2. **Kharazmi et al. (2019)**: Variational Physics-Informed Neural Networks  
   [arXiv:1912.00873](https://arxiv.org/abs/1912.00873)

3. **Jagtap & Karniadakis (2020)**: How important are loss components in physics-informed neural networks?  
   [arXiv:2006.13830](https://arxiv.org/abs/2006.13830)
