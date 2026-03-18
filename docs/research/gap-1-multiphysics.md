---
id: gap-1-multiphysics
title: "Gap 1: Coupled Multi-Physics Systems"
sidebar_position: 2
---

# Gap 1: Robust Training for Coupled Multi-Physics and Multi-Scale Systems

## Problem Statement

Almost all PINN training advances are validated on **single-physics, single-scale benchmarks** (Burgers, advection-diffusion, simple Navier-Stokes). 

**Coupled multi-physics problems** introduce qualitatively new challenges:
- Fluid-structure interaction (FSI)
- Thermo-mechanical coupling
- Reactive transport with phase change
- Electromagneto-hydrodynamics

Loss terms from different physics have:
- Different characteristic scales (Pa vs. K)
- Different stiffness ratios
- Different convergence behaviors (elliptic, parabolic, hyperbolic)

## Why Current Methods Fail

### 1. Loss Imbalance Across Physics

Physics $A$: $\mathcal{L}_A \sim 10^2$ (large residuals)  
Physics $B$: $\mathcal{L}_B \sim 10^{-3}$ (small residuals)

Fixed weights → Physics A dominates, Physics B ignored

### 2. NTK Assumptions Break Down

- NTK theory assumes **single, unified loss landscape**
- Multi-physics = **conflicting loss landscapes**
- No analysis of when NTK weighting remains valid

### 3. Causal Training Complexity

Time-dependent multi-physics:
- Physics A evolves fast
- Physics B evolves slow
- Causal growth strategies must respect both timescales

## Specific Challenge: Fluid-Structure Interaction (FSI)

### Coupled System

$$\begin{cases}
\rho_f \left( \frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla) \mathbf{v} \right) = -\nabla p + \mu \nabla^2 \mathbf{v} & (\text{Navier-Stokes})\\
\nabla \cdot \mathbf{v} = 0 & (\text{continuity})\\
\rho_s \frac{\partial^2 \mathbf{u}_s}{\partial t^2} = \nabla \cdot \boldsymbol{\sigma} & (\text{elastodynamics})\\
\mathbf{u}_s|_{\Gamma} = \mathbf{u}_f|_{\Gamma} & (\text{kinematic coupling})\\
\boldsymbol{\sigma}_s \mathbf{n}|_{\Gamma} = -p_f \mathbf{n}|_{\Gamma} & (\text{traction coupling})
\end{cases}$$

### Loss Components

- **Fluid PDE**: $\mathcal{L}_f \sim 10^1$ (pressure range Pa)
- **Structure PDE**: $\mathcal{L}_s \sim 10^{-2}$ (displacement range mm)
- **Interface coupling**: $\mathcal{L}_{interface} \sim 10^{0}$
- **Data losses**: Vary by sensor type

**Problem**: No principled way to weight these!

## Research Directions

### 1. Multi-Scale Loss Hierarchy

**Idea**: Adaptively weight based on local physics scale

```python
def multi_physics_loss(u_f, u_s, x, t):
    """FSI example"""
    L_f = pde_loss_fluid(u_f)
    L_s = pde_loss_structure(u_s)
    L_coupling = interface_loss(u_f, u_s)
    
    # Adaptive weighting based on characteristic scales
    scale_f = compute_scale(u_f)  # ~10 Pa
    scale_s = compute_scale(u_s)  # ~10^-2 m
    
    # Weight by inverse characteristic scale
    w_f = 1.0 / scale_f
    w_s = 1.0 / scale_s
    w_c = 1.0 / compute_scale(u_f, u_s)  # Interface scale
    
    # Normalized multi-physics loss
    loss = (w_f*L_f + w_s*L_s + w_c*L_coupling) / (w_f + w_s + w_c)
    return loss
```

### 2. PDE-Aware Causal Training

Extend causal training to respect multi-physics timescales:

```python
def multi_physics_causal_loss(epoch, epochs_total):
    """Grow temporal domain respecting both physics"""
    
    t_max = (epoch / epochs_total) * T_final
    
    # Physics-specific growth rates
    t_fast = min(t_max, T_fast_physics * (epoch / epochs_total)**2)
    t_slow = min(t_max, T_slow_physics * (epoch / epochs_total)**0.5)
    
    # Separately enforce temporal causality per physics
    mask_fast = t_colloc <= t_fast
    mask_slow = t_colloc <= t_slow
    
    L_fast = (mask_fast * residual_fast**2).mean()
    L_slow = (mask_slow * residual_slow**2).mean()
    
    return L_fast + L_slow
```

### 3. Coupled Optimizer Schedules

Different physics may require different learning rates:

```python
# Separate optimizers
opt_f = Adam(model_f.parameters(), lr=1e-3)
opt_s = Adam(model_s.parameters(), lr=1e-4)

# Different schedules
schedule_f = ExponentialLR(opt_f, gamma=0.99)
schedule_s = ExponentialLR(opt_s, gamma=0.95)  # Slower decay

for epoch in range(epochs):
    L_f = fluid_loss()
    L_s = structure_loss()
    L_coupling = coupling_loss()
    
    # Optimize fluid
    opt_f.zero_grad()
    L_f.backward()
    opt_f.step()
    
    # Optimize structure
    opt_s.zero_grad()
    L_s.backward()
    opt_s.step()
    
    # Update schedules
    schedule_f.step()
    schedule_s.step()
```

## Open Questions

1. **Theoretical foundation**: When does NTK-based weighting work for multi-physics?
2. **Coupling strength**: How does interaction strength affect loss balancing?
3. **Stability**: Are there stability conditions ensuring convergence?
4. **Scalability**: How many coupled physics can one PINN handle?
5. **Generalization**: Do solutions transfer across coupling strengths?

## Promising Directions for Contribution

✅ **Develop systematic multi-physics loss framework** with theoretical justification  
✅ **Extend causal training** to multi-timescale systems  
✅ **Validate on benchmark problems** (FSI, thermoelasticity, reactive transport)  
✅ **Compare with classical solvers** on real industrial problems  
✅ **Provide software implementation** (DeepXDE extension?)

## Expected Impact

- **High**: Unlock industrial adoption of PINNs
- **Novelty**: First systematic framework for multi-physics PINNs
- **Venue**: NeurIPS, ICML, SIAM J. Scientific Computing

---

**See also**: [Research Gaps Overview](./gaps-overview.md)
