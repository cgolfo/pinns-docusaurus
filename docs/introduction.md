---
id: introduction
title: Introduction to PINNs
sidebar_position: 1
slug: /
---

# Physics-Informed Neural Networks

**Physics-Informed Neural Networks (PINNs)** are a revolutionary approach to solving partial differential equations (PDEs) by embedding physical laws directly into neural network loss functions. This enables mesh-free PDE solving with remarkably little data.

## What are PINNs?

PINNs merge two powerful paradigms:
- **Physics**: Domain knowledge encoded as differential equations
- **Deep Learning**: Neural networks as universal function approximators

By combining these, PINNs can:
- ✅ Solve PDEs with sparse, noisy data
- ✅ Learn PDE coefficients directly from observations (inverse problems)
- ✅ Eliminate the need for computational meshes (mesh-free)
- ✅ Scale to high-dimensional problems
- ✅ Naturally handle irregular boundary conditions

## The Core Insight

The fundamental innovation is using **automatic differentiation** to compute derivatives of the neural network output. These derivatives are then embedded as physics residuals in the loss function:

```
Loss = Data Loss + Physics Loss
     = ||u_pred - u_obs||² + ||Residual(u_pred)||²
```

Where:
- **Data Loss**: Matches predictions to observations
- **Physics Loss**: Enforces that predictions satisfy the governing PDE

## A Brief History

| Year | Milestone |
|------|-----------|
| **1998** | Lagaris et al. propose using NNs for solving ODEs/PDEs |
| **2017** | Raissi, Perdikaris, Karniadakis publish canonical PINN framework |
| **2019** | PINN paper exceeds 1,000 citations; becomes foundational |
| **2020** | Hidden fluid mechanics published in *Science* |
| **2021** | Nature Reviews Physics survey; DeepXDE library released |
| **2024** | PINNacle benchmark standardizes evaluation; causal training breakthrough |
| **2026** | ~65 key papers spanning foundations to cutting-edge training methods |

## Citation Impact

The canonical PINN paper (Raissi et al., 2019) has accumulated **over 16,000 citations** and is one of the most cited papers in scientific machine learning.

---

### Quick Navigation

- **New to PINNs?** Start with [Why PINNs](./why-pinns.md)
- **Want to code?** Jump to [Quick Start](./quick-start.md)
- **Interested in theory?** Explore [Foundations](./foundations/introduction.md)
- **Looking for current research gaps?** See [Research Gaps](./research/gaps-overview.md)
- **Need references?** Browse [Literature](./literature/foundational.md)

---

## Key Applications

PINNs have been successfully applied to:

🌊 **Fluid Dynamics**: Navier-Stokes, turbulence, flow control  
🏗️ **Solid Mechanics**: Elasticity, fracture mechanics, wave propagation  
⚡ **Electromagnetics**: Maxwell equations, wave propagation  
🔬 **Chemistry**: Reaction-diffusion systems, kinetics  
🌍 **Geophysics**: Seismic wave propagation, subsurface flow  
🏥 **Biology**: Population dynamics, neural modeling  

---

## The Community

This documentation aggregates insights from **~65 foundational and recent papers**, including:
- Canonical frameworks and surveys
- Major algorithmic variants (VPINNs, cPINNs, XPINNs, B-PINNs)
- Neural operators for parametric PDEs
- Training and optimization advances
- Emerging research frontiers

**Curated by**: The PINN research community  
**Last updated**: March 2026  
**Repository sources**: idrl-lab, erfanhamdi/awesome-pinn, Event-AHU  
