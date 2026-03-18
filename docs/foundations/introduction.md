---
id: introduction
title: Foundations of PINNs
sidebar_position: 1
---

# Foundations of Physics-Informed Neural Networks

This section covers the core mathematical and conceptual foundations of PINNs.

## Overview

PINNs rest on three foundational pillars:

### 1. **Automatic Differentiation**
The ability to compute derivatives of neural network outputs with respect to inputs using computational graphs. This enables efficient computation of physics residuals embedded in the loss function.

### 2. **Physics as Regularization**
Governing equations (PDEs) act as strong inductive bias, regularizing the neural network toward solutions consistent with domain knowledge.

### 3. **Mesh-Free Collocation**
Rather than discretizing the domain (as in FEM), PINNs sample random or adaptive collocation points and enforce the PDE at those points.

---

## The Mathematical Formulation

### Forward Problem

Given a PDE:
$$\mathcal{F}(x, u, \nabla u, \nabla^2 u, \ldots) = 0$$

A PINN trains a neural network $u_\theta(x)$ to minimize:

$$\mathcal{L}_{\text{total}} = \lambda_d \mathcal{L}_d + \lambda_p \mathcal{L}_p$$

Where:
- **$\mathcal{L}_d$**: Data loss (supervised on observations)
  $$\mathcal{L}_d = \frac{1}{N_d} \sum_{i=1}^{N_d} |u_\theta(x_i) - u_i^{\text{obs}}|^2$$

- **$\mathcal{L}_p$**: Physics loss (unsupervised PDE residual)
  $$\mathcal{L}_p = \frac{1}{N_c} \sum_{j=1}^{N_c} |\mathcal{F}(x_j, u_\theta, \nabla u_\theta, \ldots)|^2$$

- **$\lambda_d, \lambda_p$**: Loss weights (critical for training stability)

### Inverse Problem

To discover unknown PDE coefficients (e.g., $\nu$ in Burgers equation):

1. Augment the network to output both $u$ and PDE coefficients
2. Include data loss on observations of $u$
3. Add regularization on coefficient values
4. Jointly optimize parameters and coefficients

---

## Key Papers

### Canonical Framework
1. **Raissi, Perdikaris, Karniadakis (2019)**: Physics-informed neural networks (PINNs)
   - Introduced continuous-time (collocation) formulation
   - Demonstrated forward and inverse problems
   - Canonical benchmark: Burgers equation

2. **Raissi, Perdikaris, Karniadakis (2017 Part II)**: Data-driven discovery of PDEs
   - Systematic approach to discovering PDE structure from data
   - Sparse identification framework
   - Demonstrated coefficient discovery

### Precursors

3. **Lagaris, Likas, Fotiadis (1998)**: Trial solution method
   - Historical foundation: neural networks for solving differential equations
   - First systematic treatment of boundary condition satisfaction

4. **Han, Jentzen, E (2018)**: Deep learning for high-dimensional PDEs
   - Solved up to 100-dimensional PDEs
   - Reformulation via stochastic control

---

## Continuous vs. Discrete Time

### Continuous-Time Formulation
$$\min_\theta \left\| u_\theta(x) - u^{\text{obs}} \right\|^2 + \left\| \mathcal{F}(u_\theta) \right\|^2$$

Advantages:
- ✅ Elegant, unified framework
- ✅ No time-stepping errors
- ✅ Naturally handles varying time scales

Challenges:
- ❌ Difficult for chaotic/turbulent systems
- ❌ Requires dense temporal samples near t=0

### Discrete-Time Formulation
Embed time-stepping (Runge-Kutta, implicit schemes) into the network:
$$u_\theta(t_{n+1}) = \text{RK}(u_\theta(t_n), \Delta t, \theta)$$

Advantages:
- ✅ Better long-time stability
- ✅ Causal training (recent breakthrough)
- ✅ Suits sequential data

Challenges:
- ❌ More complex to implement
- ❌ Computational overhead

---

## Learning Outcomes

After studying this section, you should understand:
- [ ] How automatic differentiation enables physics residuals
- [ ] The role of loss weights and loss balancing
- [ ] Forward vs. inverse problem formulations
- [ ] Continuous-time vs. discrete-time approaches
- [ ] The trade-offs between mesh-based and mesh-free methods

---

## Navigation

- **Next**: [Canonical PINN Framework](./canonical-framework.md) — Deep dive into the original architecture
- **Then**: [Survey Overview](./survey-overview.md) — Major variants and extensions
- **Core Concepts**: [Auto-Differentiation](./auto-differentiation.md), [Loss Functions](./loss-functions.md), [Optimization](./optimization.md)
