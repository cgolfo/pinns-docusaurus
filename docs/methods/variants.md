---
id: variants
title: PINN Variants & Extensions
sidebar_position: 1
---

# PINN Variants and Extensions

This section surveys major algorithmic variants and extensions to the canonical PINN framework.

## Overview

Since the original PINN work, numerous variants have been proposed to address specific challenges:

| Variant | Key Feature | Use Case |
|---------|------------|----------|
| **VPINN** | Petrov-Galerkin formulation | Reduces derivative orders |
| **hp-VPINN** | Domain decomposition + refinement | Large-scale problems |
| **cPINN** | Conservative/flux continuity | Conservation laws |
| **XPINN** | Extended domain decomposition | Parallel training |
| **B-PINN** | Bayesian inference | Uncertainty quantification |
| **fPINN** | Fractional PDEs | Non-integer derivatives |
| **Mod-PINN** | Modified loss weighting | Loss balancing |
| **Causal PINN** | Temporal causality | Time-dependent PDEs |

## Major Variants

### VPINNs (Variational PINNs)

**Reference**: Kharazmi et al. (2019)

Reformulates using Petrov-Galerkin weak form:
- Reduces derivative orders via integration by parts
- Improves stability for high-order PDEs
- Better for problems with boundary layer phenomena

### cPINNs (Conservative PINNs)

**Reference**: Jagtap et al. (2020)

Domain decomposition with flux continuity:
- Automatically enforces conservation laws
- Enables parallel training
- Suitable for multi-domain problems

### XPINNs (Extended PINNs)

**Reference**: Jagtap & Karniadakis (2020)

Space-time domain decomposition:
- Separate networks per subdomain
- Full tensor product decomposition
- Better parallelization than cPINNs

### B-PINNs (Bayesian PINNs)

**Reference**: Yang et al. (2021)

Bayesian framework for uncertainty:
- Posterior inference on parameters
- Confidence intervals on predictions
- Noise characterization

---

*This section covers major variants. See full documentation for detailed mathematical formulations, implementation guides, and application examples.*
