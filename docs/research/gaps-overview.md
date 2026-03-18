---
id: gaps-overview
title: Research Gaps & Open Problems
sidebar_position: 1
---

# Research Gaps & Open Problems in PINNs

Based on systematic analysis of ~65 papers spanning foundational work through early 2026, this section identifies **8 concrete, under-explored research frontiers** ripe for novel contributions suitable for top-tier venues.

---

## Summary Table

| # | Gap | Key Challenge | Status | Difficulty |
|---|-----|----------------|--------|-----------|
| 1 | **Coupled Multi-Physics** | Loss hierarchies in multi-scale systems | Unsolved | Hard |
| 2 | **NTK Theory (Nonlinear)** | Theory breaks down; no training diagnostics | Theoretical | Hard |
| 3 | **Hyperparameter Selection** | Sensitivity; no general AutoML framework | Practical | Medium |
| 4 | **Long-Time Integration** | Chaotic dynamics; error accumulation | Algorithmic | Medium |
| 5 | **Error Certification** | No a posteriori bounds; no safety guarantees | Mathematical | Very Hard |
| 6 | **Complex 3D Geometry** | Industrial-scale domain handling inefficient | Practical | Hard |
| 7 | **Hybrid Workflows** | FEM+PINN integration; systematic frameworks lacking | Practical | Medium |
| 8 | **Transfer Learning** | Generalization across PDE families inefficient | Algorithmic | Medium |

---

## Gap 1: Robust Training for Coupled Multi-Physics and Multi-Scale Systems

### The Problem

Almost all PINN training advances (causal weighting, NTK balancing, adaptive sampling) are validated on **single-physics, single-scale benchmarks** — typically Burgers, advection-diffusion, or simple Navier-Stokes.

**Coupled multi-physics problems** introduce fundamentally different challenges:
- Fluid-structure interaction (FSI)
- Thermo-mechanical coupling
- Reactive transport with phase change
- Electromagneto-hydrodynamics

### Why It's Hard

Loss terms from different physics have:
- ❌ Different **characteristic scales** (Pa vs. K)
- ❌ Different **stiffness ratios** (fast vs. slow dynamics)
- ❌ Different **convergence behaviors** (elliptic, parabolic, hyperbolic)
- ❌ Different **optimal loss weights**

No loss balancing method (NTK-guided, causal, eigenvalue-based) has been rigorously tested or theoretically analyzed for multi-physics coupling.

### Research Opportunity

Develop a **systematic framework** for:
- Automatic loss hierarchies based on PDE characteristics
- Multi-scale collocation strategies
- Coupled optimizer schedules
- Theoretical guarantees for multi-physics convergence

---

## Gap 2: NTK Theory and Training Dynamics for Nonlinear PDEs

### The Problem

**Key insight** from NeurIPS 2024 (multiple authors): **NTK theory fundamentally breaks down for nonlinear PDEs**.

Traditional assumptions:
- ✗ Kernel random at initialization
- ✗ Kernel **fixed** during training (infinite-width limit)
- ✗ Hessian vanishes at infinite width

**Reality for nonlinear PDEs**:
- ✓ Kernel **evolves** during training
- ✓ Network transitions from lazy to feature-learning regime
- ✓ Hessian **remains significant** even at infinite width
- ✓ Feature emergence essential for solving nonlinear equations

### Current Diagnostics Are Broken

All existing PINN training diagnostics assume linear NTK:
- ❌ Wang et al.'s NTK weighting (assumes fixed kernel)
- ❌ Eigenvalue-based scheduling (assumes lazy training)
- ❌ Causal training (heuristic, lacks theory)

### Research Opportunity

Develop rigorous theoretical framework via:
- **Mean-field theory** for PINN training dynamics
- **Dynamical systems analysis** of feature emergence
- **Feature-learning NTK** extensions to nonlinear PDEs
- Actionable training diagnostics with theoretical backing

---

## Gap 3: Automated Hyperparameter and Architecture Selection

### The Problem

PINNs remain **notoriously sensitive** to hyperparameter choices:
- Network depth/width
- Activation functions
- Loss weights (λ_p, λ_d)
- Optimizer schedules
- Collocation point density
- Domain decomposition granularity

**Current state**:
- Wang et al.'s "Expert's Guide" (2023) provides best practices
- "Auto-PINN" attempted initial framework but limited scope
- **No general-purpose neural architecture search for PDE-specific structure**

### Why It Matters

- ❌ Non-expert adoption blocked
- ❌ Reproducibility across problems inconsistent
- ❌ Expensive trial-and-error in real applications

### Research Opportunity

Develop **Auto-PINN v2** incorporating:
- **Neural architecture search** (NAS) optimized for PDEs
- **Bayesian hyperparameter optimization** with PDE priors
- **Problem-specific encodings** (e.g., equation type, domain geometry, BC complexity)
- **Efficient strategies** (multi-fidelity, transfer learning from similar problems)

---

## Gap 4: Long-Time Integration and Chaotic Dynamics Beyond Causal Training

### The Problem

**Wang et al.'s causal training (2024)** was a breakthrough for time-dependent PDEs, enabling correct temporal causality and improved long-time accuracy.

**But it's still limited**:
- Long-time integration of chaotic/turbulent systems (beyond ~10 Lyapunov times) **remains essentially unsolved**
- Current PINNs **accumulate errors exponentially**
- Hybrid time-stepping (sequential, overlapping windows) helps but **sacrifices mesh-free elegance**

### Examples of Failure

- ❌ Kolmogorov flow (chaotic fluid dynamics): Diverges within 5 eddy-turnover times
- ❌ Double gyre (ocean model): Errors grow exponentially beyond 10 days
- ❌ Lorenz system (classic chaos): Predictions fail beyond initial Lyapunov time

### Research Opportunity

Develop **principled long-horizon methods**:
- **Shadowing-based trajectory correction** (mathematics of chaos theory)
- **Symplectic structure preservation** for Hamiltonian systems
- **Attractor-aware training objectives** (statistical rather than pointwise accuracy)
- **Ensemble methods** (multiple PINN predictions with uncertainty)
- **Hybrid sequential-continuous approaches** with principled error control

---

## Gap 5: Rigorous A Posteriori Error Estimation and Certification

### The Problem

**Traditional numerical methods** (FEM, FVM) have:
- ✅ Well-established **a posteriori error estimators**
- ✅ Convergence **certificates**
- ✅ **Guaranteed bounds** on solution error

**PINNs have**:
- ❌ Essentially **none**
- ❌ PDE residual used as proxy, but relationship to true error is **unvalidated**
- ❌ No methods for certifying **safety-critical constraints**

### Current Practice

Engineers assume:
- "If PDE residual is small, solution is correct" ← **This is often wrong!**
- Validation only via expensive comparison to reference solutions
- No way to certify max stress bounds in structural problems

### Research Opportunity

Develop **rigorous error bounds** via:
- **Stability analysis** of residual-to-error mapping
- **Inf-sup stability verification** (Babuška-Brezzi conditions)
- **Energy stability** proofs for specific PDE classes
- **Localized error bounds** for subdomains
- **Constraint certification** for engineering safety requirements

---

## Gap 6: Efficient Handling of Complex 3D Industrial Geometries

### The Problem

The vast majority of PINN demonstrations use:
- Simple 2D domains (rectangles, circles)
- Simple 3D domains (cubes, spheres, cylinders)

**Real industrial geometries** involve:
- Turbine blades (thin features, curvature)
- Cardiovascular networks (branching, lumens)
- Microelectronics (multi-connected, anisotropic)

Uniform collocation is **wildly inefficient** for such geometries.

### Current Approaches (Fragmented)

- Geometry-aware collocation strategies (studied individually)
- Implicit boundary representations: SDFs, R-functions (emerging)
- Adaptive refinement near geometric features (case-by-case)
- NVIDIA Modulus platform (practical but not principled)

**No integrated framework.**

### Research Opportunity

Develop **geometry-aware PINN pipeline**:
- **CAD-to-collocation** integration (automatic mesh-less sampling from CAD)
- **Signed distance functions** (SDFs) for implicit geometry representation
- **Adaptive refinement** near edges, thin features, high-curvature regions
- **Boundary layer handling** (specialized collocation near walls)
- **Validation** on real-world industrial problems

---

## Gap 7: Bridging PINNs with Legacy Numerical Solvers in Hybrid Workflows

### The Problem

**Engineering reality**: Decades of **validated FEM/FVM infrastructure** exist.

**Opportunity**: PINNs could complement, not replace:
- Preconditioners for iterative solvers
- Coarse-grid correctors in multigrid
- Surrogate generators for boundary conditions
- Sub-grid models in large-eddy simulation (LES)

**Current state**: 
- cPINNs and XPINNs decompose by subdomain (still all-PINN)
- **No systematic hybrid integration frameworks**
- Interface conditions, error propagation largely unexplored

### Why It Matters

Practical adoption requires:
- Hybrid workflows where PINNs handle sparse-data regions
- Classical solvers handle well-resolved regions
- Guaranteed stability and error bounds

### Research Opportunity

Develop **hybrid numerical-ML framework**:
- **Interface theory** (weak coupling, strong coupling, loose coupling)
- **Stability analysis** for coupled PINN-FEM systems
- **Error propagation** through hybrid pipeline
- **Software stack** integrating FEniCS/Firedrake + PINN libraries
- **Industrial case studies** (FSI, thermal-mechanical)

---

## Gap 8: Transfer Learning and Generalization Across PDE Families

### The Problem

**Current bottleneck**: Each new PDE instance requires **training from scratch**.

While some meta-learning exists (Psaros et al., 2021; Penwarden et al., 2021):
- **Systematic transfer learning** across related PDEs is inefficient
- No foundation-model approach for PDE solvers
- Pretraining on PDE families largely unexplored

### Examples of Untapped Transfer

- ✗ Train on Burgers → fast fine-tune for advection-diffusion
- ✗ Train on Navier-Stokes (Re=100) → predict for Re=200, 500, 1000
- ✗ Train on rectangular domains → predict on new geometries
- ✗ Train on one material property → predict for 1000s of property values

### Research Opportunity

Develop **foundation models for PDE solvers**:
- **Pretraining** on large synthetic/public PDE datasets
- **Representation learning** for solution structure across PDE families
- **In-context learning** (few-shot adaptation to new problems)
- **Large pretrained models** (similar to LLM paradigm)
- **Transfer + uncertainty quantification** (Bayesian transfer)
- **Multi-task learning** (simultaneous training on related problems)

---

## How These Gaps Are Connected

```
Gap 1 (Multi-Physics) 
    ↓ requires
Gap 2 (NTK Theory) → improved training diagnostics
    ↓ enables
Gap 3 (AutoML) → automatic tuning for multi-physics
    ↓ solved problems used in
Gap 8 (Transfer Learning) → fast adaptation to new multi-physics
    ↓ deployed via
Gap 7 (Hybrid Workflows) → FEM + learned preconditioner
    ↓ validated using
Gap 5 (Error Bounds) → certified performance
    ↓ applied to
Gap 6 (Complex Geometry) → industrial problems
    ↓ overcomes via
Gap 4 (Long-Time) → accurate long-horizon predictions
```

---

## Why These Gaps Matter for Top-Tier Venues

✅ **Novel contributions**: Each gap represents unsolved problems  
✅ **High impact**: Solutions address practical bottlenecks  
✅ **Theoretical depth**: Several require new mathematics  
✅ **Empirical validation**: Benchmarks available for comparison  
✅ **Community interest**: Active interest at NeurIPS, ICML, ICLR, SIAM, JCP  

---

## Next Steps

- Explore individual gaps: [Gap 1](./gap-1-multiphysics.md), [Gap 2](./gap-2-ntk-theory.md), etc.
- Review candidate methods: [Training Advances](../training/overview.md)
- See applications: [Use Cases](../applications/fluids.md)
