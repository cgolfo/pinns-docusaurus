---
id: fluids
title: Applications in Fluid Dynamics
sidebar_position: 1
---

# PINNs for Fluid Dynamics

Fluid dynamics is one of the most successful application domains for PINNs, with problems ranging from laminar Burgers flows to turbulent Navier-Stokes.

## Classic Benchmark: Burgers Equation

### Problem

1D viscous Burgers equation:

$$u_t + u u_x = \nu u_{xx}, \quad x \in [-1,1], t \in [0,1]$$

With:
- Initial: $u(x, 0) = -\sin(\pi x)$
- Boundary: $u(-1,t) = u(1,t) = 0$
- Viscosity: $\nu = 0.01/\pi$

### PINN Solution

```python
import deepxde as dde
import numpy as np

def pde(x, u):
    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_t = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    nu = 0.01 / np.pi
    return u_t + u * u_x - nu * u_xx

# Domain
spatial = dde.geometry.Interval(-1, 1)
temporal = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(spatial, temporal)

# Conditions
def initial(x):
    return -np.sin(np.pi * x[:, 0:1])

ic = dde.IC(geomtime, initial, lambda x, u: u, on_initial=True)
bc = dde.BC(geomtime, lambda x, u: u, lambda x: np.logical_or(x[:, 0:1] == -1, x[:, 0:1] == 1))

# Problem
data = dde.data.PDE(
    geomtime, pde, [bc, ic],
    num_domain=2500,
    num_boundary=100,
    num_initial=160
)

# Model
net = dde.nn.FeedForward(
    input_size=2,
    output_size=1,
    hidden_layers=[20, 20, 20, 20],
    activation="tanh"
)

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
history = model.train(epochs=10000)

# Fine-tune
model.compile("L-BFGS")
model.train(epochs=5000)
```

### Expected Results

| Metric | Expected |
|--------|----------|
| **Relative L² error** | < 10^-3 |
| **Training time** | 30–60 min (CPU) |
| **Inference time** | < 1 ms |

---

## Moderate Complexity: Cylinder Flow (2D)

### Problem

2D flow past a cylinder at intermediate Reynolds number (Re ~40):

$$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla) \mathbf{v} = -\nabla p + \nu \nabla^2 \mathbf{v}$$
$$\nabla \cdot \mathbf{v} = 0$$

With:
- Inflow: $\mathbf{v} = (1, 0)$
- Cylinder: No-slip boundary condition
- Domain: $[-2, 10] \times [-5, 5]$

### PINN Approach

```python
# Network for velocity and pressure
class NavierStokesNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_u = DenseNet(2, 64, 2)  # Velocity (u, v)
        self.net_p = DenseNet(2, 64, 1)  # Pressure
    
    def forward(self, x):
        u = self.net_u(x)
        p = self.net_p(x)
        return u, p

# PDE residuals
def ns_loss(x, y, u, v, p):
    """Navier-Stokes residuals"""
    u_x = grad(u, x)
    u_y = grad(u, y)
    v_x = grad(v, x)
    v_y = grad(v, y)
    
    p_x = grad(p, x)
    p_y = grad(p, y)
    
    # Momentum equations
    residual_u = ... # (based on NS formulation)
    residual_v = ...
    
    # Continuity
    residual_cont = u_x + v_y
    
    return residual_u, residual_v, residual_cont
```

### Challenges

- ❌ 2D Navier-Stokes much harder than Burgers
- ❌ Pressure field adds coupling complexity
- ❌ Boundary layer near cylinder requires fine resolution
- ✅ Solution: Adaptive collocation refinement (RAR)

---

## Advanced: Turbulent Navier-Stokes (3D)

### Problem

3D incompressible turbulent flow:

$$\frac{\partial u_i}{\partial t} + u_j \frac{\partial u_i}{\partial x_j} = -\frac{\partial p}{\partial x_i} + \nu \frac{\partial^2 u_i}{\partial x_j^2}$$

With Reynolds numbers Re ~ 1,000–10,000

### PINN Challenges

| Challenge | Why Hard | Solution |
|-----------|----------|----------|
| **High dimension** | 3D + time = 4D | Reduce domain via symmetry |
| **Chaotic dynamics** | Errors amplify | Use causal training, ensemble |
| **Multi-scale** | Large eddies + turbulence | Filter small scales (LES-PINN) |
| **Computational cost** | 3D collocation expensive | Use neural operators instead |

### Emerging Approach: LES-PINN

Use PINN for **large eddy simulation** (LES): only model large scales, filter small scales

```python
# Filter small scales
def filter_high_freq(u, filter_cutoff):
    """Remove frequencies > cutoff"""
    u_fft = fft(u)
    u_fft[freq > filter_cutoff] = 0
    return ifft(u_fft)

# Train PINN on filtered solution
u_filtered = filter_high_freq(u_ref, cutoff)

# LES closure model (separate NN)
def turbulent_stress(u_filtered):
    """Approximate unresolved stresses via NN"""
    return stress_net(u_filtered)

# Combined loss
def les_loss(u_pred, u_filtered_ref):
    u_filtered_pred = filter_high_freq(u_pred, cutoff)
    
    # Fit filtered solution
    L_data = (u_filtered_pred - u_filtered_ref)**2
    
    # Enforce filtered NS + turbulent closure
    residual = ns_pde_filtered(u_filtered_pred, turbulent_stress)
    L_pde = residual**2
    
    return L_data + L_pde
```

**Reference**: Beck et al. (2021) "Learning the structure of high-dimensional data with the Kolmogorov-Arnold Network"

---

## Hidden Fluid Mechanics (Inverse Problem)

### Problem

Given video of fluid flow (no explicit velocity measurements), infer velocity and pressure fields.

**Landmark application**: Raissi et al. (2020) *Science* paper

### Challenge

- ❌ Only observe scalar visualizations (dye, particles)
- ❌ No direct velocity/pressure measurements
- ❌ Highly under-determined inverse problem

### PINN Solution

```python
# Training data: intensity field I(x, t) from video
I_observed = load_video_frames()

# Learn velocity field u, v jointly with intensity advection
def intensity_advection_pde(x, t, I, u, v):
    """Intensity advection equation: I_t + u*I_x + v*I_y = 0"""
    I_t = grad_t(I)
    I_x = grad_x(I)
    I_y = grad_y(I)
    
    return I_t + u*I_x + v*I_y  # Should be ~0

# Loss combines:
# - Data loss on observed intensity
# - PDE loss on intensity advection
# - PDE loss on Navier-Stokes (for velocities)

loss = data_loss(I_pred, I_obs) + \
       pde_loss(intensity_advection_pde) + \
       pde_loss(navier_stokes_pde)
```

**Results**:
- ✅ Reconstructs velocity/pressure without explicit BC
- ✅ Works from passive visualization (dye, particles)
- ✅ Validated against reference CFD solutions

---

## Comparison: PINNs vs. Classical CFD

| Aspect | PINN | CFD (FVM/LES) |
|--------|------|--------------|
| **Training/Setup** | Hours | Days |
| **Inference** | <10 ms | 1–100 s |
| **Data required** | Sparse | Dense grid |
| **Mesh handling** | Mesh-free | Mesh-dependent |
| **Accuracy** | Moderate (10^-2–10^-3) | High (10^-4–10^-6) |
| **Inverse problems** | ✅ Easy | ❌ Hard |

**Verdict**: PINNs excel at:
- ✅ Real-time inference
- ✅ Inverse problems (parameter discovery)
- ✅ Sparse-data scenarios
- ❌ But not as accurate as classical CFD

---

## Practical Challenges in Fluid Applications

### 1. Boundary Condition Handling

No-slip BC for viscous flows must be carefully enforced:

```python
# Dirichlet BC: u = 0 on wall
bc_loss_wall = (u_pred_wall)**2

# Robin BC (slip): u + alpha*du/dn = 0 (more general)
bc_loss_slip = (u_pred + alpha*du_dn)**2
```

### 2. Pressure Decoupling

Incompressibility constraint $\nabla \cdot \mathbf{v} = 0$ is hard to enforce:

**Solution 1**: Use velocity potential $\mathbf{v} = \nabla \phi$ (automatic div-free)

**Solution 2**: Penalty term: $\lambda (\nabla \cdot \mathbf{v})^2$

### 3. Sharp Gradients

Viscous shear layers near walls need special care:

```python
# Adaptive refinement near walls
x_wall_dist = distance_to_wall(x_colloc)
refine_mask = x_wall_dist < boundary_layer_thickness

# Increase collocation density near walls
x_refined = add_collocation_points(x_refine_mask)
```

---

## State-of-the-Art (2025)

### Breakthrough: Causal PINNs for Temporal Evolution

Wang et al. (2024) shows causal training dramatically improves long-time accuracy

### PINN-Hybrid Approaches

Coupling PINNs with:
- **Reduced-order models** (ROM)
- **Classical time-stepping**
- **Finite element preconditioners**

### Neural Operators

For parametric flows (varying Re, geometry), use:
- **FNO** (Fourier Neural Operator)
- **DeepONet** (Deep Operator Network)
- **GraphKAN** (graph-based)

---

## Recommended Resources

- **DeepXDE Examples**: [github.com/lululxvi/deepxde/examples](https://github.com/lululxvi/deepxde/examples)
- **Raissi et al. Codes**: [github.com/maziarraissi/PINNs](https://github.com/maziarraissi/PINNs)
- **NVIDIA Modulus**: [developer.nvidia.com/modulus](https://developer.nvidia.com/modulus) (production-level)

---

## Key Takeaways

✅ **Burgers → proof of concept** (10^-3 error easily achievable)  
✅ **Cylinder flow → intermediate challenge** (careful BC handling)  
✅ **Turbulence → frontier** (LES-PINN, neural operators recommended)  
✅ **Inverse fluids → unique strength** (better than classical methods)  
✅ **Long-time → causal training essential** (recent breakthrough)

See [Applications Overview](../applications) for other domains.
