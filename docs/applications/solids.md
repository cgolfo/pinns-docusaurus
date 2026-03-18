---
id: solids
title: Applications in Solid Mechanics
sidebar_position: 2
---

# PINNs for Solid Mechanics

Solid mechanics presents unique opportunities for PINNs: inverse problems, parameter discovery, and time-dependent nonlinear behavior.

## Linear Elasticity (Foundational)

### Problem

Static linear elasticity:

$$\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = 0$$
$$\boldsymbol{\sigma} = 2\mu \boldsymbol{\varepsilon} + \lambda (\text{tr} \boldsymbol{\varepsilon}) \mathbf{I}$$
$$\boldsymbol{\varepsilon} = \frac{1}{2}(\nabla \mathbf{u} + (\nabla \mathbf{u})^T)$$

Where:
- $\mathbf{u}$: Displacement
- $\boldsymbol{\sigma}$: Stress
- $\boldsymbol{\varepsilon}$: Strain
- $\lambda, \mu$: Lamé parameters

### PINN Approach

Network outputs displacement $\mathbf{u}(x, y)$ for 2D:

```python
def elasticity_pde(x_y, u_v):
    """Linear elasticity PDE"""
    u, v = u_v[:, 0:1], u_v[:, 1:2]
    
    # Compute strains
    u_x = grad(u, x_y, axis=0)
    u_y = grad(u, x_y, axis=1)
    v_x = grad(v, x_y, axis=0)
    v_y = grad(v, x_y, axis=1)
    
    # Stress (constitutive relation)
    sigma_xx = 2*mu*u_x + lambda*(u_x + v_y)
    sigma_yy = 2*mu*v_y + lambda*(u_x + v_y)
    sigma_xy = mu*(u_y + v_x)
    
    # Equilibrium equations
    eq_x = grad(sigma_xx, x_y, axis=0) + grad(sigma_xy, x_y, axis=1) + f_x
    eq_y = grad(sigma_xy, x_y, axis=0) + grad(sigma_yy, x_y, axis=1) + f_y
    
    return eq_x, eq_y
```

**Key advantage**: Better error bounds than CFD (PDE is linear)

---

## Inverse Problems: Parameter Discovery

### Example: Material Property Identification

Given:
- Measured displacement at sensors
- Known applied forces
- Unknown: material properties (Young's modulus E, Poisson's ratio ν)

**Classical approach**: Expensive, non-unique

**PINN approach**: Learn E, ν directly via augmented network

```python
class ElasticityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.displacement_net = DenseNet(2, 64, 2)
        
        # Learnable material properties
        self.E = nn.Parameter(torch.tensor([1.0]))
        self.nu = nn.Parameter(torch.tensor([0.3]))
    
    def forward(self, x_y):
        u, v = self.displacement_net(x_y)
        return u, v
    
    def elastic_constants(self):
        """Convert E, nu to lambda, mu"""
        E = self.E.clamp(min=0.1)  # Ensure positive
        nu = self.nu.clamp(-1, 0.5)  # Ensure stable
        mu = E / (2*(1 + nu))
        lambda_ = E*nu / ((1+nu)*(1-2*nu))
        return mu, lambda_

# Loss: data + PDE + regularization
def inverse_loss(model, x_sensors, u_measured):
    u, v = model(x_sensors)
    
    # Data loss (match measurements)
    L_data = ((u - u_measured)**2).mean()
    
    # PDE loss (enforced everywhere)
    # ... elasticity_pde residual ...
    
    # Regularization (encourage realistic parameters)
    E, nu = model.E.item(), model.nu.item()
    L_reg = (E - E_prior)**2 + (nu - nu_prior)**2
    
    return L_data + L_pde + lambda_reg*L_reg
```

---

## Nonlinear Elasticity & Hyperelasticity

For large deformations:

$$\boldsymbol{\sigma} = 2\frac{\partial W}{\partial \mathbf{C}} \mathbf{C}^{-T}$$

where $W(\mathbf{C})$ is stored energy function (e.g., Neo-Hookean)

**PINN benefit**: Learn $W$ directly from deformation data!

---

## Fracture Mechanics & Damage

### Phase-Field Fracture Model

Smeared crack representation via order parameter $d(x)$ (0=intact, 1=broken):

$$\frac{\partial d}{\partial t} = ... \quad \text{(phase-field evolution)}$$

**PINN strength**: Learn fracture pattern without explicit tracking

---

## Recommended Resources

- **DeepXDE elasticity examples**: [github.com/lululxvi/deepxde/tree/master/examples/pinn](https://github.com/lululxvi/deepxde/tree/master/examples/pinn)
- **NVIDIA Modulus solid mechanics**: [docs.nvidia.com/modulus/user-guide/physics-informed-neural-networks/pinn-solid-mechanics.html](https://docs.nvidia.com/modulus)

---

**See also**: [Electromagnetic Applications](./electromagnetics.md), [Inverse Problems](./inverse-problems.md)
