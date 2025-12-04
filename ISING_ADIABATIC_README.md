# Adiabatic Passage Through Quantum Critical Point: t-DMRG Simulation

## Overview

This project implements time-dependent Density Matrix Renormalization Group (t-DMRG) simulations to study adiabatic passage through the quantum critical point of the **Transverse Field Ising Model (TFIM)**.

### Physical System

The TFIM Hamiltonian is:

```
H(h) = -J Σᵢ σᵢᶻσᵢ₊₁ᶻ - h Σᵢ σᵢˣ
```

where:
- J is the Ising coupling (set to 1)
- h is the transverse field strength
- The quantum critical point is at h_c = 1

### Key Physics

1. **Adiabatic Evolution**: Starting from the ground state at h_initial > h_c (paramagnetic phase), we slowly ramp the field to h_final < h_c (ferromagnetic phase), crossing the quantum critical point.

2. **Landau-Zener Regime** (fast passage, small T):
   - The system doesn't have time to follow the instantaneous ground state
   - Residual excitation energy: δE ~ exp(-aT)

3. **Kibble-Zurek Regime** (slow passage, large T):
   - Near the critical point, the gap closes and adiabatic following breaks down
   - Residual excitation energy: δE ~ T^(-dν/(1+zν))
   - For 1D TFIM: d=1, z=1, ν=1 → **δE ~ T^(-1/2)**

## Implementation

### Core Components

1. **`MPS` class**: Matrix Product State representation
   - Efficient representation of 1D quantum states
   - Bond dimension χ controls approximation accuracy
   - Implements canonical forms for numerical stability

2. **`TransverseFieldIsing` class**: TFIM Hamiltonian
   - Two-site Hamiltonians for local updates
   - Exact ground state energy (for comparison)

3. **`GroundStateDMRG` class**: Ground state optimization
   - Variational optimization of MPS to find ground state
   - Two-site updates with SVD truncation
   - Energy convergence monitoring

4. **`TEBD` class**: Time evolution
   - Time-Evolving Block Decimation algorithm
   - Second-order Trotter decomposition
   - Automatic normalization to prevent numerical overflow

5. **`AdiabaticPassage` class**: Adiabatic evolution protocol
   - Linear ramping of transverse field
   - Configurable passage time and parameters
   - Real-time measurement and data collection

6. **`KibbleZurekAnalysis` class**: Scaling analysis
   - Automatic regime detection (LZ vs KZ)
   - Critical exponent extraction
   - Comparison with theoretical predictions

### Measurements

- **Energy**: ⟨H⟩ computed via two-site expectation values
- **Correlation length**: Estimated from entanglement (bond dimension)
- **Bond dimension growth**: Tracks entanglement during evolution

## Files

### Main Implementation
- **`ising_adiabatic_dmrg.py`**: Complete t-DMRG implementation
  - ~700 lines of documented code
  - All classes and algorithms
  - Plotting utilities

### Example Scripts
- **`run_ising_adiabatic.py`**: Comprehensive example suite
  - Example 1: Single adiabatic passage
  - Example 2: Kibble-Zurek scaling analysis
  - Example 3: System size comparison

- **`run_quick_demo.py`**: Fast demonstration
  - Reduced parameter space for quick results
  - ~5-10 minute runtime
  - Generates publication-quality plots

### Test Scripts
- **`test_ising_basic.py`**: Unit tests for basic functionality
- **`test_time_evolution.py`**: Numerical stability verification

## Usage

### Quick Start

```python
from ising_adiabatic_dmrg import *

# Setup
L = 20  # System size
model = TransverseFieldIsing(L)
params = DMRGParameters(chi_max=30, dt=0.02)
adiabatic = AdiabaticPassage(model, params)

# Run adiabatic passage
result = adiabatic.simulate(
    h_initial=2.0,    # Paramagnetic phase
    h_final=0.5,      # Ferromagnetic phase
    T_total=10.0,     # Total time
    save_every=10
)

# Analyze results
print(f"Final energy: {result['final_energy']}")
print(f"Correlation length: {result['final_xi']}")

# Plot
plot_adiabatic_passage_results(result, save_path='results.png')
```

### Running Examples

```bash
# Basic functionality test
python test_ising_basic.py

# Time evolution stability test
python test_time_evolution.py

# Quick demonstration (5-10 minutes)
python run_quick_demo.py

# Full example suite (30-60 minutes)
python run_ising_adiabatic.py
```

## Results

### Expected Outputs

The simulations generate several plots:

1. **Adiabatic Passage Trajectory**
   - Field h(t) vs time
   - Energy E(t) vs time
   - Bond dimension χ(t) vs time (entanglement growth)
   - Energy vs field (hysteresis-like plot)

2. **Kibble-Zurek Scaling**
   - Residual energy δE vs passage time T
   - Log-log plot with power-law fits
   - Comparison of fitted vs theoretical exponents

3. **System Size Comparison**
   - Scaling behavior for different L
   - Finite-size effects
   - Energy density δE/L

### Physical Interpretation

1. **Fast Passage (T small)**:
   - Large residual excitation energy
   - Exponential suppression with T (Landau-Zener)
   - Non-adiabatic transitions dominate

2. **Slow Passage (T large)**:
   - Smaller but non-zero residual energy
   - Power-law scaling: δE ~ T^(-1/2)
   - Kibble-Zurek mechanism: inevitability of defect formation
   - Critical slowing down near h_c

3. **Critical Region**:
   - Maximum entanglement growth near h_c  = 1
   - Bond dimension peaks at criticality
   - Correlation length diverges

## Parameters

### DMRG Parameters

```python
DMRGParameters(
    chi_max=50,           # Maximum bond dimension (accuracy vs cost)
    chi_init=10,          # Initial bond dimension
    n_sweeps=10,          # DMRG sweeps for ground state
    convergence_tol=1e-8, # Energy convergence tolerance
    dt=0.02,              # Time step (smaller = more accurate but slower)
    trotter_order=2       # Trotter decomposition order
)
```

### Recommended Values

| System Size | chi_max | dt | Notes |
|-------------|---------|-----|----|
| L ≤ 20 | 20-30 | 0.02 | Fast, good for testing |
| L = 20-40 | 30-50 | 0.01-0.02 | Production runs |
| L > 40 | 50-100 | 0.01 | Requires more memory/time |

### Trade-offs

- **Larger χ_max**: Better accuracy, higher computational cost (memory ~ χ³)
- **Smaller dt**: More accurate time evolution, more time steps
- **Larger L**: Closer to thermodynamic limit, exponentially harder

## Theoretical Background

### Kibble-Zurek Mechanism

When a system is driven through a continuous phase transition:

1. **Adiabatic regime** breaks down near the critical point where the gap closes
2. **Freeze-out time**: τ ~ |δh|^(-zν) where δh = h - h_c
3. **Defect density**: Scales as n ~ T^(-dν/(1+zν))
4. **Residual energy**: δE ~ n ~ T^(-dν/(1+zν))

For 1D TFIM:
- Critical exponents: z=1 (dynamical), ν=1 (correlation length)
- Dimension: d=1
- **Prediction: δE ~ T^(-1/2)**

### Universality

The exponent α = dν/(1+zν) is universal:
- Independent of microscopic details
- Depends only on dimension and universality class
- Same for all 1D quantum Ising transitions

## Numerical Methods

### Matrix Product States (MPS)

Efficient representation:
```
|ψ⟩ = Σ A[0]^s₀ A[1]^s₁ ... A[L-1]^sₗ₋₁ |s₀s₁...sₗ₋₁⟩
```

- Captures area-law entanglement
- Computational cost: O(L χ³)
- Memory: O(L χ² d)

### TEBD Algorithm

1. **Trotter decomposition**: exp(-iHdt) ≈ exp(-iH_even dt/2) exp(-iH_odd dt) exp(-iH_even dt/2)
2. **Two-site gates**: Apply exp(-iH_bond dt) locally
3. **SVD truncation**: Keep largest χ singular values
4. **Normalization**: Prevent numerical overflow

### Advantages of t-DMRG

- Handles dynamics through critical points
- Adaptive entanglement representation
- Exact for χ → ∞
- Efficient for 1D systems

## Validation

### Tests Implemented

1. **MPS normalization**: ⟨ψ|ψ⟩ = 1
2. **Ground state energy**: Comparison with exact diagonalization (L ≤ 20)
3. **Time evolution unitarity**: Norm conservation
4. **Critical exponent**: α ≈ 0.5 ± error

### Known Limitations

1. **1D only**: MPS efficient only in 1D
2. **Area law**: Fails for highly entangled states
3. **Finite χ**: Truncation errors accumulate
4. **Finite dt**: Trotter errors ~ O(dt³) for second-order
5. **Finite L**: Boundary effects, finite-size gap

## Extensions

Possible improvements and extensions:

1. **Higher-order Trotter**: 4th order for better accuracy
2. **Adaptive time stepping**: Larger dt away from h_c
3. **DMRG+TEBD hybrid**: Re-optimize with DMRG periodically
4. **Other models**: XX, XXZ, Hubbard models
5. **Non-linear ramps**: Optimal protocols
6. **Excited states**: Multiple defects, finite temperature

## References

### Kibble-Zurek Mechanism

1. Kibble, T. W. B. (1976). "Topology of cosmic domains and strings". J. Phys. A 9, 1387.
2. Zurek, W. H. (1985). "Cosmological experiments in superfluid helium?" Nature 317, 505.
3. Dziarmaga, J. (2010). "Dynamics of a quantum phase transition...". Adv. Phys. 59, 1063.

### Tensor Networks & DMRG

4. White, S. R. (1992). "Density matrix formulation...". Phys. Rev. Lett. 69, 2863.
5. Vidal, G. (2003). "Efficient classical simulation...". Phys. Rev. Lett. 91, 147902.
6. Schollwöck, U. (2011). "The density-matrix renormalization group...". Ann. Phys. 326, 96.

### Transverse Field Ising Model

7. Sachdev, S. (2011). "Quantum Phase Transitions" (2nd ed.). Cambridge University Press.
8. Pfeuty, P. (1970). "The one-dimensional Ising model...". Ann. Phys. 57, 79.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ising_adiabatic_dmrg,
  title={Adiabatic Passage Through Quantum Critical Point: t-DMRG Simulation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Introduction-to-Tensor-Network-Methods}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

*Last updated: 2024*
