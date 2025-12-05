"""
Test time evolution stability
"""

import numpy as np
from ising_adiabatic_dmrg import (
    TransverseFieldIsing,
    GroundStateDMRG,
    AdiabaticPassage,
    DMRGParameters,
    Measurements
)

print("="*70)
print("Testing Time Evolution Stability")
print("="*70)

# Small system for quick testing
L = 10
h_initial = 2.0
h_final = 0.5
T = 2.0

# Parameters with small time step
params = DMRGParameters(
    chi_max=20,
    chi_init=8,
    n_sweeps=5,
    convergence_tol=1e-6,
    dt=0.02
)

print(f"\nSystem: L={L}, h_i={h_initial}, h_f={h_final}, T={T}")
print(f"Time step: dt={params.dt}")
print(f"Number of steps: {int(T/params.dt)}")

# Create model
model = TransverseFieldIsing(L)
adiabatic = AdiabaticPassage(model, params)

# Get ground state at final field
print("\nComputing ground state...")
dmrg = GroundStateDMRG(model, params)
_, E_gs = dmrg.run(h_final)
print(f"Ground state energy at h={h_final}: {E_gs:.6f}")

# Run adiabatic passage
print(f"\nRunning adiabatic passage...")
result = adiabatic.simulate(h_initial, h_final, T, save_every=10)

print(f"\nResults:")
print(f"  Final energy: {result['final_energy']:.6f}")
print(f"  Ground energy: {E_gs:.6f}")
print(f"  Residual δE: {result['final_energy'] - E_gs:.6e}")
print(f"  Max bond dim: {max(result['bond_dims'])}")

# Check if energy is reasonable
if np.abs(result['final_energy']) < 1e6:
    print("\n✓ Time evolution is numerically stable!")
else:
    print("\n✗ WARNING: Numerical instability detected!")

# Check that energy stays finite throughout
all_finite = all(np.isfinite(result['energies']))
print(f"\nAll energies finite: {all_finite}")

if all_finite:
    print("✓ All tests passed!")
else:
    print("✗ Some energies became infinite!")
