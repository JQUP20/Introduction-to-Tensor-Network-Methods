"""
Quick test of basic functionality
"""

import numpy as np
from ising_adiabatic_dmrg import (
    TransverseFieldIsing,
    GroundStateDMRG,
    AdiabaticPassage,
    DMRGParameters,
    Measurements,
    MPS
)

print("="*70)
print("Basic Functionality Test")
print("="*70)

# Test 1: MPS creation
print("\n1. Testing MPS creation...")
L = 10
mps = MPS(L, chi=5)
print(f"   Created MPS with L={L}")
print(f"   Bond dimensions: {mps.get_bond_dimensions()}")
print(f"   Norm: {mps.norm():.6f}")
print("   ✓ MPS creation successful")

# Test 2: TFIM model
print("\n2. Testing TFIM model...")
model = TransverseFieldIsing(L)
H_bond = model.two_site_hamiltonian(h=1.0, site=0)
print(f"   Created TFIM model with L={L}")
print(f"   Two-site Hamiltonian shape: {H_bond.shape}")
print("   ✓ TFIM model creation successful")

# Test 3: Ground state DMRG
print("\n3. Testing ground state DMRG...")
params = DMRGParameters(
    chi_max=20,
    chi_init=5,
    n_sweeps=5,
    convergence_tol=1e-6,
    dt=0.05
)
dmrg = GroundStateDMRG(model, params)
mps_gs, energy = dmrg.run(h=0.5)
print(f"   Ground state energy at h=0.5: {energy:.6f}")
print(f"   Max bond dimension: {max(mps_gs.get_bond_dimensions())}")
print("   ✓ Ground state DMRG successful")

# Test 4: Measurements
print("\n4. Testing measurements...")
E_measured = Measurements.measure_energy(mps_gs, model, h=0.5)
xi = Measurements.measure_correlation_length(mps_gs)
print(f"   Measured energy: {E_measured:.6f}")
print(f"   Correlation length: {xi:.6f}")
print("   ✓ Measurements successful")

# Test 5: Short adiabatic passage
print("\n5. Testing adiabatic passage...")
adiabatic = AdiabaticPassage(model, params)
result = adiabatic.simulate(
    h_initial=2.0,
    h_final=0.5,
    T_total=1.0,
    save_every=5
)
print(f"   Initial field: h_i = 2.0")
print(f"   Final field: h_f = 0.5")
print(f"   Total time: T = 1.0")
print(f"   Final energy: {result['final_energy']:.6f}")
print(f"   Final correlation length: {result['final_xi']:.6f}")
print("   ✓ Adiabatic passage successful")

print("\n" + "="*70)
print("All basic tests passed!")
print("="*70)
