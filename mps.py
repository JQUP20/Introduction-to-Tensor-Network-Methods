"""
Matrix Product State (MPS) Class
矩阵乘积态类

This module implements the Matrix Product State representation and operations:
- MPS initialization (random, product state)
- Norm computation
- Local and nearest-neighbor operator expectation values
- Canonicalization (left/right canonical forms)
- MPS compression
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from tensor import Tensor


class MPS:
    """
    Matrix Product State representation.

    An MPS represents a quantum state as a chain of tensors:
    |ψ⟩ = Σ A[1]^{s1} A[2]^{s2} ... A[L]^{sL} |s1,s2,...,sL⟩

    Each tensor A[i] has shape (D_{i-1}, d_i, D_i) where:
    - D_{i-1}, D_i are bond dimensions (left and right virtual indices)
    - d_i is the physical dimension at site i

    For boundary sites:
    - A[0] has shape (1, d_0, D_0) or (d_0, D_0)
    - A[L-1] has shape (D_{L-2}, d_{L-1}, 1) or (D_{L-2}, d_{L-1})

    Attributes:
    -----------
    L : int
        Number of sites (length of the chain)
    tensors : list of Tensor
        List of MPS tensors, one per site
    physical_dims : list of int
        Physical dimensions at each site
    bond_dims : list of int
        Bond dimensions (length L+1, including boundaries = 1)
    """

    def __init__(self, tensors: List[Tensor]):
        """
        Initialize MPS from a list of tensors.

        Parameters:
        -----------
        tensors : list of Tensor
            List of MPS tensors. Each should have 3 indices: (left_bond, physical, right_bond)
            Boundary tensors can have bond dimension 1 at boundaries.
        """
        self.L = len(tensors)
        if self.L == 0:
            raise ValueError("MPS must have at least one site")

        self.tensors = [t.copy() for t in tensors]

        # Extract dimensions
        self.physical_dims = []
        self.bond_dims = [1]  # Left boundary

        for i, tensor in enumerate(self.tensors):
            if tensor.ndim != 3:
                raise ValueError(f"MPS tensor at site {i} must be rank-3, got rank-{tensor.ndim}")

            D_left, d, D_right = tensor.shape
            self.physical_dims.append(d)
            self.bond_dims.append(D_right)

            # Check bond dimension consistency
            if D_left != self.bond_dims[i]:
                raise ValueError(
                    f"Bond dimension mismatch at site {i}: "
                    f"expected {self.bond_dims[i]}, got {D_left}"
                )

        # Check right boundary
        if self.bond_dims[-1] != 1:
            raise ValueError(f"Right boundary must have bond dimension 1, got {self.bond_dims[-1]}")

    @classmethod
    def random(cls, L: int, d: Union[int, List[int]], D: int, seed: Optional[int] = None) -> 'MPS':
        """
        Create a random MPS.

        Parameters:
        -----------
        L : int
            Number of sites
        d : int or list of int
            Physical dimension(s). If int, same for all sites.
        D : int
            Maximum bond dimension
        seed : int, optional
            Random seed

        Returns:
        --------
        MPS
            Random MPS (not normalized)
        """
        if isinstance(d, int):
            physical_dims = [d] * L
        else:
            physical_dims = d
            if len(physical_dims) != L:
                raise ValueError(f"Length of physical_dims must match L={L}")

        if seed is not None:
            np.random.seed(seed)

        # Calculate bond dimensions first
        # Bond dimensions grow from left until they reach D or the maximum possible
        bond_dims = [1]  # Left boundary
        for i in range(L - 1):
            # Maximum bond dimension is limited by Hilbert space dimension
            max_possible = min(np.prod(physical_dims[:i + 1]), np.prod(physical_dims[i + 1:]))
            bond_dim = min(D, max_possible)
            bond_dims.append(bond_dim)
        bond_dims.append(1)  # Right boundary

        # Create tensors with consistent bond dimensions
        tensors = []
        for i in range(L):
            shape = (bond_dims[i], physical_dims[i], bond_dims[i + 1])
            tensor = Tensor.random(shape, seed=None if seed is None else seed + i)
            tensors.append(tensor)

        return cls(tensors)

    @classmethod
    def product_state(cls, states: List[np.ndarray]) -> 'MPS':
        """
        Create an MPS representing a product state.

        Parameters:
        -----------
        states : list of np.ndarray
            Local quantum state at each site (1D arrays)

        Returns:
        --------
        MPS
            MPS with bond dimension 1 representing the product state
        """
        L = len(states)
        tensors = []

        for i, state in enumerate(states):
            # State should be a 1D array
            state = np.array(state, dtype=complex)
            if state.ndim != 1:
                raise ValueError(f"State at site {i} must be 1D, got shape {state.shape}")

            d = len(state)
            # Shape: (1, d, 1) for MPS tensor
            tensor_data = state.reshape(1, d, 1)
            tensors.append(Tensor(tensor_data))

        return cls(tensors)

    def copy(self) -> 'MPS':
        """Create a deep copy of the MPS."""
        return MPS([t.copy() for t in self.tensors])

    def __repr__(self) -> str:
        return f"MPS(L={self.L}, physical_dims={self.physical_dims}, bond_dims={self.bond_dims})"

    # ========== Norm and Inner Product ==========

    def norm(self) -> float:
        """
        Compute the norm of the MPS: ||ψ||.

        This contracts the MPS with its conjugate:
        ⟨ψ|ψ⟩ = Σ A*[1] A*[2] ... A*[L] A[L] ... A[2] A[1]

        Returns:
        --------
        float
            Norm of the state
        """
        # Start with identity on the left boundary (dimension 1x1)
        # We'll build the contraction from left to right
        E = np.ones((1, 1), dtype=complex)

        for i in range(self.L):
            A = self.tensors[i].data  # Shape: (D_left, d, D_right)
            A_conj = np.conj(A)       # Complex conjugate

            # Contract: E[α,α'] * A[α,s,β] * A*[α',s,β']
            # Result: E_new[β,β']

            # First contract E with A: E[α,α'] * A[α,s,β] = temp[α',s,β]
            temp = np.tensordot(E, A, axes=([0], [0]))  # Contract left bond of A

            # Then contract with A_conj: temp[α',s,β] * A*[α',s,β'] = E_new[β,β']
            # Contract over α' and s
            E = np.tensordot(temp, A_conj, axes=([0, 1], [0, 1]))

        # E should now be 1x1
        assert E.shape == (1, 1), f"Expected (1,1), got {E.shape}"
        return np.sqrt(np.abs(E[0, 0]))

    def normalize(self) -> float:
        """
        Normalize the MPS in place.

        Returns:
        --------
        float
            The original norm (before normalization)
        """
        norm_val = self.norm()
        if norm_val > 1e-14:
            # Normalize the first tensor
            self.tensors[0].data /= norm_val
        return norm_val

    def inner_product(self, other: 'MPS') -> complex:
        """
        Compute the inner product with another MPS: ⟨self|other⟩.

        Parameters:
        -----------
        other : MPS
            Another MPS

        Returns:
        --------
        complex
            Inner product ⟨self|other⟩
        """
        if self.L != other.L:
            raise ValueError("MPS must have the same length")

        # Build contraction from left to right
        E = np.ones((1, 1), dtype=complex)

        for i in range(self.L):
            A_self = np.conj(self.tensors[i].data)  # Bra: conjugate
            A_other = other.tensors[i].data          # Ket

            # Contract: E[α,α'] * A_self*[α,s,β] * A_other[α',s,β']
            temp = np.tensordot(E, A_self, axes=([0], [0]))
            E = np.tensordot(temp, A_other, axes=([0, 1], [0, 1]))

        return E[0, 0]

    # ========== Expectation Values ==========

    def expect_local(self, operator: np.ndarray, site: int) -> complex:
        """
        Compute expectation value of a local operator at a given site.

        ⟨O_i⟩ = ⟨ψ|O_i|ψ⟩

        Parameters:
        -----------
        operator : np.ndarray
            Local operator matrix (d × d)
        site : int
            Site index (0 to L-1)

        Returns:
        --------
        complex
            Expectation value ⟨O_i⟩
        """
        if site < 0 or site >= self.L:
            raise ValueError(f"Site {site} out of range [0, {self.L - 1}]")

        d = self.physical_dims[site]
        if operator.shape != (d, d):
            raise ValueError(f"Operator shape must be ({d}, {d}), got {operator.shape}")

        # Build left environment (contract sites 0 to site-1)
        E_left = np.ones((1, 1), dtype=complex)
        for i in range(site):
            A = self.tensors[i].data
            A_conj = np.conj(A)
            temp = np.tensordot(E_left, A, axes=([0], [0]))
            E_left = np.tensordot(temp, A_conj, axes=([0, 1], [0, 1]))

        # Apply operator at site
        A = self.tensors[site].data      # Shape: (D_left, d, D_right)
        A_conj = np.conj(A)

        # Contract: E_left[α,α'] * A[α,s,β] = temp[α',s,β]
        temp = np.tensordot(E_left, A, axes=([0], [0]))

        # Apply operator: temp[α',s,β] * O[s,s'] = temp2[α',β,s']
        temp2 = np.tensordot(temp, operator, axes=([1], [1]))

        # Contract with conjugate: temp2[α',β,s'] * A*[α',s',β'] = E_site[β,β']
        E_site = np.tensordot(temp2, A_conj, axes=([0, 2], [0, 1]))

        # Build right environment (contract sites site+1 to L-1)
        E_right = np.ones((1, 1), dtype=complex)
        for i in range(self.L - 1, site, -1):
            A = self.tensors[i].data      # Shape: (D_left, d, D_right)
            A_conj = np.conj(A)
            # Contract: A[α,s,β] * E_right[β,β'] = temp[α,s,β']
            temp = np.tensordot(A, E_right, axes=([2], [0]))
            # Contract: A*[α',s,β'] * temp[α,s,β'] = E_new[α',α] -> need to swap -> [α,α']
            E_right = np.tensordot(A_conj, temp, axes=([1, 2], [1, 2])).T

        # Final contraction: E_site[β,β'] * E_right[β,β']
        result = np.tensordot(E_site, E_right, axes=([0, 1], [0, 1]))

        return result

    def expect_nn(self, operator: np.ndarray, site: int) -> complex:
        """
        Compute expectation value of a nearest-neighbor operator.

        ⟨O_{i,i+1}⟩ = ⟨ψ|O_{i,i+1}|ψ⟩

        Parameters:
        -----------
        operator : np.ndarray
            Two-site operator matrix (d1*d2 × d1*d2) in basis |s1,s2⟩
            Can also be (d1, d2, d1, d2) tensor
        site : int
            Left site index (operator acts on sites site and site+1)

        Returns:
        --------
        complex
            Expectation value ⟨O_{i,i+1}⟩
        """
        if site < 0 or site >= self.L - 1:
            raise ValueError(f"Site {site} out of range [0, {self.L - 2}]")

        d1 = self.physical_dims[site]
        d2 = self.physical_dims[site + 1]

        # Reshape operator if needed
        if operator.shape == (d1 * d2, d1 * d2):
            # Reshape to (d1, d2, d1, d2)
            operator = operator.reshape(d1, d2, d1, d2)
        elif operator.shape != (d1, d2, d1, d2):
            raise ValueError(
                f"Operator shape must be ({d1 * d2}, {d1 * d2}) or ({d1}, {d2}, {d1}, {d2}), "
                f"got {operator.shape}"
            )

        # Build left environment (contract sites 0 to site-1)
        E_left = np.ones((1, 1), dtype=complex)
        for i in range(site):
            A = self.tensors[i].data
            A_conj = np.conj(A)
            temp = np.tensordot(E_left, A, axes=([0], [0]))
            E_left = np.tensordot(temp, A_conj, axes=([0, 1], [0, 1]))

        # Contract two-site tensor
        A1 = self.tensors[site].data      # Shape: (D_left, d1, D_mid)
        A2 = self.tensors[site + 1].data  # Shape: (D_mid, d2, D_right)
        A1_conj = np.conj(A1)
        A2_conj = np.conj(A2)

        # Contract A1 and A2: A1[α,s1,β] * A2[β,s2,γ] = A12[α,s1,s2,γ]
        A12 = np.tensordot(A1, A2, axes=([2], [0]))  # Shape: (D_left, d1, d2, D_right)

        # Contract with left environment: E_left[α,α'] * A12[α,s1,s2,γ] = temp[α',s1,s2,γ]
        temp = np.tensordot(E_left, A12, axes=([0], [0]))

        # Apply operator: temp[α',s1,s2,γ] * O[s1,s2,s1',s2'] = temp2[α',γ,s1',s2']
        temp2 = np.tensordot(temp, operator, axes=([1, 2], [0, 1]))

        # Contract with conjugate: A1*[α',s1',β'] * A2*[β',s2',γ']
        A12_conj = np.tensordot(A1_conj, A2_conj, axes=([2], [0]))  # Shape: (D_left, d1, d2, D_right)

        # Contract: temp2[α',γ,s1',s2'] * A12*[α',s1',s2',γ'] = E_site[γ,γ']
        E_site = np.tensordot(temp2, A12_conj, axes=([0, 2, 3], [0, 1, 2]))

        # Build right environment (contract sites site+2 to L-1)
        E_right = np.ones((1, 1), dtype=complex)
        for i in range(self.L - 1, site + 1, -1):
            A = self.tensors[i].data      # Shape: (D_left, d, D_right)
            A_conj = np.conj(A)
            # Contract: A[α,s,β] * E_right[β,β'] = temp[α,s,β']
            temp = np.tensordot(A, E_right, axes=([2], [0]))
            # Contract: A*[α',s,β'] * temp[α,s,β'] = E_new[α',α] -> need to swap -> [α,α']
            E_right = np.tensordot(A_conj, temp, axes=([1, 2], [1, 2])).T

        # Final contraction
        result = np.tensordot(E_site, E_right, axes=([0, 1], [0, 1]))

        return result

    def expect_operator_sum(self, operators: List[np.ndarray], operator_type: str = 'local') -> complex:
        """
        Compute expectation value of a sum of operators.

        For local operators: ⟨Σ_i O_i⟩
        For nn operators: ⟨Σ_i O_{i,i+1}⟩

        Parameters:
        -----------
        operators : list of np.ndarray
            List of operators (one per site or bond)
        operator_type : str
            'local' or 'nn' (nearest-neighbor)

        Returns:
        --------
        complex
            Total expectation value
        """
        if operator_type == 'local':
            if len(operators) != self.L:
                raise ValueError(f"Expected {self.L} operators, got {len(operators)}")
            return sum(self.expect_local(op, i) for i, op in enumerate(operators))
        elif operator_type == 'nn':
            if len(operators) != self.L - 1:
                raise ValueError(f"Expected {self.L - 1} operators, got {len(operators)}")
            return sum(self.expect_nn(op, i) for i, op in enumerate(operators))
        else:
            raise ValueError(f"Unknown operator type: {operator_type}")

    # ========== Canonicalization ==========

    def left_canonicalize(self, site: int) -> np.ndarray:
        """
        Left-canonicalize the MPS tensor at given site using QR decomposition.

        After this, the tensor satisfies: Σ_s A[s]† A[s] = I

        Parameters:
        -----------
        site : int
            Site to canonicalize

        Returns:
        --------
        R : np.ndarray
            Upper triangular matrix to be absorbed into next site
        """
        if site < 0 or site >= self.L:
            raise ValueError(f"Site {site} out of range")

        A = self.tensors[site].data  # Shape: (D_left, d, D_right)
        D_left, d, D_right = A.shape

        # Reshape to matrix: (D_left * d, D_right)
        A_mat = A.reshape(D_left * d, D_right)

        # QR decomposition
        Q, R = np.linalg.qr(A_mat)

        # Reshape Q back to tensor
        new_D_right = Q.shape[1]
        self.tensors[site].data = Q.reshape(D_left, d, new_D_right)

        # Update bond dimension
        self.bond_dims[site + 1] = new_D_right

        return R

    def right_canonicalize(self, site: int) -> np.ndarray:
        """
        Right-canonicalize the MPS tensor at given site using QR decomposition.

        After this, the tensor satisfies: Σ_s A[s] A[s]† = I

        Parameters:
        -----------
        site : int
            Site to canonicalize

        Returns:
        --------
        L : np.ndarray
            Lower triangular matrix to be absorbed into previous site
        """
        if site < 0 or site >= self.L:
            raise ValueError(f"Site {site} out of range")

        A = self.tensors[site].data  # Shape: (D_left, d, D_right)
        D_left, d, D_right = A.shape

        # Reshape to matrix: (D_left, d * D_right)
        A_mat = A.reshape(D_left, d * D_right)

        # QR decomposition on transpose
        Q, R = np.linalg.qr(A_mat.T)
        L = R.T
        Q = Q.T

        # Reshape Q back to tensor
        new_D_left = Q.shape[0]
        self.tensors[site].data = Q.reshape(new_D_left, d, D_right)

        # Update bond dimension
        self.bond_dims[site] = new_D_left

        return L

    def canonicalize(self, center: int = 0):
        """
        Bring MPS to canonical form with orthogonality center at given site.

        Sites to the left of center are left-canonical.
        Sites to the right of center are right-canonical.

        Parameters:
        -----------
        center : int
            Site for the orthogonality center
        """
        # Left-canonicalize sites 0 to center-1
        for i in range(center):
            R = self.left_canonicalize(i)
            if i < self.L - 1:
                # Absorb R into next site
                self.tensors[i + 1].data = np.tensordot(R, self.tensors[i + 1].data, axes=([1], [0]))

        # Right-canonicalize sites L-1 down to center+1
        for i in range(self.L - 1, center, -1):
            L = self.right_canonicalize(i)
            if i > 0:
                # Absorb L into previous site
                self.tensors[i - 1].data = np.tensordot(self.tensors[i - 1].data, L, axes=([2], [0]))


if __name__ == "__main__":
    print("=" * 70)
    print("Matrix Product State (MPS) Demo")
    print("=" * 70)

    # Example 1: Create random MPS and compute norm
    print("\n1. Random MPS and Norm Computation")
    print("-" * 70)
    L = 6
    d = 2  # Spin-1/2
    D = 4  # Bond dimension

    mps = MPS.random(L=L, d=d, D=D, seed=42)
    print(f"MPS: {mps}")
    print(f"Norm: {mps.norm():.6f}")

    original_norm = mps.norm()
    mps.normalize()
    print(f"After normalization: {mps.norm():.6f}")

    # Example 2: Product state
    print("\n2. Product State (all spins up)")
    print("-" * 70)
    spin_up = np.array([1.0, 0.0])
    states = [spin_up] * L
    mps_up = MPS.product_state(states)
    print(f"Product state MPS: {mps_up}")
    print(f"Norm: {mps_up.norm():.6f}")

    # Example 3: Local operator expectation value
    print("\n3. Local Operator Expectation Values")
    print("-" * 70)

    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Expectation value of sigma_z on all-up state
    print("All-up state: ⟨σ_z⟩ at each site:")
    for i in range(L):
        exp_val = mps_up.expect_local(sigma_z, i)
        print(f"  Site {i}: {exp_val.real:.6f}")

    # Random MPS
    print("\nRandom MPS: ⟨σ_z⟩ at each site:")
    for i in range(L):
        exp_val = mps.expect_local(sigma_z, i)
        print(f"  Site {i}: {exp_val.real:.6f}")

    # Example 4: Nearest-neighbor operator
    print("\n4. Nearest-Neighbor Operator Expectation Values")
    print("-" * 70)

    # σ_z ⊗ σ_z operator
    sigma_zz = np.kron(sigma_z, sigma_z).reshape(2, 2, 2, 2)

    print("Random MPS: ⟨σ_z σ_z⟩ for each bond:")
    for i in range(L - 1):
        exp_val = mps.expect_nn(sigma_zz, i)
        print(f"  Bond ({i},{i + 1}): {exp_val.real:.6f}")

    # Example 5: Inner product
    print("\n5. Inner Product")
    print("-" * 70)
    mps2 = MPS.random(L=L, d=d, D=D, seed=100)
    mps2.normalize()

    overlap = mps.inner_product(mps2)
    print(f"⟨mps|mps⟩ = {mps.inner_product(mps):.6f}")
    print(f"⟨mps|mps2⟩ = {overlap:.6f}")
    print(f"|⟨mps|mps2⟩| = {abs(overlap):.6f}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
