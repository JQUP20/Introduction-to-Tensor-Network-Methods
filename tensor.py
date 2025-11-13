"""
Tensor Network Library - Tensor Class
张量网络库 - 张量类

This module implements a general n-rank tensor class with basic and advanced operations:
- Initialization from arrays or random generation
- Tensor contraction
- Index fusion and reshaping
- Singular Value Decomposition (SVD)
- Tensor compression via truncated SVD
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from scipy.linalg import svd


class Tensor:
    """
    General n-rank tensor class with tensor network operations.

    Attributes:
    -----------
    data : np.ndarray
        The underlying tensor data
    shape : tuple
        Shape of the tensor (dimensions of each index)
    ndim : int
        Number of indices (rank) of the tensor
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize a tensor from a numpy array.

        Parameters:
        -----------
        data : np.ndarray
            Tensor data as a numpy array
        """
        self.data = np.array(data, dtype=complex)
        self.shape = self.data.shape
        self.ndim = len(self.shape)

    @classmethod
    def random(cls, shape: Tuple[int, ...], seed: Optional[int] = None) -> 'Tensor':
        """
        Create a random tensor with given shape.

        Parameters:
        -----------
        shape : tuple of int
            Shape of the tensor
        seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        Tensor
            Random tensor with complex entries
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random complex tensor
        real_part = np.random.randn(*shape)
        imag_part = np.random.randn(*shape)
        data = real_part + 1j * imag_part

        return cls(data)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> 'Tensor':
        """Create a tensor filled with zeros."""
        return cls(np.zeros(shape, dtype=complex))

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> 'Tensor':
        """Create a tensor filled with ones."""
        return cls(np.ones(shape, dtype=complex))

    @classmethod
    def eye(cls, dim: int) -> 'Tensor':
        """Create an identity matrix as a rank-2 tensor."""
        return cls(np.eye(dim, dtype=complex))

    def copy(self) -> 'Tensor':
        """Create a deep copy of the tensor."""
        return Tensor(self.data.copy())

    def conj(self) -> 'Tensor':
        """Return complex conjugate of the tensor."""
        return Tensor(np.conj(self.data))

    def norm(self) -> float:
        """Compute the Frobenius norm of the tensor."""
        return np.linalg.norm(self.data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.data.dtype})"

    def __str__(self) -> str:
        return f"Tensor with shape {self.shape}:\n{self.data}"

    # ========== Basic Tensor Operations ==========

    def contract(self, other: 'Tensor',
                 axes_self: Union[int, List[int]],
                 axes_other: Union[int, List[int]]) -> 'Tensor':
        """
        Contract this tensor with another tensor along specified axes.

        This performs generalized Einstein summation over the specified indices.
        For example, if we have tensors A[i,j,k] and B[j,l,m], contracting
        axis 1 of A with axis 0 of B gives C[i,k,l,m].

        Parameters:
        -----------
        other : Tensor
            The other tensor to contract with
        axes_self : int or list of int
            Axis/axes of this tensor to contract
        axes_other : int or list of int
            Axis/axes of other tensor to contract

        Returns:
        --------
        Tensor
            Result of the contraction

        Examples:
        ---------
        >>> A = Tensor(np.random.randn(3, 4, 5))
        >>> B = Tensor(np.random.randn(4, 5, 6))
        >>> C = A.contract(B, [1, 2], [0, 1])  # Contract indices 1,2 of A with 0,1 of B
        >>> C.shape
        (3, 6)
        """
        # Convert to lists if single integers
        if isinstance(axes_self, int):
            axes_self = [axes_self]
        if isinstance(axes_other, int):
            axes_other = [axes_other]

        # Validate input
        if len(axes_self) != len(axes_other):
            raise ValueError("Number of axes to contract must match")

        for ax_s, ax_o in zip(axes_self, axes_other):
            if self.shape[ax_s] != other.shape[ax_o]:
                raise ValueError(
                    f"Dimension mismatch: axis {ax_s} of self (dim={self.shape[ax_s]}) "
                    f"vs axis {ax_o} of other (dim={other.shape[ax_o]})"
                )

        # Use numpy's tensordot for efficient contraction
        result_data = np.tensordot(self.data, other.data, axes=(axes_self, axes_other))

        return Tensor(result_data)

    def trace(self, axis1: int, axis2: int) -> 'Tensor':
        """
        Take the trace over two indices of the tensor.

        Parameters:
        -----------
        axis1 : int
            First axis to trace over
        axis2 : int
            Second axis to trace over

        Returns:
        --------
        Tensor
            Tensor with two fewer indices
        """
        result_data = np.trace(self.data, axis1=axis1, axis2=axis2)
        return Tensor(result_data)

    # ========== Advanced Tensor Operations ==========

    def fuse_indices(self, indices: List[int]) -> Tuple['Tensor', Tuple[int, ...]]:
        """
        Fuse (combine) multiple indices into a single index.

        This operation combines several indices into one composite index.
        The new dimension is the product of the fused dimensions.

        Parameters:
        -----------
        indices : list of int
            List of indices to fuse (must be consecutive or will be permuted first)

        Returns:
        --------
        fused_tensor : Tensor
            Tensor with fused indices
        original_dims : tuple of int
            Original dimensions of the fused indices (needed for unfusing)

        Examples:
        ---------
        >>> T = Tensor(np.random.randn(2, 3, 4, 5))
        >>> T_fused, orig_dims = T.fuse_indices([1, 2])  # Fuse indices 1 and 2
        >>> T_fused.shape
        (2, 12, 5)  # 3*4 = 12
        >>> orig_dims
        (3, 4)
        """
        if not indices:
            return self.copy(), ()

        # Sort indices
        indices = sorted(indices)

        # Check if indices are consecutive
        if indices != list(range(indices[0], indices[-1] + 1)):
            # Need to permute first to make them consecutive
            # This is more complex - for now, require consecutive indices
            raise ValueError("Indices to fuse must be consecutive. Use permute() first if needed.")

        # Get dimensions to fuse
        original_dims = tuple(self.shape[i] for i in indices)
        fused_dim = np.prod(original_dims)

        # Build new shape
        new_shape = list(self.shape[:indices[0]])
        new_shape.append(fused_dim)
        new_shape.extend(self.shape[indices[-1] + 1:])

        # Reshape
        fused_data = self.data.reshape(new_shape)

        return Tensor(fused_data), original_dims

    def unfuse_index(self, index: int, original_dims: Tuple[int, ...]) -> 'Tensor':
        """
        Unfuse (split) a composite index back into multiple indices.

        Parameters:
        -----------
        index : int
            Index to unfuse
        original_dims : tuple of int
            Original dimensions to split into

        Returns:
        --------
        Tensor
            Tensor with unfused indices
        """
        # Check that dimensions match
        if self.shape[index] != np.prod(original_dims):
            raise ValueError(
                f"Dimension mismatch: index has dim {self.shape[index]} "
                f"but original dims multiply to {np.prod(original_dims)}"
            )

        # Build new shape
        new_shape = list(self.shape[:index])
        new_shape.extend(original_dims)
        new_shape.extend(self.shape[index + 1:])

        # Reshape
        unfused_data = self.data.reshape(new_shape)

        return Tensor(unfused_data)

    def reshape(self, new_shape: Tuple[int, ...]) -> 'Tensor':
        """
        Reshape the tensor to a new shape.

        The total number of elements must remain the same.

        Parameters:
        -----------
        new_shape : tuple of int
            New shape for the tensor

        Returns:
        --------
        Tensor
            Reshaped tensor
        """
        if np.prod(new_shape) != np.prod(self.shape):
            raise ValueError(
                f"Cannot reshape: total size must remain constant "
                f"(old: {np.prod(self.shape)}, new: {np.prod(new_shape)})"
            )

        reshaped_data = self.data.reshape(new_shape)
        return Tensor(reshaped_data)

    def permute(self, axes: List[int]) -> 'Tensor':
        """
        Permute (transpose) the tensor indices.

        Parameters:
        -----------
        axes : list of int
            New order of axes

        Returns:
        --------
        Tensor
            Permuted tensor
        """
        permuted_data = np.transpose(self.data, axes)
        return Tensor(permuted_data)

    def svd(self, left_indices: List[int],
            right_indices: Optional[List[int]] = None,
            full_matrices: bool = False) -> Tuple['Tensor', np.ndarray, 'Tensor']:
        """
        Perform Singular Value Decomposition on the tensor.

        This reshapes the tensor into a matrix by grouping indices, performs SVD,
        and returns U, S, Vt as tensors with appropriate shapes.

        For a tensor T with indices split into left and right groups,
        this computes: T = U @ diag(S) @ Vt

        Parameters:
        -----------
        left_indices : list of int
            Indices to group on the left (rows of matrix)
        right_indices : list of int, optional
            Indices to group on the right (columns of matrix)
            If None, uses all indices not in left_indices
        full_matrices : bool
            Whether to compute full or reduced SVD

        Returns:
        --------
        U : Tensor
            Left singular vectors with shape (*left_dims, bond_dim)
        S : np.ndarray
            Singular values (1D array)
        Vt : Tensor
            Right singular vectors with shape (bond_dim, *right_dims)

        Examples:
        ---------
        >>> T = Tensor(np.random.randn(3, 4, 5))
        >>> U, S, Vt = T.svd([0], [1, 2])
        >>> # U has shape (3, min(3, 20))
        >>> # S has length min(3, 20)
        >>> # Vt has shape (min(3, 20), 4, 5)
        """
        # Determine right indices if not specified
        if right_indices is None:
            all_indices = set(range(self.ndim))
            right_indices = sorted(all_indices - set(left_indices))

        # Validate indices
        all_specified = sorted(left_indices + right_indices)
        if all_specified != list(range(self.ndim)):
            raise ValueError("left_indices and right_indices must cover all indices exactly once")

        # Compute dimensions
        left_dims = [self.shape[i] for i in left_indices]
        right_dims = [self.shape[i] for i in right_indices]
        left_dim = np.prod(left_dims)
        right_dim = np.prod(right_dims)

        # Permute to group left and right indices
        perm = left_indices + right_indices
        permuted = self.permute(perm)

        # Reshape to matrix
        matrix = permuted.data.reshape(left_dim, right_dim)

        # Perform SVD
        U_mat, S, Vt_mat = svd(matrix, full_matrices=full_matrices, lapack_driver='gesvd')

        # Reshape U and Vt back to tensors
        bond_dim = len(S)
        U_shape = tuple(left_dims) + (bond_dim,)
        Vt_shape = (bond_dim,) + tuple(right_dims)

        U = Tensor(U_mat.reshape(U_shape))
        Vt = Tensor(Vt_mat.reshape(Vt_shape))

        return U, S, Vt

    def compress(self, left_indices: List[int],
                 right_indices: Optional[List[int]] = None,
                 max_bond_dim: Optional[int] = None,
                 cutoff: float = 1e-10,
                 relative_cutoff: bool = False) -> Tuple['Tensor', 'Tensor']:
        """
        Compress the tensor using truncated SVD.

        This splits the tensor into two tensors with a controlled bond dimension,
        minimizing the approximation error.

        Parameters:
        -----------
        left_indices : list of int
            Indices to group on the left tensor
        right_indices : list of int, optional
            Indices to group on the right tensor
        max_bond_dim : int, optional
            Maximum bond dimension to keep
        cutoff : float
            Cutoff for singular values (absolute or relative)
        relative_cutoff : bool
            If True, cutoff is relative to largest singular value

        Returns:
        --------
        left_tensor : Tensor
            Left tensor with shape (*left_dims, bond_dim)
        right_tensor : Tensor
            Right tensor with shape (bond_dim, *right_dims)

        Notes:
        ------
        The original tensor can be approximately reconstructed by contracting
        left_tensor and right_tensor along their shared bond dimension.
        """
        # Perform SVD
        U, S, Vt = self.svd(left_indices, right_indices, full_matrices=False)

        # Determine how many singular values to keep
        if relative_cutoff:
            threshold = cutoff * S[0]  # S[0] is the largest singular value
        else:
            threshold = cutoff

        # Find number of singular values above threshold
        n_keep = np.sum(S > threshold)

        # Apply max_bond_dim constraint
        if max_bond_dim is not None:
            n_keep = min(n_keep, max_bond_dim)

        # Ensure we keep at least one singular value
        n_keep = max(1, n_keep)

        # Truncate
        S_trunc = S[:n_keep]

        # Absorb singular values into right tensor (could also split or put in left)
        # U * S * Vt, we compute U and (S * Vt)
        U_trunc = Tensor(U.data[..., :n_keep])

        # Multiply S into Vt
        S_tensor = Tensor(S_trunc[:, np.newaxis])  # Shape (n_keep, 1)
        Vt_trunc = Tensor(Vt.data[:n_keep, ...])   # Shape (n_keep, *right_dims)

        # Reshape for multiplication: (n_keep,) times (n_keep, ...) -> (n_keep, ...)
        right_tensor = Tensor(S_trunc[:, np.newaxis] * Vt_trunc.data.reshape(n_keep, -1))
        right_tensor = right_tensor.reshape(Vt_trunc.shape)

        return U_trunc, right_tensor

    def truncation_error(self, left_indices: List[int],
                        right_indices: Optional[List[int]] = None,
                        max_bond_dim: Optional[int] = None,
                        cutoff: float = 1e-10) -> float:
        """
        Compute the truncation error for a given compression.

        Returns the discarded weight (sum of squared singular values).

        Parameters:
        -----------
        left_indices : list of int
            Indices to group on the left
        right_indices : list of int, optional
            Indices to group on the right
        max_bond_dim : int, optional
            Maximum bond dimension
        cutoff : float
            Cutoff for singular values

        Returns:
        --------
        float
            Truncation error (discarded weight)
        """
        # Perform SVD
        _, S, _ = self.svd(left_indices, right_indices, full_matrices=False)

        # Determine how many to keep
        n_keep = np.sum(S > cutoff)
        if max_bond_dim is not None:
            n_keep = min(n_keep, max_bond_dim)
        n_keep = max(1, n_keep)

        # Compute discarded weight
        if n_keep < len(S):
            discarded = np.sum(S[n_keep:]**2)
        else:
            discarded = 0.0

        return discarded


# Convenience functions for tensor operations
def tensor_contract(T1: Tensor, T2: Tensor,
                   axes1: Union[int, List[int]],
                   axes2: Union[int, List[int]]) -> Tensor:
    """Contract two tensors (convenience function)."""
    return T1.contract(T2, axes1, axes2)


def tensor_trace(T: Tensor, axis1: int, axis2: int) -> Tensor:
    """Take trace over two indices (convenience function)."""
    return T.trace(axis1, axis2)


def tensor_outer_product(T1: Tensor, T2: Tensor) -> Tensor:
    """
    Compute the outer product of two tensors.

    This creates a tensor with all indices from both tensors,
    with no contraction.
    """
    result_data = np.tensordot(T1.data, T2.data, axes=0)
    return Tensor(result_data)


if __name__ == "__main__":
    print("=" * 70)
    print("Tensor Network Library - Tensor Class Demo")
    print("=" * 70)

    # Example 1: Basic tensor creation and contraction
    print("\n1. Basic Tensor Operations")
    print("-" * 70)
    A = Tensor.random((3, 4, 5), seed=42)
    B = Tensor.random((4, 5, 6), seed=43)
    print(f"Tensor A: {A}")
    print(f"Tensor B: {B}")

    C = A.contract(B, [1, 2], [0, 1])
    print(f"\nContraction A[i,j,k] * B[j,k,l] = C[i,l]")
    print(f"Result C: {C}")
    print(f"C norm: {C.norm():.6f}")

    # Example 2: SVD decomposition
    print("\n2. Singular Value Decomposition")
    print("-" * 70)
    T = Tensor.random((4, 5, 6), seed=44)
    print(f"Original tensor: {T}")

    U, S, Vt = T.svd([0], [1, 2])
    print(f"\nSVD decomposition:")
    print(f"U shape: {U.shape}")
    print(f"S (singular values): {S}")
    print(f"Vt shape: {Vt.shape}")

    # Reconstruct and check error
    S_tensor = Tensor(np.diag(S))
    reconstructed = U.contract(S_tensor, 1, 0).contract(Vt, 1, 0)
    reconstruction_error = (T.data - reconstructed.data)
    print(f"\nReconstruction error: {np.linalg.norm(reconstruction_error):.2e}")

    # Example 3: Tensor compression
    print("\n3. Tensor Compression")
    print("-" * 70)
    T = Tensor.random((10, 10, 10), seed=45)
    print(f"Original tensor: {T}")
    print(f"Original norm: {T.norm():.6f}")

    # Compress with different bond dimensions
    for max_bond in [5, 3, 2]:
        left, right = T.compress([0], [1, 2], max_bond_dim=max_bond)
        print(f"\nCompression with max_bond_dim={max_bond}:")
        print(f"  Left tensor shape: {left.shape}")
        print(f"  Right tensor shape: {right.shape}")

        # Reconstruct
        reconstructed = left.contract(right, -1, 0)
        error = np.linalg.norm(T.data - reconstructed.data) / T.norm()
        print(f"  Relative error: {error:.6e}")

        trunc_error = T.truncation_error([0], [1, 2], max_bond_dim=max_bond)
        print(f"  Truncation error: {trunc_error:.6e}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
