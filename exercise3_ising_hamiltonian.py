"""
练习 3：Ising 哈密顿量
Exercise 3: Ising Hamiltonian

实现横向场 Ising 模型（Transverse Field Ising Model, TFIM）：
H^λ = -∑ᵢ σᵢˣσᵢ₊₁ˣ + λ ∑ᵢ σᵢᶻ

研究量子相变、能级交叉和有限尺寸效应
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_la
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time


class IsingHamiltonianAnalyzer:
    """横向场 Ising 模型分析器"""

    def __init__(self, N: int):
        """
        初始化分析器

        Parameters:
        -----------
        N : int
            自旋数量
        """
        self.N = N
        self.dim = 2**N  # 希尔伯特空间维度
        self.H = None
        self.eigenvalues = None
        self.eigenvectors = None

        # Pauli 矩阵
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(2, dtype=complex)

    def tensor_product(self, matrices: List[np.ndarray]) -> np.ndarray:
        """
        计算矩阵的张量积

        Parameters:
        -----------
        matrices : List[np.ndarray]
            矩阵列表

        Returns:
        --------
        result : np.ndarray
            张量积结果
        """
        result = matrices[0]
        for mat in matrices[1:]:
            result = np.kron(result, mat)
        return result

    def construct_sigma_x_i(self, i: int) -> np.ndarray:
        """
        构造作用在第 i 个格点上的 σˣ 算符

        Parameters:
        -----------
        i : int
            格点索引（从 0 开始）

        Returns:
        --------
        operator : np.ndarray
            σˣᵢ 算符
        """
        matrices = []
        for j in range(self.N):
            if j == i:
                matrices.append(self.sigma_x)
            else:
                matrices.append(self.identity)
        return self.tensor_product(matrices)

    def construct_sigma_z_i(self, i: int) -> np.ndarray:
        """
        构造作用在第 i 个格点上的 σᶻ 算符

        Parameters:
        -----------
        i : int
            格点索引（从 0 开始）

        Returns:
        --------
        operator : np.ndarray
            σᶻᵢ 算符
        """
        matrices = []
        for j in range(self.N):
            if j == i:
                matrices.append(self.sigma_z)
            else:
                matrices.append(self.identity)
        return self.tensor_product(matrices)

    def construct_sigma_x_i_sigma_x_j(self, i: int, j: int) -> np.ndarray:
        """
        构造两格点相互作用算符 σˣᵢ σˣⱼ

        Parameters:
        -----------
        i, j : int
            格点索引

        Returns:
        --------
        operator : np.ndarray
            σˣᵢσˣⱼ 算符
        """
        matrices = []
        for k in range(self.N):
            if k == i or k == j:
                matrices.append(self.sigma_x)
            else:
                matrices.append(self.identity)
        return self.tensor_product(matrices)

    def construct_hamiltonian_dense(self, lambda_field: float) -> np.ndarray:
        """
        构造 Ising 哈密顿量（稠密矩阵）

        H^λ = -∑ᵢ σᵢˣσᵢ₊₁ˣ + λ ∑ᵢ σᵢᶻ

        Parameters:
        -----------
        lambda_field : float
            横向场强度 λ

        Returns:
        --------
        H : np.ndarray
            哈密顿矩阵
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)

        # 第一项：相邻自旋相互作用 -∑ᵢ σᵢˣσᵢ₊₁ˣ
        for i in range(self.N - 1):
            sigma_x_i_sigma_x_ip1 = self.construct_sigma_x_i_sigma_x_j(i, i + 1)
            H -= sigma_x_i_sigma_x_ip1

        # 第二项：横向场 λ ∑ᵢ σᵢᶻ
        for i in range(self.N):
            sigma_z_i = self.construct_sigma_z_i(i)
            H += lambda_field * sigma_z_i

        self.H = H
        return H

    def construct_hamiltonian_sparse_efficient(self, lambda_field: float) -> sparse.csr_matrix:
        """
        高效构造 Ising 哈密顿量（稀疏矩阵，使用位操作）

        这个方法比逐个构造算符更快，特别是对于大的 N

        Parameters:
        -----------
        lambda_field : float
            横向场强度 λ

        Returns:
        --------
        H : sparse.csr_matrix
            稀疏哈密顿矩阵
        """
        from scipy.sparse import lil_matrix

        H = lil_matrix((self.dim, self.dim), dtype=complex)

        # 对每个基态 |b₁b₂...bₙ⟩
        for basis_idx in range(self.dim):
            # 横向场项 λ ∑ᵢ σᵢᶻ
            # σᵢᶻ |...bᵢ...⟩ = (-1)^bᵢ |...bᵢ...⟩
            diagonal_element = 0.0
            for i in range(self.N):
                bit = (basis_idx >> i) & 1  # 提取第 i 个比特
                if bit == 0:  # spin up: +1
                    diagonal_element += lambda_field
                else:  # spin down: -1
                    diagonal_element -= lambda_field

            H[basis_idx, basis_idx] = diagonal_element

            # 相互作用项 -∑ᵢ σᵢˣσᵢ₊₁ˣ
            # σᵢˣσᵢ₊₁ˣ 翻转第 i 和 i+1 个自旋
            for i in range(self.N - 1):
                # 翻转第 i 和 i+1 个比特
                flipped_idx = basis_idx ^ (1 << i) ^ (1 << (i + 1))
                H[basis_idx, flipped_idx] -= 1.0

        return H.tocsr()

    def diagonalize(self, use_sparse: bool = False, k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        对角化哈密顿量

        Parameters:
        -----------
        use_sparse : bool
            是否使用稀疏矩阵算法
        k : int, optional
            如果使用稀疏算法，计算的本征态数量（默认：全部或 min(6, dim-1)）

        Returns:
        --------
        eigenvalues : np.ndarray
            本征值（按递增顺序）
        eigenvectors : np.ndarray
            本征向量
        """
        if self.H is None:
            raise ValueError("请先构造哈密顿量")

        if use_sparse:
            if k is None:
                k = min(6, self.dim - 2)
            # 使用稀疏矩阵算法计算最低的 k 个本征态
            eigenvalues, eigenvectors = sparse_la.eigsh(
                self.H, k=k, which='SA'  # SA = Smallest Algebraic
            )
        else:
            # 对厄米矩阵使用 eigh（更快且数值稳定）
            if sparse.issparse(self.H):
                H_dense = self.H.toarray()
            else:
                H_dense = self.H

            eigenvalues, eigenvectors = la.eigh(H_dense)

        # 确保按递增顺序排列
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors

        return eigenvalues, eigenvectors


def compute_energy_spectrum(N_values: List[int],
                           lambda_values: np.ndarray,
                           num_levels: int = 10,
                           use_sparse: bool = True,
                           verbose: bool = True) -> Dict:
    """
    计算不同 N 和 λ 值下的能谱

    Parameters:
    -----------
    N_values : List[int]
        自旋数量列表
    lambda_values : np.ndarray
        横向场强度列表
    num_levels : int
        保留的最低能级数量
    use_sparse : bool
        对于大 N 使用稀疏矩阵方法
    verbose : bool
        是否打印进度信息

    Returns:
    --------
    results : Dict
        包含所有 N 和 λ 的能谱数据
    """
    results = {}

    for N in N_values:
        if verbose:
            print(f"\n计算 N = {N} 的能谱...")
            print(f"  希尔伯特空间维度: 2^{N} = {2**N}")

        analyzer = IsingHamiltonianAnalyzer(N)
        energy_levels = []

        # 决定是否使用稀疏方法
        use_sparse_for_N = use_sparse and (N >= 10)

        start_time = time.time()

        for i, lam in enumerate(lambda_values):
            # 构造哈密顿量
            if use_sparse_for_N:
                H = analyzer.construct_hamiltonian_sparse_efficient(lam)
                analyzer.H = H
            else:
                H = analyzer.construct_hamiltonian_dense(lam)

            # 对角化
            if use_sparse_for_N:
                eigenvalues, _ = analyzer.diagonalize(use_sparse=True, k=num_levels)
            else:
                eigenvalues, _ = analyzer.diagonalize(use_sparse=False)
                eigenvalues = eigenvalues[:num_levels]  # 只保留最低的几个

            energy_levels.append(eigenvalues)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (len(lambda_values) - i - 1) * elapsed / (i + 1)
                print(f"  进度: {i+1}/{len(lambda_values)} "
                      f"({100*(i+1)/len(lambda_values):.1f}%) "
                      f"- 预计剩余时间: {remaining:.1f}秒")

        elapsed = time.time() - start_time
        if verbose:
            print(f"  完成! 用时: {elapsed:.2f}秒")

        results[N] = {
            'lambda_values': lambda_values,
            'energy_levels': np.array(energy_levels),
            'N': N,
            'num_levels': num_levels
        }

    return results


def plot_energy_spectrum(results: Dict,
                         N_values_to_plot: List[int] = None,
                         save_path: str = None):
    """
    绘制能级图

    Parameters:
    -----------
    results : Dict
        从 compute_energy_spectrum 返回的结果
    N_values_to_plot : List[int], optional
        要绘制的 N 值（如果为 None，绘制全部）
    save_path : str, optional
        保存图像的路径
    """
    if N_values_to_plot is None:
        N_values_to_plot = sorted(results.keys())

    n_plots = len(N_values_to_plot)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0, 0.9, 10))

    for idx, N in enumerate(N_values_to_plot):
        if N not in results:
            continue

        ax = axes[idx]
        data = results[N]
        lambda_values = data['lambda_values']
        energy_levels = data['energy_levels']

        # 绘制每个能级
        for level_idx in range(energy_levels.shape[1]):
            ax.plot(lambda_values, energy_levels[:, level_idx],
                   linewidth=1.5, color=colors[level_idx % len(colors)],
                   label=f'E_{level_idx}' if level_idx < 5 else '')

        # 标记量子相变点（λ ≈ 1）
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label='相变点 λ=1')

        ax.set_xlabel('横向场强度 λ', fontsize=12)
        ax.set_ylabel('能量 E', fontsize=12)
        ax.set_title(f'N = {N} (维度 = 2^{N} = {2**N})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if idx == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=9, loc='best', ncol=2)

    # 隐藏多余的子图
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图像已保存到: {save_path}")

    return fig


def plot_energy_gaps(results: Dict, save_path: str = None):
    """
    绘制能隙（第一激发态与基态的能量差）

    Parameters:
    -----------
    results : Dict
        能谱数据
    save_path : str, optional
        保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    N_values = sorted(results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_values)))

    for idx, N in enumerate(N_values):
        data = results[N]
        lambda_values = data['lambda_values']
        energy_levels = data['energy_levels']

        # 计算能隙
        energy_gap = energy_levels[:, 1] - energy_levels[:, 0]

        # 线性标度
        ax1.plot(lambda_values, energy_gap, linewidth=2,
                color=colors[idx], label=f'N = {N}', marker='o', markersize=3)

        # 对数标度
        ax2.semilogy(lambda_values, np.abs(energy_gap) + 1e-10, linewidth=2,
                    color=colors[idx], label=f'N = {N}', marker='o', markersize=3)

    # 标记相变点
    for ax in [ax1, ax2]:
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label='相变点 λ=1')
        ax.set_xlabel('横向场强度 λ', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    ax1.set_ylabel('能隙 Δ = E₁ - E₀', fontsize=12)
    ax1.set_title('(a) 能隙 vs λ (线性)', fontsize=13, fontweight='bold')

    ax2.set_ylabel('能隙 Δ = E₁ - E₀ (对数)', fontsize=12)
    ax2.set_title('(b) 能隙 vs λ (对数)', fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"能隙图已保存到: {save_path}")

    return fig


def analyze_phase_transition(results: Dict, lambda_critical: float = 1.0):
    """
    分析量子相变特性

    Parameters:
    -----------
    results : Dict
        能谱数据
    lambda_critical : float
        临界场强度
    """
    print("\n" + "="*70)
    print("量子相变分析")
    print("="*70)

    for N in sorted(results.keys()):
        data = results[N]
        lambda_values = data['lambda_values']
        energy_levels = data['energy_levels']

        # 找到最接近临界点的索引
        idx_critical = np.argmin(np.abs(lambda_values - lambda_critical))
        lambda_at_critical = lambda_values[idx_critical]

        # 计算临界点处的能隙
        gap_at_critical = energy_levels[idx_critical, 1] - energy_levels[idx_critical, 0]

        # 计算铁磁相（λ < 1）和顺磁相（λ > 1）的能隙
        idx_ferro = np.argmin(np.abs(lambda_values - 0.5))
        idx_para = np.argmin(np.abs(lambda_values - 1.5))

        gap_ferro = energy_levels[idx_ferro, 1] - energy_levels[idx_ferro, 0]
        gap_para = energy_levels[idx_para, 1] - energy_levels[idx_para, 0]

        print(f"\nN = {N}:")
        print(f"  临界点 λ ≈ {lambda_at_critical:.3f}:")
        print(f"    能隙: {gap_at_critical:.6f}")
        print(f"  铁磁相 λ = 0.5:")
        print(f"    能隙: {gap_ferro:.6f}")
        print(f"  顺磁相 λ = 1.5:")
        print(f"    能隙: {gap_para:.6f}")
        print(f"  基态能量:")
        print(f"    E₀(λ=0.5) = {energy_levels[idx_ferro, 0]:.6f}")
        print(f"    E₀(λ=1.0) = {energy_levels[idx_critical, 0]:.6f}")
        print(f"    E₀(λ=1.5) = {energy_levels[idx_para, 0]:.6f}")


if __name__ == "__main__":
    print("="*70)
    print("练习 3：Ising 哈密顿量")
    print("Exercise 3: Ising Hamiltonian")
    print("="*70)

    # 任务 a & b: 构造并对角化 Ising 哈密顿量
    print("\n任务 (a) & (b): 构造并对角化 Ising 哈密顿量")
    print("-"*70)

    # 测试小系统
    print("\n测试：N = 3 的小系统")
    test_analyzer = IsingHamiltonianAnalyzer(N=3)

    lambda_test = 1.0
    print(f"λ = {lambda_test}")

    print("\n使用稠密矩阵方法...")
    H_dense = test_analyzer.construct_hamiltonian_dense(lambda_test)
    print(f"哈密顿矩阵形状: {H_dense.shape}")
    print(f"矩阵是厄米的: {np.allclose(H_dense, H_dense.conj().T)}")

    eigenvalues_dense, _ = test_analyzer.diagonalize(use_sparse=False)
    print(f"\n前 5 个能级:")
    for i, E in enumerate(eigenvalues_dense[:5]):
        print(f"  E_{i} = {E:.6f}")

    # 任务 c: 绘制能级图
    print("\n任务 (c): 绘制不同 N 和 λ 的能级图")
    print("-"*70)

    # 参数设置
    N_values = [2, 3, 4, 5, 6]  # 自旋数量
    lambda_values = np.linspace(0.0, 3.0, 60)  # λ ∈ [0, 3]
    num_levels = 10  # 保留的能级数量

    print(f"\n参数设置:")
    print(f"  自旋数量 N: {N_values}")
    print(f"  横向场范围 λ: [{lambda_values[0]}, {lambda_values[-1]}]")
    print(f"  λ 采样点数: {len(lambda_values)}")
    print(f"  保留能级数: {num_levels}")

    # 计算能谱
    results = compute_energy_spectrum(
        N_values,
        lambda_values,
        num_levels=num_levels,
        use_sparse=True,
        verbose=True
    )

    # 绘制能级图
    print("\n生成能级图...")
    fig_spectrum = plot_energy_spectrum(
        results,
        N_values_to_plot=N_values,
        save_path='exercise3_energy_spectrum.png'
    )

    # 绘制能隙分析
    print("\n生成能隙分析图...")
    fig_gaps = plot_energy_gaps(
        results,
        save_path='exercise3_energy_gaps.png'
    )

    # 相变分析
    analyze_phase_transition(results, lambda_critical=1.0)

    print("\n练习 3 完成!")
    print("="*70)

    # 额外分析：谱的特性评论
    print("\n谱的特性分析:")
    print("-"*70)
    print("""
观察与评论：

1. **量子相变** (λ ≈ 1):
   - 在 λ = 1 附近，系统发生量子相变
   - λ < 1: 铁磁相，相邻自旋倾向于平行
   - λ > 1: 顺磁相，自旋沿 z 方向极化

2. **能级交叉/避免交叉**:
   - 在相变点附近可能观察到能级交叉或避免交叉
   - 这反映了系统对称性的变化

3. **能隙关闭**:
   - 在热力学极限 (N → ∞)，相变点处能隙关闭
   - 有限尺寸系统中，能隙始终非零但在相变点附近最小

4. **有限尺寸效应**:
   - 小 N: 离散能级，明显的能隙
   - 大 N: 能级变密，接近连续谱
   - 相变特征随 N 增大变得更加明显

5. **基态能量**:
   - λ = 0: E₀ = -N+1 (所有自旋平行)
   - λ → ∞: E₀ → -Nλ (所有自旋沿 z 向下)

6. **对称性**:
   - λ = 0: Z₂ 对称性（铁磁态简并）
   - λ > 0: 对称性破缺
    """)

    print("\n所有图表已生成。请查看:")
    print("  - exercise3_energy_spectrum.png: 能级图")
    print("  - exercise3_energy_gaps.png: 能隙分析")
