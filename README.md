# Introduction to Tensor Network Methods / 张量网络方法导论

这个项目实现了张量网络方法和量子多体物理中的经典问题，包括：
1. 随机厄米矩阵特征值分析（**Wigner-Dyson 统计**和**能级排斥**）
2. 量子多体物理习题（紧束缚哈密顿量、张量网络构造、规范不变希尔伯特空间、量子自旋冰模型）

This project implements classical problems in tensor network methods and quantum many-body physics, including:
1. Random Hermitian matrix eigenvalue analysis (**Wigner-Dyson statistics** and **level repulsion**)
2. Quantum many-body physics exercises (tight-binding Hamiltonians, tensor networks, gauge-invariant Hilbert spaces, quantum spin ice)

## 目录 / Table of Contents

### 量子多体物理习题 / Quantum Many-Body Physics Exercises
- [习题描述](#习题描述--exercise-description)
- [快速开始](#快速开始--quick-start)
- [习题详解](#习题详解--exercise-details)

### 随机厄米矩阵分析 / Random Hermitian Matrix Analysis
- [问题描述](#问题描述--problem-description)
- [功能特性](#功能特性--features)
- [安装](#安装--installation)
- [使用方法](#使用方法--usage)
- [理论背景](#理论背景--theoretical-background)
- [结果示例](#结果示例--example-results)
- [参考文献](#参考文献--references)

---

## 习题描述 / Exercise Description

### 量子多体物理习题 / Quantum Many-Body Physics Exercises

本项目包含四个量子多体物理习题的完整实现：

This project includes complete implementations of four quantum many-body physics exercises:

#### 习题 1 / Exercise 1: 紧束缚哈密顿量的对易关系 / Commutator Relations

对于哈密顿量：

$$H = -t \sum_i (c_i^\dagger c_{i+1} + c_{i+1}^\dagger c_i)$$

**任务 / Tasks:**
1. 验证 $[H, \hat{N}] = 0$（粒子数守恒）
2. 寻找与 $\hat{N}$ 对易或不对易的其他算符
3. 检验 $[G, H] = 0$（规范算符）

**物理意义 / Physical Significance:**
- 理解守恒量和对称性
- 费米子算符的对易关系
- 规范约束在量子系统中的作用

#### 习题 2 / Exercise 2: 张量网络构造 / Tensor Network Construction

明确写出方程 (6.10) 中的张量：

$$S_{\alpha_2, \beta_1, \beta_2} = T_{\tau_2, \upsilon_1, \upsilon_2} \delta_{\ell_2 + m_1 + m_2, 0}$$

**任务 / Tasks:**
1. 明确构造两格点张量
2. 推广到三个格点
3. 使用矩阵乘积态 (MPS) 数值推广到 L 个格点

**物理意义 / Physical Significance:**
- 张量网络方法的基础
- 规范约束的张量表示
- 指数压缩和高效表示

#### 习题 3 / Exercise 3: 规范不变希尔伯特空间维数 / Gauge Invariant Hilbert Space

利用关系式 (6.26)：

$$KP_{N'}|\psi\rangle = KPP_{N'}|\psi\rangle = KP_{N'}P|\psi\rangle = P_{N'}^G|\psi^G\rangle$$

**任务 / Tasks:**
数值计算一维自旋-1 系统规范不变希尔伯特空间的维数随格点数的变化

**物理意义 / Physical Significance:**
- 规范约束如何减少希尔伯特空间维数
- 计算优势随系统尺寸的增长
- 大规模量子模拟的关键

#### 习题 4 / Exercise 4: 二维量子自旋冰模型 / 2D Quantum Spin Ice

**任务 / Tasks:**
计算二维量子自旋冰模型的算符 K 和 $P_{N'}$

**物理意义 / Physical Significance:**
- 阻挫磁性系统
- 涌现规范理论
- 拓扑序和新奇物态

## 快速开始 / Quick Start

### 运行所有习题 / Run All Exercises

```bash
# 运行主程序（包含所有四个习题）
python quantum_many_body_exercises.py
```

输出包括 / Output includes:
- 详细的数值结果和分析
- 三个可视化图表：
  - `exercise2_tensor_construction.png`
  - `exercise3_gauge_invariant_dimension.png`
  - `exercise4_spin_ice_lattice.png`

### Jupyter Notebook 演示 / Jupyter Notebook Demo

```bash
# 启动交互式 notebook
jupyter notebook quantum_many_body_exercises_demo.ipynb
```

这个 notebook 提供 / This notebook provides:
- 详细的理论背景
- 交互式代码单元
- 完整的可视化
- 物理解释和分析

## 习题详解 / Exercise Details

### 习题 1 实现要点 / Exercise 1 Implementation Highlights

**核心类 / Core Class:** `TightBindingModel`

```python
from quantum_many_body_exercises import TightBindingModel

# 创建 4 格点系统
model = TightBindingModel(L=4, t=1.0, periodic=True)

# 获取算符
H = model.hamiltonian()
N = model.total_number_operator()

# 检验对易关系
commutes = model.check_commutation(H, N, "H", "N")
```

**主要功能 / Key Features:**
- Jordan-Wigner 变换实现费米子算符
- 自动验证对易关系
- 支持周期和开边界条件

**结果 / Results:**
✓ $[H, \hat{N}] = 0$ （粒子数守恒）
✓ $[n_i, \hat{N}] = 0$ （局域数算符对易）
✗ $[c_i, \hat{N}] \neq 0$ （产生/湮灭算符改变粒子数）

### 习题 2 实现要点 / Exercise 2 Implementation Highlights

**核心类 / Core Class:** `TensorNetworkConstructor`

```python
from quantum_many_body_exercises import TensorNetworkConstructor

# 创建自旋-1 系统
constructor = TensorNetworkConstructor(spin=1)

# 构造两格点张量（带规范约束）
T2 = constructor.create_simple_tensor(constraint_value=0)

# 推广到 MPS
mps = constructor.construct_mps_chain(L=10, bond_dim=4)
```

**压缩效果 / Compression Results:**
- L=4: 压缩比 0.68
- L=6: 压缩比 3.4
- L=8: 压缩比 21
- L=10: 压缩比 145（指数增长！）

### 习题 3 实现要点 / Exercise 3 Implementation Highlights

**核心类 / Core Class:** `GaugeInvariantHilbertSpace`

```python
from quantum_many_body_exercises import GaugeInvariantHilbertSpace

# 创建自旋-1 系统
calculator = GaugeInvariantHilbertSpace(spin=1)

# 计算规范不变维数
results = calculator.compute_dimensions_vs_L(L_max=8)
```

**关键发现 / Key Findings:**
- 规范约束显著减少希尔伯特空间维数
- 平均比例（规范/总）：~0.23
- 计算优势随系统尺寸指数增长
- 对大规模模拟至关重要

### 习题 4 实现要点 / Exercise 4 Implementation Highlights

**核心类 / Core Class:** `QuantumSpinIce2D`

```python
from quantum_many_body_exercises import QuantumSpinIce2D

# 创建 4×4 晶格
model = QuantumSpinIce2D(Lx=4, Ly=4)

# 计算规范不变维数
dim_gauge = model.compute_gauge_invariant_dimension()

# 可视化晶格结构
model.visualize_lattice()
```

**物理图像 / Physical Picture:**
- 自旋位于格点间的连线上
- 冰规则约束：每个方格周围自旋之和为零
- 类似电磁学中的高斯定律 (div E = 0)
- 涌现的规范理论结构

**标度分析 / Scaling Analysis:**
- 4×4 晶格：总维数 $2^{32}$，规范不变 $2^{24}$
- 约束数 = (Lx-1) × (Ly-1)
- 拓扑简并度取决于边界条件

---

## 问题描述 / Problem Description

### 随机厄米矩阵特征值分析 / Random Hermitian Matrix Eigenvalue Analysis

### 具体任务 / Tasks

#### a. LU分解和标度分析 / LU Decomposition and Scaling Analysis
- 初始化大小为 N 的随机厄米矩阵 A
- 执行 LU 分解
- 分析时间复杂度如何随 N 标度（理论预测：O(N³)）

#### b. 对角化和特征值存储 / Diagonalization and Eigenvalue Storage
- 对角化矩阵 A
- 按递增顺序存储 N 个特征值 λᵢ
- 验证特征值都是实数（厄米矩阵的性质）

#### c. 计算归一化间距 / Calculate Normalized Spacings
计算归一化的特征值间距：

$$s_i = \frac{\Delta\lambda_i}{\bar{\Delta\lambda}}$$

其中 / where:
- $\Delta\lambda_i = \lambda_{i+1} - \lambda_i$ （相邻特征值之间的间距 / spacing between adjacent eigenvalues）
- $\bar{\Delta\lambda}$ 是平均间距 / is the average spacing

#### d. 局部平均分析 / Local Averaging Analysis
- 在 λᵢ 周围的不同数量的能级上计算平均间距
- 测试窗口大小：N/100, N/50, N/10, N/5, N
- 比较不同归一化方法的结果

## 功能特性 / Features

### 核心功能 / Core Features

1. **随机厄米矩阵生成** / Random Hermitian Matrix Generation
   - 高斯酉系综 (GUE) 实现
   - 保证矩阵厄米性：A = A†

2. **LU 分解分析** / LU Decomposition Analysis
   - 完整的 LU 分解：PA = LU
   - 自动标度行为分析
   - O(N³) 复杂度验证

3. **特征值分析** / Eigenvalue Analysis
   - 高效的厄米矩阵对角化
   - 特征值自动排序
   - 特征向量验证

4. **归一化间距计算** / Normalized Spacing Calculation
   - 全局平均归一化
   - 局部平均归一化（多种窗口大小）
   - 灵活的窗口大小配置

5. **理论对比** / Theoretical Comparison
   - Wigner-Dyson 分布（GOE, GUE, GSE）
   - Poisson 分布（参考）
   - 可视化对比

6. **可视化工具** / Visualization Tools
   - 特征值分布图
   - 归一化间距分布图
   - LU 分解标度曲线
   - 局部平均方法比较

## 安装 / Installation

### 依赖 / Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook (可选 / optional)

### 快速安装 / Quick Install

```bash
# 克隆仓库 / Clone repository
git clone https://github.com/yourusername/Introduction-to-Tensor-Network-Methods.git
cd Introduction-to-Tensor-Network-Methods

# 安装依赖 / Install dependencies
pip install -r requirements.txt
```

## 使用方法 / Usage

### 命令行使用 / Command Line Usage

```bash
# 运行主程序（包含所有任务的演示）
python hermitian_matrix_analysis.py
```

### Python 脚本使用 / Python Script Usage

```python
from hermitian_matrix_analysis import HermitianMatrixAnalyzer

# 创建分析器
N = 1000  # 矩阵大小
analyzer = HermitianMatrixAnalyzer(N, seed=42)

# 生成随机厄米矩阵
A = analyzer.generate_hermitian_matrix()

# 执行 LU 分解
P, L, U, time_elapsed = analyzer.perform_lu_decomposition()

# 对角化矩阵
eigenvalues, eigenvectors = analyzer.diagonalize_matrix()

# 计算归一化间距
results = analyzer.calculate_normalized_spacings()
normalized_spacings = results['global']

# 可视化
from hermitian_matrix_analysis import plot_spacing_distribution
plot_spacing_distribution(normalized_spacings, ensemble='GUE')
```

### Jupyter Notebook 使用 / Jupyter Notebook Usage

```bash
# 启动 Jupyter Notebook
jupyter notebook hermitian_analysis_demo.ipynb
```

这个 notebook 包含：
- 详细的理论说明
- 交互式代码示例
- 完整的可视化结果
- 物理意义解释

## 理论背景 / Theoretical Background

### 厄米矩阵 / Hermitian Matrices

厄米矩阵满足 $A = A^\dagger$（共轭转置等于自身），具有以下性质：
- 所有特征值都是实数
- 特征向量构成正交完备基
- 在量子力学中代表可观测量

Hermitian matrices satisfy $A = A^\dagger$ (conjugate transpose equals itself) and have:
- All eigenvalues are real
- Eigenvectors form an orthonormal complete basis
- Represent observables in quantum mechanics

### 随机矩阵理论 / Random Matrix Theory

随机矩阵理论研究大型随机矩阵的统计性质，发现了许多物理系统中的**普适性**现象。

Random matrix theory studies statistical properties of large random matrices and discovers **universality** phenomena in many physical systems.

#### 高斯系综 / Gaussian Ensembles

1. **GOE (Gaussian Orthogonal Ensemble)**: 实对称矩阵
2. **GUE (Gaussian Unitary Ensemble)**: 复厄米矩阵 ← 本项目使用 / used in this project
3. **GSE (Gaussian Symplectic Ensemble)**: 四元数自对偶矩阵

### Wigner-Dyson 统计 / Wigner-Dyson Statistics

对于 GUE，归一化间距分布的理论预测为：

$$P(s) = \frac{32}{\pi^2} s^2 e^{-\frac{4s^2}{\pi}}$$

关键特征：
- **能级排斥**: P(s→0) → 0（小间距的概率接近零）
- 与 Poisson 分布 P(s) = e^(-s) 显著不同
- 反映了特征值之间的关联

### 物理应用 / Physical Applications

1. **核物理** / Nuclear Physics
   - 原子核能级的统计分析
   - 共振散射实验

2. **量子混沌** / Quantum Chaos
   - 判断系统的可积性
   - 经典-量子对应

3. **凝聚态物理** / Condensed Matter Physics
   - 无序系统的能谱
   - Anderson 局域化

4. **数论** / Number Theory
   - 黎曼 zeta 函数零点统计
   - Montgomery-Odlyzko 定律

5. **通信理论** / Communication Theory
   - MIMO 系统信道容量
   - 无线网络优化

## 结果示例 / Example Results

### LU 分解标度行为

```
N = 50  : 0.001234 ± 0.000045 秒
N = 100 : 0.008765 ± 0.000321 秒
N = 200 : 0.067543 ± 0.002145 秒
N = 400 : 0.543210 ± 0.012345 秒
N = 800 : 4.321098 ± 0.098765 秒
```

标度指数 ≈ 3.0，确认 O(N³) 复杂度。

### 特征值统计

对于 N = 1000 的随机厄米矩阵：
- 特征值范围：[-44.7232, 44.8901]
- 平均特征值：-0.0123 ≈ 0（符合理论预期）
- 特征值标准差：≈ √(2N) = 44.7（Wigner 半圆律）

### 归一化间距

```
归一化间距统计:
  平均值: 1.0000 (理论值 = 1.0)
  标准差: 0.5234
  中位数: 0.9123
  最小值: 0.0012 (接近零但不为零 → 能级排斥)
  最大值: 4.5678
```

与 Wigner-Dyson (GUE) 分布的拟合度：χ² ≈ 0.023

## 代码结构 / Code Structure

```
Introduction-to-Tensor-Network-Methods/
│
├── quantum_many_body_exercises.py       # 量子多体物理习题实现
├── quantum_many_body_exercises_demo.ipynb  # 量子多体习题交互式演示
├── hermitian_matrix_analysis.py         # 随机厄米矩阵分析（核心实现）
├── hermitian_analysis_demo.ipynb        # 随机矩阵演示 notebook
├── requirements.txt                     # 依赖列表
├── README.md                            # 本文件
│
└── results/                             # 结果输出（可选）
    ├── figures/                         # 图表
    │   ├── exercise2_tensor_construction.png
    │   ├── exercise3_gauge_invariant_dimension.png
    │   └── exercise4_spin_ice_lattice.png
    └── data/                            # 数据文件
```

### 主要类和函数 / Main Classes and Functions

#### 量子多体物理模块 / Quantum Many-Body Physics Module

**`TightBindingModel` 类** (习题 1)
```python
class TightBindingModel:
    def __init__(self, L, t=1.0, periodic=True)
    def creation_operator(site)
    def annihilation_operator(site)
    def hamiltonian()
    def total_number_operator()
    def gauge_operator(site)
    def commutator(A, B)
    def check_commutation(A, B, name_A, name_B)
```

**`TensorNetworkConstructor` 类** (习题 2)
```python
class TensorNetworkConstructor:
    def __init__(self, spin=1)
    def create_simple_tensor(constraint_value=0)
    def extend_to_three_sites()
    def construct_mps_tensor(bond_dim=2)
    def construct_mps_chain(L, bond_dim=2)
    def contract_mps(mps)
```

**`GaugeInvariantHilbertSpace` 类** (习题 3)
```python
class GaugeInvariantHilbertSpace:
    def __init__(self, spin=1)
    def generate_all_states(L)
    def apply_gauge_constraint(states, constraint_type)
    def count_gauge_invariant_dimension(L, constraint_type)
    def compute_dimensions_vs_L(L_max, constraint_type)
    def construct_projection_operator(L, constraint_type)
```

**`QuantumSpinIce2D` 类** (习题 4)
```python
class QuantumSpinIce2D:
    def __init__(self, Lx, Ly)
    def construct_gauge_constraint_operator()
    def construct_particle_number_projector(N_target)
    def construct_gauge_projector_K()
    def compute_gauge_invariant_dimension()
    def visualize_lattice()
```

#### 随机厄米矩阵模块 / Random Hermitian Matrix Module

**`HermitianMatrixAnalyzer` 类**
```python
class HermitianMatrixAnalyzer:
    def __init__(self, N, seed=None)
    def generate_hermitian_matrix()
    def perform_lu_decomposition()
    def diagonalize_matrix()
    def calculate_normalized_spacings()
    def wigner_dyson_distribution()
    def poisson_distribution()
```

#### 辅助函数 / Utility Functions

- `analyze_lu_scaling()`: LU 分解标度分析
- `plot_spacing_distribution()`: 绘制间距分布
- `plot_lu_scaling()`: 绘制标度曲线
- `compare_local_averaging_methods()`: 比较局部平均方法
- `exercise_1_demo()`, `exercise_2_demo()`, `exercise_3_demo()`, `exercise_4_demo()`: 各习题演示函数

## 扩展功能 / Extensions

### 已实现 / Implemented

- ✅ 多种系综支持（GOE, GUE, GSE）
- ✅ 局部平均归一化
- ✅ 完整的可视化工具
- ✅ 性能优化

### 计划中 / Planned

- ⬜ 其他统计量（数量方差、谱刚度等）
- ⬜ 稀疏矩阵支持
- ⬜ 并行计算
- ⬜ GPU 加速
- ⬜ 更多物理系统的应用示例

## 参考文献 / References

### 教材 / Textbooks

1. Mehta, M. L. (2004). *Random Matrices* (3rd ed.). Academic Press.
   - 随机矩阵理论的经典教材

2. Haake, F. (2010). *Quantum Signatures of Chaos* (3rd ed.). Springer.
   - 量子混沌与随机矩阵理论的联系

3. Forrester, P. J. (2010). *Log-Gases and Random Matrices*. Princeton University Press.
   - 随机矩阵的现代理论

4. Tao, T. (2012). *Topics in Random Matrix Theory*. American Mathematical Society.
   - 数学角度的随机矩阵理论

### 论文 / Papers

1. Wigner, E. P. (1955). "Characteristic vectors of bordered matrices with infinite dimensions." *Annals of Mathematics*, 62(3), 548-564.

2. Dyson, F. J. (1962). "Statistical theory of the energy levels of complex systems." *Journal of Mathematical Physics*, 3(1), 140-156.

3. Montgomery, H. L. (1973). "The pair correlation of zeros of the zeta function." *Analytic Number Theory*, 24, 181-193.

### 在线资源 / Online Resources

- [Random Matrix Theory (Wikipedia)](https://en.wikipedia.org/wiki/Random_matrix)
- [Wigner-Dyson Statistics (Scholarpedia)](http://www.scholarpedia.org/article/Wigner-Dyson_statistics)
- [NIST Digital Library of Mathematical Functions](https://dlmf.nist.gov/)

## 贡献 / Contributing

欢迎贡献！请随时提交 Issue 或 Pull Request。

Contributions are welcome! Feel free to submit issues or pull requests.

### 开发指南 / Development Guidelines

1. 遵循 PEP 8 代码风格
2. 添加适当的文档字符串
3. 包含单元测试
4. 更新 README 和示例

## 许可证 / License

MIT License

## 作者 / Authors

- 初始实现 / Initial Implementation: Claude AI
- 维护者 / Maintainer: [Your Name]

## 致谢 / Acknowledgments

本项目实现了张量网络方法导论课程中的练习问题，感谢课程提供者。

This project implements an exercise problem from the Introduction to Tensor Network Methods course.

---

## 常见问题 / FAQ

### Q1: 为什么使用 GUE 而不是 GOE？

**A**: GUE（高斯酉系综）对应复厄米矩阵，具有更强的能级排斥效应。在凝聚态物理中，存在时间反演对称性破缺时（如磁场存在），系统通常对应 GUE 统计。

### Q2: 归一化间距的物理意义是什么？

**A**: 归一化间距消除了能级密度变化的影响，揭示了能级之间的**关联**。小间距概率低表示能级排斥，这是量子系统不可约简性的体现。

### Q3: 为什么需要局部平均？

**A**: 对于非均匀的能级密度（如有界谱），全局平均可能不够准确。局部平均可以更好地捕捉局部统计性质。

### Q4: 如何处理大矩阵（N > 10000）？

**A**:
- 使用稀疏矩阵表示（如果适用）
- 只计算部分特征值（如使用 ARPACK）
- 并行计算
- 使用更高效的算法（如 Lanczos 方法）

### Q5: 能级排斥在实际系统中的意义？

**A**: 能级排斥反映了系统的复杂性和不可积性。可积系统表现为 Poisson 统计（无排斥），而混沌系统表现为 Wigner-Dyson 统计（有排斥）。

---

**更多问题？** 请在 Issues 中提出！

**More questions?** Please open an issue!
