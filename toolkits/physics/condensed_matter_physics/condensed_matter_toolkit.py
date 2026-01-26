# Filename: condensed_matter_toolkit.py
# 统一的凝聚态物理计算工具包
# 合并自 condensed_matter_tools_4179.py 和 condensed_matter_tools_17363.py

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Tuple, Optional, Union, Dict, List, Callable

# 确保图片保存目录存在
if not os.path.exists("./images"):
    os.makedirs("./images")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 量子系统模块 (Quantum Systems Module)
# ============================================================================

def construct_hamiltonian(system_size, interaction_matrix, potential=None, periodic=False):
    """
    构建量子多体系统的哈密顿量矩阵（稀疏格式）
    
    该函数用于构建一维或多维量子系统的哈密顿量，支持自定义相互作用和势能项。
    基于紧束缚模型(tight-binding model)的理论框架。
    
    Parameters:
    -----------
    system_size : tuple of int or int
        系统尺寸，如(Nx, Ny)表示二维晶格，(N,)或N表示一维链
    interaction_matrix : list or array-like
        相互作用矩阵，描述粒子间的相互作用强度
    potential : list, array-like or callable, optional
        外部势能场，可以是数组或函数
    periodic : bool, optional
        是否使用周期性边界条件，默认为False
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        系统哈密顿量的稀疏矩阵表示
    """
    # 转换为numpy数组（如果不是callable）并做健壮性检查
    if potential is not None and not callable(potential):
        potential = np.asarray(potential)
    if not callable(interaction_matrix):
        # 明确拒绝字符串/0维字符串数组
        if isinstance(interaction_matrix, (str, bytes)):
            raise ValueError("interaction_matrix 不应为字符串，请传数值/数值数组/2x2矩阵")
        interaction_matrix = np.asarray(interaction_matrix)
        if interaction_matrix.dtype.kind in ('U', 'S') and interaction_matrix.ndim == 0:
            raise ValueError("interaction_matrix 解析为0维字符串数组，请传数值/数值数组/2x2矩阵")
    
    # 统一 system_size 表示为元组，便于处理一维/二维
    if isinstance(system_size, tuple):
        size_tuple = tuple(int(x) for x in system_size)
    elif isinstance(system_size, list) or isinstance(system_size, np.ndarray):
        flat = list(system_size)
        if len(flat) == 1:
            size_tuple = (int(flat[0]),)
        elif len(flat) == 2:
            size_tuple = (int(flat[0]), int(flat[1]))
        else:
            raise ValueError("system_size 列表长度必须为1或2")
    else:
        size_tuple = (int(system_size),)

    # 计算系统总格点数
    total_sites = int(np.prod(size_tuple))
    
    # 初始化稀疏矩阵
    H = sp.lil_matrix((total_sites, total_sites), dtype=np.float64)
    
    # 构建相互作用项
    if len(size_tuple) == 1:  # 一维系统 (N,)
        for i in range(total_sites):
            # 对角项（on-site能量）
            if potential is not None:
                if callable(potential):
                    H[i, i] = potential(i)
                else:
                    H[i, i] = potential[i]
            
            # 最近邻相互作用
            if i < total_sites - 1:  # 右邻居
                # 支持三种格式：标量、数值数组（取第一项）、callable（已在外部排除）、否则报错
                if callable(interaction_matrix):
                    hop = float(interaction_matrix(i, i+1))
                elif np.isscalar(interaction_matrix):
                    hop = float(interaction_matrix)
                else:
                    # 数值数组：允许长度>=1，取第一个元素
                    hop = float(np.asarray(interaction_matrix).ravel()[0])
                H[i, i+1] = hop
                H[i+1, i] = hop  # 厄米共轭
            
            # 周期性边界条件
            if periodic and i == total_sites - 1:
                if callable(interaction_matrix):
                    hop = float(interaction_matrix(i, 0))
                elif np.isscalar(interaction_matrix):
                    hop = float(interaction_matrix)
                else:
                    hop = float(np.asarray(interaction_matrix).ravel()[0])
                H[i, 0] = hop
                H[0, i] = hop
    
    elif len(size_tuple) == 2:  # 二维系统 (Nx, Ny)
        nx, ny = size_tuple
        # interaction_matrix 期望为 2x2 数值矩阵 [[t_xx, t_xy],[t_yx, t_yy]]
        if callable(interaction_matrix):
            # 用户自定义：期望形如 f((ix,iy),(jx,jy)) 返回浮点
            get_hop = interaction_matrix
        else:
            im = np.asarray(interaction_matrix)
            if im.shape != (2, 2):
                raise ValueError("二维 system_size 需要 2x2 的 interaction_matrix，例如 [[t_xx, t_xy],[t_yx, t_yy]]")
            t_x = float(im[0, 0])  # x方向近邻
            t_y = float(im[0, 1])  # y方向近邻
        for ix in range(nx):
            for iy in range(ny):
                i = ix * ny + iy  # 线性索引
                
                # 对角项
                if potential is not None:
                    if callable(potential):
                        H[i, i] = potential(ix, iy)
                    else:
                        H[i, i] = potential[ix, iy]
                
                # x方向相互作用
                if ix < nx - 1:  # 右邻居
                    j = (ix + 1) * ny + iy
                    val = get_hop((ix, iy), (ix+1, iy)) if callable(interaction_matrix) else t_x
                    H[i, j] = val
                    H[j, i] = val
                elif periodic:  # 周期性边界
                    j = 0 * ny + iy
                    val = get_hop((ix, iy), (0, iy)) if callable(interaction_matrix) else t_x
                    H[i, j] = val
                    H[j, i] = val
                
                # y方向相互作用
                if iy < ny - 1:  # 上邻居
                    j = ix * ny + (iy + 1)
                    val = get_hop((ix, iy), (ix, iy+1)) if callable(interaction_matrix) else t_y
                    H[i, j] = val
                    H[j, i] = val
                elif periodic:  # 周期性边界
                    j = ix * ny + 0
                    val = get_hop((ix, iy), (ix, 0)) if callable(interaction_matrix) else t_y
                    H[i, j] = val
                    H[j, i] = val
    
    return H.tocsr()  # 转换为CSR格式以提高计算效率


def _eigsolve_from_matrix(hamiltonian, k=6, which='SA'):
    """
    内部工具：给定矩阵，求解本征值/本征向量并排序。
    """
    if sp.issparse(hamiltonian):
        if not np.allclose((hamiltonian - hamiltonian.getH()).data, 0):
            print("警告：哈密顿量不是厄米矩阵，结果可能不物理")
        eigenvalues, eigenvectors = spla.eigsh(hamiltonian, k=k, which=which)
    else:
        if not np.allclose(hamiltonian, hamiltonian.conj().T):
            print("警告：哈密顿量不是厄米矩阵，结果可能不物理")
        eigenvalues, eigenvectors = la.eigh(hamiltonian)
        if k < len(eigenvalues):
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]

    idx = eigenvalues.argsort()
    return eigenvalues[idx], eigenvectors[:, idx]


def solve_eigensystem(
    system_size,
    interaction_matrix,
    potential=None,
    periodic=False,
    *,
    k=6,
    which='SA',
):
    """
    求解哈密顿量的本征值和本征态（无入参 hamiltonian 版本）。

    本函数将使用传入的构造参数先调用 construct_hamiltonian，随后对所得矩阵进行本征求解。

    Parameters:
    -----------
    system_size : int | tuple[int, int]
        系统尺寸（见 construct_hamiltonian）。
    interaction_matrix : number | array-like | 2x2 array
        相互作用参数（见 construct_hamiltonian）。
    potential : array-like or callable, optional
        外部势能。
    periodic : bool, optional
        周期性边界条件，默认 False。
    k : int, optional
        求解的本征值数量，默认 6。
    which : str, optional
        'SA'（默认）、'LA'、'SM'、'LM'。

    Returns:
    --------
    eigenvalues : ndarray
        本征值（升序）。
    eigenvectors : ndarray
        本征向量（列为态）。
    """
    H = construct_hamiltonian(system_size, interaction_matrix, potential=potential, periodic=periodic)
    return _eigsolve_from_matrix(H, k=k, which=which)


def solve_eigensystem_from_matrix(hamiltonian, k=6, which='SA'):
    """
    直接从已给定哈密顿量矩阵求解（为内部/向后兼容保留）。
    """
    return _eigsolve_from_matrix(hamiltonian, k=k, which=which)


def simulate_quantum_system(hamiltonian, num_states=5):
    """
    模拟量子多体系统，求解哈密顿量的本征值和本征态（密集矩阵方法）
    
    适用于小规模量子系统的精确对角化。
    
    Parameters:
    -----------
    hamiltonian : numpy.ndarray
        系统的哈密顿量矩阵
    num_states : int, optional
        需要计算的能级数量，默认为5
    
    Returns:
    --------
    tuple
        (eigenvalues, eigenvectors) 本征值和对应的本征态
    """
    # 确保哈密顿量是厄米矩阵
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("哈密顿量必须是厄米矩阵")
    
    # 求解本征值问题
    eigenvalues, eigenvectors = la.eigh(hamiltonian)
    
    # 返回最低的num_states个能级和对应的本征态
    return eigenvalues[:num_states], eigenvectors[:, :num_states]


def calculate_density_matrix(eigenvector, trace_subsystem=None):
    """
    计算量子态的密度矩阵或约化密度矩阵
    
    可用于计算纯态的密度矩阵，或通过部分迹运算得到子系统的约化密度矩阵，
    是量子纠缠和量子信息研究的基础工具。
    
    Parameters:
    -----------
    eigenvector : list or array-like
        量子态的向量表示
    trace_subsystem : tuple, optional
        需要追踪的子系统维度，如果为None则计算完整密度矩阵
        
    Returns:
    --------
    density_matrix : ndarray
        密度矩阵或约化密度矩阵
    """
    # 转换为numpy数组
    eigenvector = np.asarray(eigenvector)
    
    # 确保eigenvector是列向量
    if eigenvector.ndim == 1:
        eigenvector = eigenvector.reshape(-1, 1)
    
    # 计算完整密度矩阵 ρ = |ψ⟩⟨ψ|
    full_dm = np.outer(eigenvector, eigenvector.conj())
    
    # 如果不需要部分迹，直接返回完整密度矩阵
    if trace_subsystem is None:
        return full_dm
    
    # 计算约化密度矩阵（通过部分迹）
    # 这里实现一个简化版本，假设系统可以分为两个子系统A和B
    dim_a, dim_b = trace_subsystem
    reduced_dm = np.zeros((dim_a, dim_a), dtype=complex)
    
    # 执行部分迹运算
    for i in range(dim_b):
        # 提取对应于B系统第i个基矢的子块
        block = full_dm[i::dim_b, i::dim_b]
        reduced_dm += block
    
    return reduced_dm


# ============================================================================
# 统计物理模块 (Statistical Physics Module)
# ============================================================================

def monte_carlo_ising(lattice_size, temperature, num_steps, J=1.0):
    """
    使用Metropolis算法进行二维Ising模型的蒙特卡洛模拟
    
    实现经典的Metropolis算法模拟二维Ising模型的热力学行为，
    可用于研究相变、临界现象和标度行为。
    
    Parameters:
    -----------
    lattice_size : tuple of int
        晶格尺寸，如(L, L)表示LxL的二维方格
    temperature : float
        系统温度，单位为J/kB
    num_steps : int
        蒙特卡洛步数
    J : float, optional
        自旋间相互作用强度，J>0表示铁磁性，默认为1.0
        
    Returns:
    --------
    dict
        包含模拟结果的字典，包括：
        - 'lattice': 最终晶格构型
        - 'energy': 能量时间序列
        - 'magnetization': 磁化强度时间序列
        - 'specific_heat': 比热
        - 'susceptibility': 磁化率
    """
    # 初始化随机晶格构型
    lattice = np.random.choice([-1, 1], size=lattice_size)
    
    # 预分配结果数组
    energy_samples = np.zeros(num_steps)
    mag_samples = np.zeros(num_steps)
    
    # 计算初始能量
    energy = 0.0
    for i in range(lattice_size[0]):
        for j in range(lattice_size[1]):
            spin = lattice[i, j]
            # 考虑最近邻相互作用（周期性边界条件）
            neighbors = (
                lattice[(i+1)%lattice_size[0], j] + 
                lattice[i, (j+1)%lattice_size[1]] + 
                lattice[(i-1)%lattice_size[0], j] + 
                lattice[i, (j-1)%lattice_size[1]]
            )
            energy += -J * spin * neighbors
    # 避免重复计算，除以2
    energy /= 2.0
    
    # 计算初始磁化强度
    magnetization = np.sum(lattice)
    
    # Metropolis算法主循环
    for step in range(num_steps):
        # 随机选择一个格点
        i, j = np.random.randint(0, lattice_size[0]), np.random.randint(0, lattice_size[1])
        spin = lattice[i, j]
        
        # 计算翻转自旋导致的能量变化
        neighbors = (
            lattice[(i+1)%lattice_size[0], j] + 
            lattice[i, (j+1)%lattice_size[1]] + 
            lattice[(i-1)%lattice_size[0], j] + 
            lattice[i, (j-1)%lattice_size[1]]
        )
        delta_E = 2 * J * spin * neighbors
        
        # Metropolis接受/拒绝准则
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / temperature):
            # 接受翻转
            lattice[i, j] = -spin
            energy += delta_E
            magnetization -= 2 * spin
        
        # 记录样本
        energy_samples[step] = energy
        mag_samples[step] = magnetization
    
    # 计算热力学量
    # 丢弃前20%的样本作为热化阶段
    discard = int(0.2 * num_steps)
    energy_samples = energy_samples[discard:]
    mag_samples = mag_samples[discard:]
    
    # 计算平均能量、磁化强度、比热和磁化率
    N = np.prod(lattice_size)  # 总格点数
    avg_energy = np.mean(energy_samples) / N
    avg_mag = np.mean(np.abs(mag_samples)) / N  # 使用绝对值避免不同域的抵消
    
    # 比热: C = (⟨E²⟩ - ⟨E⟩²) / (kB*T²*N)
    specific_heat = (np.mean(energy_samples**2) - np.mean(energy_samples)**2) / (temperature**2 * N)
    
    # 磁化率: χ = (⟨M²⟩ - ⟨M⟩²) / (kB*T*N)
    susceptibility = (np.mean(mag_samples**2) - np.mean(mag_samples)**2) / (temperature * N)
    
    return {
        'lattice': lattice,
        'energy': energy_samples,
        'magnetization': mag_samples,
        'avg_energy': avg_energy,
        'avg_magnetization': avg_mag,
        'specific_heat': specific_heat,
        'susceptibility': susceptibility
    }


def monte_carlo_ising_step(lattice, beta, J=1.0):
    """
    执行一步蒙特卡洛模拟（Metropolis算法）用于伊辛模型
    
    Parameters:
    -----------
    lattice : list or array-like
        表示伊辛模型的晶格，元素为+1或-1
    beta : float
        逆温度参数 β = 1/(k_B*T)
    J : float, optional
        自旋间相互作用强度，默认为1.0（铁磁性）
    
    Returns:
    --------
    numpy.ndarray
        更新后的晶格
    """
    # 转换为numpy数组（确保可变性）
    lattice = np.asarray(lattice, dtype=np.int8).copy()
    
    L = lattice.shape[0]
    
    # 随机选择一个格点
    i, j = np.random.randint(0, L, size=2)
    
    # 计算翻转自旋前后的能量变化
    s = lattice[i, j]
    neighbors = lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + lattice[i, (j+1)%L] + lattice[i, (j-1)%L]
    delta_E = 2 * J * s * neighbors
    
    # Metropolis准则
    if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
        lattice[i, j] = -s
    
    return lattice


def simulate_ising_model(L=50, T=2.27, steps=10000, equilibration=1000, J=1.0):
    """
    模拟二维伊辛模型的相变
    
    Parameters:
    -----------
    L : int, optional
        晶格大小，默认为50
    T : float, optional
        温度（单位为J/k_B），默认为2.27（接近临界温度）
    steps : int, optional
        模拟步数，默认为10000
    equilibration : int, optional
        平衡步数，默认为1000
    J : float, optional
        自旋间相互作用强度，默认为1.0
    
    Returns:
    --------
    tuple
        (lattice, magnetizations, energies) 最终晶格状态、磁化强度历史和能量历史
    """
    # 初始化随机晶格
    lattice = np.random.choice([-1, 1], size=(L, L))
    
    # 计算逆温度
    beta = 1.0 / T
    
    # 记录磁化强度和能量
    magnetizations = []
    energies = []
    
    # 模拟主循环
    for step in range(steps + equilibration):
        # 执行L^2次自旋翻转尝试（一个蒙特卡洛步）
        for _ in range(L*L):
            lattice = monte_carlo_ising_step(lattice, beta, J)
        
        # 平衡后开始记录
        if step >= equilibration:
            # 计算磁化强度
            m = np.abs(np.mean(lattice))
            magnetizations.append(m)
            
            # 计算能量
            e = 0
            for i in range(L):
                for j in range(L):
                    e -= J * lattice[i, j] * (lattice[(i+1)%L, j] + lattice[i, (j+1)%L])
            energies.append(e / (L*L))
    
    return lattice, np.array(magnetizations), np.array(energies)


def calculate_correlation_function(lattice, max_distance=None):
    """
    计算二维Ising模型的自旋关联函数
    
    关联函数描述了不同距离自旋之间的相关性，是研究临界现象的重要工具。
    
    Parameters:
    -----------
    lattice : list or array-like
        二维晶格数组，值为+1或-1
    max_distance : int, optional
        计算关联函数的最大距离，默认为晶格尺寸的一半
        
    Returns:
    --------
    distances : ndarray
        距离数组
    correlation : ndarray
        对应距离的关联函数值
    """
    # 转换为numpy数组
    lattice = np.asarray(lattice)
    
    L = lattice.shape[0]
    if max_distance is None:
        max_distance = L // 2
    
    # 初始化结果数组
    distances = np.arange(0, max_distance + 1)
    correlation = np.zeros(max_distance + 1)
    
    # 计算平均磁化强度
    avg_spin = np.mean(lattice)
    
    # 计算关联函数
    # C(r) = <s_i * s_j> - <s_i><s_j>，其中r是距离
    for r in distances:
        # 统计所有间隔为r的自旋对
        count = 0
        corr_sum = 0
        
        for i in range(L):
            for j in range(L):
                # 水平方向
                if j + r < L:
                    corr_sum += lattice[i, j] * lattice[i, j+r]
                    count += 1
                
                # 垂直方向
                if i + r < L:
                    corr_sum += lattice[i, j] * lattice[i+r, j]
                    count += 1
        
        if count > 0:
            correlation[r] = corr_sum / count - avg_spin**2
    
    return distances, correlation


# ============================================================================
# 晶体结构模块 (Crystal Structure Module)
# ============================================================================

def calculate_bond_angle(v1, v2):
    """
    计算两个键向量之间的夹角
    
    Parameters:
    -----------
    v1 : list or array-like
        第一个键向量
    v2 : list or array-like
        第二个键向量
    
    Returns:
    --------
    float
        两个键向量之间的夹角（以度为单位）
    """
    # 转换为numpy数组
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    
    # 计算向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 计算夹角的余弦值
    cos_angle = dot_product / (norm_v1 * norm_v2)
    
    # 由于浮点数精度问题，确保cos_angle在[-1, 1]范围内
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # 计算夹角（以度为单位）
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle


def generate_diamond_lattice(size=2):
    """
    生成金刚石晶格结构
    
    Parameters:
    -----------
    size : int, optional
        晶格的大小（单位晶胞的数量），默认为2
    
    Returns:
    --------
    tuple
        (positions, bonds) 其中positions是原子位置的数组，bonds是键连接的索引对列表
    """
    # 金刚石晶格的基本单位晶胞
    a = 1.0  # 晶格常数
    
    # FCC晶格点
    fcc_points = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    
    # 金刚石晶格点（FCC + 基础移位）
    diamond_basis = np.array([
        [0, 0, 0],
        [0.25, 0.25, 0.25]
    ]) * a
    
    # 生成完整的金刚石晶格
    positions = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k]) * a
                for fcc_point in fcc_points:
                    for basis_point in diamond_basis:
                        position = fcc_point + basis_point + offset
                        positions.append(position)
    
    positions = np.array(positions)
    
    # 生成键连接
    bonds = []
    # 键长阈值（金刚石的键长约为a*sqrt(3)/4）
    bond_threshold = a * np.sqrt(3) / 4 * 1.1
    
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < bond_threshold:
                bonds.append((i, j))
    
    return positions, bonds


def calculate_diamond_bond_angles():
    """
    计算金刚石晶格中键之间的夹角
    
    Returns:
    --------
    float
        金刚石晶格中任意两个键之间的夹角（以度为单位）
    """
    # 金刚石晶格中心原子到四个最近邻的向量
    # 这些向量指向正四面体的四个顶点
    vectors = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / np.sqrt(3)
    
    # 计算任意两个键之间的夹角
    angles = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            angle = calculate_bond_angle(vectors[i], vectors[j])
            angles.append(angle)
    
    # 所有夹角应该相同，返回平均值
    return np.mean(angles)


# ============================================================================
# 可视化模块 (Visualization Module)
# ============================================================================

def plot_phase_diagram(temp_range, results, observable='specific_heat', title=None):
    """
    绘制相变图，展示物理量随温度的变化
    
    可视化相变过程中物理量的变化，帮助识别临界点和相变行为。
    
    Parameters:
    -----------
    temp_range : array_like
        温度范围数组
    results : list of dict
        每个温度点的模拟结果字典列表
    observable : str, optional
        要绘制的物理量，可选值：'specific_heat', 'susceptibility', 
        'avg_energy', 'avg_magnetization'，默认为'specific_heat'
    title : str, optional
        图表标题，如果为None则自动生成
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        生成的图表对象
    """
    # 提取要绘制的物理量
    y_values = [result[observable] for result in results]
    
    # 设置图表标题
    if title is None:
        observable_names = {
            'specific_heat': '比热',
            'susceptibility': '磁化率',
            'avg_energy': '平均能量',
            'avg_magnetization': '平均磁化强度'
        }
        title = f"Ising模型 - {observable_names.get(observable, observable)}随温度的变化"
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_range, y_values, 'o-', linewidth=2)
    ax.set_xlabel('温度 (J/k_B)', fontsize=12)
    ax.set_ylabel(observable_names.get(observable, observable), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    filename = f"./images/phase_diagram_{observable}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {filename}")
    
    return fig


def visualize_lattice(lattice, title=None):
    """
    可视化二维晶格构型（热图方式）
    
    用于展示Ising模型的自旋构型，帮助直观理解系统状态。
    
    Parameters:
    -----------
    lattice : list or array-like
        二维晶格数组，值为+1或-1
    title : str, optional
        图表标题，如果为None则使用默认标题
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        生成的图表对象
    """
    # 转换为numpy数组
    lattice = np.asarray(lattice)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 使用imshow显示晶格，+1为白色，-1为黑色
    cmap = plt.cm.binary
    im = ax.imshow(lattice, cmap=cmap, vmin=-1, vmax=1)
    
    # 设置标题
    if title is None:
        title = "Ising模型晶格构型"
    ax.set_title(title, fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([-1, 1])
    cbar.set_ticklabels(['自旋向下', '自旋向上'])
    
    # 保存图表
    filename = "./images/ising_lattice.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"晶格图已保存至: {filename}")
    
    return fig


def visualize_diamond_lattice(size=2, show=True, save_path=None):
    """
    可视化金刚石晶格结构（3D方式）
    
    Parameters:
    -----------
    size : int, optional
        晶格的大小，默认为2
    show : bool, optional
        是否显示图形，默认为True
    save_path : str, optional
        保存图形的路径，如果为None则不保存
    
    Returns:
    --------
    None
    """
    positions, bonds = generate_diamond_lattice(size)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原子
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               color='gray', s=100, alpha=0.7)
    
    # 绘制键
    for bond in bonds:
        i, j = bond
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                [positions[i, 2], positions[j, 2]], 'k-', lw=1)
    
    # 突出显示一个原子及其四个最近邻
    # 选择一个位于晶格中心附近的原子
    center_idx = np.argmin(np.sum((positions - np.mean(positions, axis=0))**2, axis=1))
    ax.scatter(positions[center_idx, 0], positions[center_idx, 1], positions[center_idx, 2], 
               color='red', s=150)
    
    # 找到与中心原子相连的原子
    neighbor_indices = [j if i == center_idx else i for i, j in bonds if i == center_idx or j == center_idx]
    
    # 突出显示这些原子和键
    for idx in neighbor_indices:
        ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], 
                   color='blue', s=120)
        ax.plot([positions[center_idx, 0], positions[idx, 0]],
                [positions[center_idx, 1], positions[idx, 1]],
                [positions[center_idx, 2], positions[idx, 2]], 'r-', lw=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('金刚石晶格结构')
    
    # 设置坐标轴范围
    ax.set_xlim([0, size])
    ax.set_ylim([0, size])
    ax.set_zlim([0, size])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_energy_spectrum(eigenvalues, labels=None, show=True, save_path=None):
    """
    绘制能谱图
    
    Parameters:
    -----------
    eigenvalues : list or array-like
        能级值数组
    labels : list, optional
        能级标签
    show : bool, optional
        是否显示图形，默认为True
    save_path : str, optional
        保存图形的路径，如果为None则不保存
    
    Returns:
    --------
    None
    """
    # 转换为numpy数组
    eigenvalues = np.asarray(eigenvalues)
    
    plt.figure(figsize=(8, 6))
    
    # 绘制能级
    for i, energy in enumerate(eigenvalues):
        plt.plot([0, 1], [energy, energy], 'b-', linewidth=2)
        
        # 添加标签
        if labels and i < len(labels):
            plt.text(1.1, energy, labels[i], fontsize=12)
        else:
            plt.text(1.1, energy, f"E{i}", fontsize=12)
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(min(eigenvalues) - 0.5, max(eigenvalues) + 0.5)
    plt.ylabel('能量')
    plt.title('量子系统能谱')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ising_results(lattice, magnetizations, energies, T, show=True, save_path=None):
    """
    绘制伊辛模型模拟结果
    
    Parameters:
    -----------
    lattice : list or array-like
        最终晶格状态
    magnetizations : list or array-like
        磁化强度历史
    energies : list or array-like
        能量历史
    T : float
        模拟温度
    show : bool, optional
        是否显示图形，默认为True
    save_path : str, optional
        保存图形的路径前缀，如果为None则不保存
    
    Returns:
    --------
    None
    """
    # 转换为numpy数组
    lattice = np.asarray(lattice)
    magnetizations = np.asarray(magnetizations)
    energies = np.asarray(energies)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制晶格状态
    ax1 = fig.add_subplot(131)
    im = ax1.imshow(lattice, cmap='coolwarm')
    ax1.set_title(f'伊辛模型晶格 (T={T:.2f})')
    plt.colorbar(im, ax=ax1, label='自旋')
    
    # 绘制磁化强度历史
    ax2 = fig.add_subplot(132)
    ax2.plot(magnetizations, 'b-')
    ax2.set_xlabel('蒙特卡洛步')
    ax2.set_ylabel('磁化强度 |m|')
    ax2.set_title('磁化强度演化')
    ax2.grid(True)
    
    # 绘制能量历史
    ax3 = fig.add_subplot(133)
    ax3.plot(energies, 'r-')
    ax3.set_xlabel('蒙特卡洛步')
    ax3.set_ylabel('能量密度 e')
    ax3.set_title('能量演化')
    ax3.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_T{T:.2f}.png", dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# 统一接口类 (Unified Interface Classes)
# ============================================================================

class HamiltonianSolver:
    """
    统一的哈密顿量求解器
    自动根据系统规模选择最优算法
    """
    
    def __init__(self, method='auto'):
        """
        Parameters:
        -----------
        method : str
            'sparse' - 稀疏矩阵方法（适合大系统）
            'dense' - 密集矩阵方法（适合小系统）
            'auto' - 自动选择（根据系统大小）
        """
        self.method = method
    
    def solve(self, hamiltonian, k=6, which='SA'):
        """
        求解哈密顿量的本征值和本征态
        
        Parameters:
        -----------
        hamiltonian : scipy.sparse.csr_matrix or ndarray
            系统的哈密顿量
        k : int, optional
            需要计算的本征值数量
        which : str, optional
            指定计算哪些本征值
        
        Returns:
        --------
        eigenvalues : ndarray
            本征值数组
        eigenvectors : ndarray
            本征向量数组
        """
        if self.method == 'auto':
            size = hamiltonian.shape[0]
            use_sparse = size > 50 or sp.issparse(hamiltonian)
        else:
            use_sparse = (self.method == 'sparse')
        
        if use_sparse:
            return solve_eigensystem_from_matrix(hamiltonian, k, which)
        else:
            return simulate_quantum_system(hamiltonian, num_states=k)


class IsingSimulator:
    """
    统一的Ising模型模拟器
    """
    
    def run_simulation(self, lattice_size, temperature, num_steps, 
                      algorithm='metropolis', compute_correlation=False, **kwargs):
        """
        运行Ising模型模拟
        
        Parameters:
        -----------
        lattice_size : tuple
            晶格尺寸
        temperature : float
            温度
        num_steps : int
            模拟步数
        algorithm : str
            算法类型（目前支持'metropolis'）
        compute_correlation : bool
            是否计算关联函数
        
        Returns:
        --------
        dict
            模拟结果
        """
        if algorithm == 'metropolis':
            results = monte_carlo_ising(lattice_size, temperature, num_steps, 
                                       J=kwargs.get('J', 1.0))
            
            if compute_correlation:
                distances, correlation = calculate_correlation_function(results['lattice'])
                results['correlation_distances'] = distances
                results['correlation_function'] = correlation
            
            return results
        else:
            raise ValueError(f"不支持的算法: {algorithm}")


class LatticeGenerator:
    """
    通用晶格生成器
    """
    
    @staticmethod
    def generate(lattice_type, size, **kwargs):
        """
        生成指定类型的晶格
        
        Parameters:
        -----------
        lattice_type : str
            晶格类型，如 'diamond'
        size : int
            晶格大小
        
        Returns:
        --------
        tuple
            (positions, bonds)
        """
        if lattice_type == 'diamond':
            return generate_diamond_lattice(size)
        else:
            raise ValueError(f"不支持的晶格类型: {lattice_type}")
    
    @staticmethod
    def calculate_bond_angles(lattice_type):
        """
        计算指定晶格类型的键角
        
        Parameters:
        -----------
        lattice_type : str
            晶格类型
        
        Returns:
        --------
        float
            键角（度）
        """
        if lattice_type == 'diamond':
            return calculate_diamond_bond_angles()
        else:
            raise ValueError(f"不支持的晶格类型: {lattice_type}")


class CondensedMatterVisualizer:
    """
    统一的可视化接口
    """
    
    @staticmethod
    def visualize_lattice_2d(lattice, title=None):
        """2D晶格可视化（热图）"""
        return visualize_lattice(lattice, title)
    
    @staticmethod
    def visualize_lattice_3d(positions, bonds, size, show=True, save_path=None):
        """3D晶格可视化"""
        # 直接调用3D可视化，但这里我们需要自定义实现
        # 为了兼容性，我们包装一下
        return visualize_diamond_lattice(size, show, save_path)
    
    @staticmethod
    def plot_phase_diagram(temp_range, results, observable='specific_heat', title=None):
        """相图绘制"""
        return plot_phase_diagram(temp_range, results, observable, title)
    
    @staticmethod
    def plot_energy_spectrum(eigenvalues, labels=None, show=True, save_path=None):
        """能谱图"""
        return plot_energy_spectrum(eigenvalues, labels, show, save_path)
    
    @staticmethod
    def plot_ising_results(lattice, magnetizations, energies, T, show=True, save_path=None):
        """Ising模拟结果"""
        return plot_ising_results(lattice, magnetizations, energies, T, show, save_path)


# ============================================================================
# 主函数和测试 (Main Functions and Tests)
# ============================================================================

def main_version_4179():
    """
    主函数：演示如何使用工具函数求解凝聚态物理中的关键问题
    （来自condensed_matter_tools_4179.py）
    """
    print("凝聚态物理计算工具演示 (Version 4179)")
    print("=" * 50)
    
    # 示例1：量子多体系统模拟
    print("\n1. 量子多体系统模拟 - 一维紧束缚模型")
    # 系统参数
    system_size = 20  # 一维链长度
    hopping = -1.0    # 跃迁积分
    
    # 构建相互作用矩阵
    interaction = np.array([hopping])
    
    # 构建哈密顿量
    H = construct_hamiltonian(system_size, interaction, periodic=True)
    print(f"哈密顿量构建完成，维度: {H.shape}")
    
    # 求解本征值和本征态
    eigenvalues, eigenvectors = solve_eigensystem(system_size=system_size, interaction_matrix=interaction, periodic=True, k=6)
    print("最低的6个能级:")
    for i, e in enumerate(eigenvalues):
        print(f"  E{i} = {e:.6f}")
    
    # 计算基态密度矩阵
    ground_state = eigenvectors[:, 0]
    density_matrix = calculate_density_matrix(ground_state)
    print(f"基态密度矩阵迹: {np.trace(density_matrix):.6f}")
    
    # 可视化基态波函数
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(system_size), np.abs(ground_state)**2, 'o-')
    plt.xlabel('格点位置')
    plt.ylabel('概率密度')
    plt.title('一维紧束缚模型基态波函数')
    plt.grid(True)
    plt.savefig('./images/ground_state_wavefunction.png', dpi=300)
    print("基态波函数图已保存至: ./images/ground_state_wavefunction.png")
    plt.close()
    
    # 示例2：相变与临界现象模拟
    print("\n2. 相变与临界现象模拟 - 二维Ising模型")
    
    # 系统参数
    lattice_size = (20, 20)
    temperatures = np.linspace(1.5, 3.5, 5)  # 温度范围，包含临界温度Tc≈2.269
    mc_steps = 10000  # 蒙特卡洛步数
    
    # 存储不同温度的结果
    results = []
    
    # 对每个温度进行模拟
    for temp in temperatures:
        print(f"  模拟温度 T = {temp:.2f}...")
        result = monte_carlo_ising(lattice_size, temp, mc_steps)
        results.append(result)
        
        # 如果是接近临界温度的模拟，可视化晶格构型
        if abs(temp - 2.27) < 0.1:
            visualize_lattice(result['lattice'], 
                            f"Ising模型晶格构型 (T ≈ Tc ≈ {temp:.2f})")
            plt.close()
    
    # 绘制相变图
    plot_phase_diagram(temperatures, results, 'specific_heat')
    plt.close()
    plot_phase_diagram(temperatures, results, 'avg_magnetization')
    plt.close()
    
    # 计算并绘制关联函数（在临界温度附近）
    critical_idx = np.argmin(np.abs(temperatures - 2.27))
    critical_lattice = results[critical_idx]['lattice']
    distances, correlation = calculate_correlation_function(critical_lattice)
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(distances, np.abs(correlation), 'o-')
    plt.xlabel('距离 r')
    plt.ylabel('关联函数 |C(r)|')
    plt.title(f'Ising模型关联函数 (T ≈ Tc ≈ {temperatures[critical_idx]:.2f})')
    plt.grid(True)
    plt.savefig('./images/correlation_function.png', dpi=300)
    print("关联函数图已保存至: ./images/correlation_function.png")
    plt.close()
    
    print("\n计算完成！所有结果已保存到images目录")


def main_version_17363():
    """
    主函数：演示如何使用工具函数求解凝聚态物理中的关键问题
    （来自condensed_matter_tools_17363.py）
    """
    print("凝聚态物理计算工具演示 (Version 17363)")
    print("=" * 50)
    
    # 示例1：金刚石晶格键角计算
    print("\n1. 金刚石晶格键角计算")
    print("-" * 50)
    
    # 计算金刚石晶格中键之间的夹角
    bond_angle = calculate_diamond_bond_angles()
    print(f"金刚石晶格中任意两个键之间的夹角: {bond_angle:.2f}°")
    
    # 可视化金刚石晶格结构
    print("正在生成金刚石晶格结构可视化...")
    visualize_diamond_lattice(size=2, show=False, save_path="./images/diamond_lattice.png")
    print("金刚石晶格结构图已保存至 ./images/diamond_lattice.png")
    
    # 示例2：量子多体系统模拟
    print("\n2. 量子多体系统模拟")
    print("-" * 50)
    
    # 创建一个简单的量子系统哈密顿量（例如，一维紧束缚模型）
    print("模拟一维紧束缚模型...")
    N = 10  # 系统大小
    t = 1.0  # 跃迁强度
    
    # 构建哈密顿量
    H = np.zeros((N, N))
    for i in range(N-1):
        H[i, i+1] = -t
        H[i+1, i] = -t
    
    # 求解本征值和本征态
    eigenvalues, eigenvectors = simulate_quantum_system(H, num_states=N)
    
    print(f"系统的最低5个能级:")
    for i in range(min(5, len(eigenvalues))):
        print(f"E{i} = {eigenvalues[i]:.4f}")
    
    # 绘制能谱
    print("正在生成能谱图...")
    plot_energy_spectrum(eigenvalues, show=False, save_path="./images/energy_spectrum.png")
    print("能谱图已保存至 ./images/energy_spectrum.png")


def main():
    """
    统一主函数：演示统一接口的使用
    """
    print("=" * 60)
    print("凝聚态物理统一工具包演示")
    print("=" * 60)
    
    # 使用统一接口
    print("\n使用统一的HamiltonianSolver...")
    solver = HamiltonianSolver(method='auto')
    
    # 构建一个简单的哈密顿量
    N = 10
    H = np.zeros((N, N))
    for i in range(N-1):
        H[i, i+1] = -1.0
        H[i+1, i] = -1.0
    
    eigenvalues, eigenvectors = solver.solve(H, k=5)
    print(f"自动求解得到的最低5个能级: {eigenvalues}")
    
    print("\n=" * 60)
    print("运行原版本的测试...")
    print("=" * 60)
    
    print("\n" + "="*60)
    main_version_4179()
    print("\n" + "="*60)
    main_version_17363()


if __name__ == "__main__":
    main()

