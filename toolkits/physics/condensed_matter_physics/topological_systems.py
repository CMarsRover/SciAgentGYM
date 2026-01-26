#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
拓扑系统模块

提供拓扑物理相关的计算工具，包括：
- SSH模型（Su-Schrieffer-Heeger）
- 拓扑不变量计算
- 边缘态分析

物理背景：
    SSH模型是最简单的1D拓扑绝缘体，展示了拓扑相变和边缘态。
    2016年诺贝尔物理学奖授予了拓扑相变和拓扑物质态的理论发现。
    
作者：SciTools Team
版本：1.0.0
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from typing import Tuple, Dict, Optional
import os

# matplotlib是可选依赖（用于绘图功能）
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def construct_ssh_hamiltonian(N, t1, t2, periodic=False):
    """
    构建SSH模型哈密顿量
    
    SSH模型：H = ∑ₙ [t₁ c†ₙ,ₐ cₙ,ᵦ + t₂ c†ₙ,ᵦ cₙ₊₁,ₐ + h.c.]
    
    其中：
    - t₁: 单胞内跃迁（intra-cell hopping）
    - t₂: 单胞间跃迁（inter-cell hopping）
    - N: 单胞数（总格点数为2N）
    
    Parameters:
    -----------
    N : int
        单胞数
    t1 : float
        单胞内跃迁积分
    t2 : float
        单胞间跃迁积分
    periodic : bool, default=False
        是否使用周期性边界条件（PBC）
        False: 开边界条件（OBC），可以观察边缘态
        True: 周期性边界条件
        
    Returns:
    --------
    H : np.ndarray
        哈密顿量矩阵（2N × 2N）
        
    Notes:
    ------
    拓扑不变量：
    - δ = (t₁ - t₂)/(t₁ + t₂)
    - δ > 0: 平庸相（trivial phase）
    - δ < 0: 拓扑相（topological phase，有边缘态）
    - δ = 0: 拓扑相变点
    
    Examples:
    ---------
    >>> # 拓扑相（有边缘态）
    >>> H = construct_ssh_hamiltonian(N=25, t1=0.5, t2=1.5)
    >>> eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    >>> # 平庸相（无边缘态）
    >>> H = construct_ssh_hamiltonian(N=25, t1=1.5, t2=0.5)
    """
    total_sites = 2 * N  # 总格点数
    H = np.zeros((total_sites, total_sites), dtype=np.float64)
    
    # 构建SSH哈密顿量
    # 格点编号：0(A), 1(B), 2(A), 3(B), ..., 2N-2(A), 2N-1(B)
    
    for n in range(N):
        # 单胞内跃迁：t₁ (A → B)
        site_A = 2 * n
        site_B = 2 * n + 1
        
        if site_B < total_sites:
            H[site_A, site_B] = t1
            H[site_B, site_A] = t1  # 厄米共轭
        
        # 单胞间跃迁：t₂ (B → next A)
        if site_B < total_sites - 1:
            next_A = site_B + 1
            H[site_B, next_A] = t2
            H[next_A, site_B] = t2  # 厄米共轭
    
    # 周期性边界条件
    if periodic and N > 1:
        # 连接最后一个B和第一个A
        last_B = 2 * N - 1
        first_A = 0
        H[last_B, first_A] = t2
        H[first_A, last_B] = t2
    
    return H


def calculate_ssh_gap(eigenvalues, exclude_edge_states=True, edge_threshold=0.01):
    """
    计算SSH模型的能隙
    
    对于半填充系统，能隙是最高占据态和最低空态之间的能量差。
    对于拓扑相，可以排除能隙中的边缘态来计算体态能隙。
    
    Parameters:
    -----------
    eigenvalues : list or array-like
        本征能量（已排序）
    exclude_edge_states : bool, default=True
        是否排除能隙中的边缘态（零能态）
        True: 计算体态能隙（bulk gap）
        False: 计算包含边缘态的总能隙
    edge_threshold : float, default=0.01
        判断零能态的阈值
        
    Returns:
    --------
    gap : float
        能隙
    valence_max : float
        价带顶（最高占据态能量）
    conduction_min : float
        导带底（最低空态能量）
        
    Examples:
    ---------
    >>> eigenvalues, _ = np.linalg.eigh(H)
    >>> gap, E_v, E_c = calculate_ssh_gap(eigenvalues)  # 体态能隙
    >>> gap_total, _, _ = calculate_ssh_gap(eigenvalues, exclude_edge_states=False)  # 总能隙
    """
    # 转换为numpy数组
    eigenvalues = np.asarray(eigenvalues)
    N_states = len(eigenvalues)
    
    # 如果需要排除边缘态
    if exclude_edge_states:
        # 找出非边缘态（能量远离零）
        non_edge_mask = np.abs(eigenvalues) > edge_threshold
        
        if np.sum(non_edge_mask) >= 2:
            # 使用体态来计算能隙
            bulk_eigenvalues = eigenvalues[non_edge_mask]
            N_bulk = len(bulk_eigenvalues)
            N_filled_bulk = N_bulk // 2
            
            if N_filled_bulk >= N_bulk:
                return 0.0, bulk_eigenvalues[-1], bulk_eigenvalues[-1]
            
            valence_max = bulk_eigenvalues[N_filled_bulk - 1]
            conduction_min = bulk_eigenvalues[N_filled_bulk]
            gap = conduction_min - valence_max
            
            return gap, valence_max, conduction_min
    
    # 不排除边缘态，或没有边缘态
    N_filled = N_states // 2
    
    if N_filled >= N_states:
        return 0.0, eigenvalues[-1], eigenvalues[-1]
    
    valence_max = eigenvalues[N_filled - 1]
    conduction_min = eigenvalues[N_filled]
    gap = conduction_min - valence_max
    
    return gap, valence_max, conduction_min


def identify_edge_states(eigenvalues, eigenvectors, gap_threshold=0.1):
    """
    识别SSH模型的边缘态
    
    边缘态的特征：
    1. 能量接近零（在能隙中）
    2. 波函数局域在边缘
    
    Parameters:
    -----------
    eigenvalues : list or array-like
        本征能量
    eigenvectors : list or array-like
        本征态（列向量）
    gap_threshold : float, default=0.1
        判断是否在能隙中的阈值
        
    Returns:
    --------
    edge_state_indices : list
        边缘态的索引
    edge_state_energies : list
        边缘态的能量
        
    Examples:
    ---------
    >>> indices, energies = identify_edge_states(eigenvalues, eigenvectors)
    >>> print(f"找到 {len(indices)} 个边缘态")
    """
    # 转换为numpy数组
    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)
    
    edge_state_indices = []
    edge_state_energies = []
    
    N_states = len(eigenvalues)
    N_filled = N_states // 2
    
    # 计算能隙
    gap, E_v, E_c = calculate_ssh_gap(eigenvalues)
    
    # 寻找能隙中的态
    for i in range(N_states):
        E = eigenvalues[i]
        # 判断是否在能隙中（接近零能）
        if abs(E) < gap_threshold and E_v < E < E_c:
            edge_state_indices.append(i)
            edge_state_energies.append(E)
    
    return edge_state_indices, edge_state_energies


def calculate_edge_state_localization(eigenvector, edge='left'):
    """
    计算边缘态的局域化程度
    
    Parameters:
    -----------
    eigenvector : list or array-like
        本征态波函数
    edge : str, default='left'
        'left': 左边缘
        'right': 右边缘
        
    Returns:
    --------
    localization : float
        边缘局域化程度（0-1，1表示完全局域在边缘）
        
    Examples:
    ---------
    >>> psi = eigenvectors[:, edge_state_idx]
    >>> loc = calculate_edge_state_localization(psi, edge='left')
    """
    # 转换为numpy数组
    eigenvector = np.asarray(eigenvector)
    N = len(eigenvector)
    prob_density = np.abs(eigenvector)**2
    
    # 定义边缘区域（前1/4或后1/4）
    edge_size = max(N // 4, 1)
    
    if edge == 'left':
        edge_prob = np.sum(prob_density[:edge_size])
    elif edge == 'right':
        edge_prob = np.sum(prob_density[-edge_size:])
    else:
        raise ValueError("edge must be 'left' or 'right'")
    
    return edge_prob


def scan_ssh_model(N, t_sum, delta_values, periodic=False):
    """
    扫描SSH模型的拓扑相变
    
    固定 t₁ + t₂ = t_sum，扫描 δ = (t₁ - t₂)/(t₁ + t₂)
    
    Parameters:
    -----------
    N : int
        单胞数
    t_sum : float
        t₁ + t₂ 的值
    delta_values : list or array-like
        δ值列表
    periodic : bool, default=False
        边界条件
        
    Returns:
    --------
    results : dict
        包含以下键：
        - 'delta_values': δ值
        - 't1_values': t₁值
        - 't2_values': t₂值
        - 'gaps': 能隙
        - 'eigenvalues_list': 所有本征能量
        - 'edge_states_count': 边缘态数量
        - 'phases': 相分类（'topological', 'trivial', 'critical'）
        - 'analysis': 分析结果字典，包含：
            * 'critical_point': 拓扑相变点（δ≈0）
            * 'min_gap': 最小能隙
            * 'topological_count': 拓扑相点数
            * 'edge_states_found': 是否找到边缘态
        
    Examples:
    ---------
    >>> delta_values = np.linspace(-0.8, 0.8, 21)
    >>> results = scan_ssh_model(N=25, t_sum=2.0, delta_values=delta_values)
    >>> print(f"δ = -0.5 时的能隙：{results['gaps'][5]:.4f}")
    >>> print(f"拓扑相变点：δ ≈ {results['analysis']['critical_point']:.3f}")
    """
    # 转换为numpy数组
    delta_values = np.asarray(delta_values)
    
    results = {
        'delta_values': delta_values.copy(),
        't1_values': [],
        't2_values': [],
        'gaps': [],
        'eigenvalues_list': [],
        'edge_states_count': [],
        'phases': []  # 新增：相分类
    }
    
    for delta in delta_values:
        # 从 δ 计算 t₁ 和 t₂
        # δ = (t₁ - t₂)/(t₁ + t₂)
        # t₁ + t₂ = t_sum
        # 解得：t₁ = t_sum * (1 + δ) / 2
        #       t₂ = t_sum * (1 - δ) / 2
        
        t1 = t_sum * (1 + delta) / 2
        t2 = t_sum * (1 - delta) / 2
        
        # 构建哈密顿量
        H = construct_ssh_hamiltonian(N, t1, t2, periodic=periodic)
        
        # 求解
        eigenvalues, eigenvectors = la.eigh(H)
        
        # 计算能隙
        gap, _, _ = calculate_ssh_gap(eigenvalues)
        
        # 识别边缘态（只对开边界条件）
        if not periodic and delta < 0:  # 拓扑相
            edge_indices, _ = identify_edge_states(eigenvalues, eigenvectors)
            n_edge = len(edge_indices)
        else:
            n_edge = 0
        
        # 判断相
        if abs(delta) < 0.05:  # 临界区域
            phase = 'critical'
        elif delta < 0:
            phase = 'topological'
        else:
            phase = 'trivial'
        
        # 存储结果
        results['t1_values'].append(t1)
        results['t2_values'].append(t2)
        results['gaps'].append(gap)
        results['eigenvalues_list'].append(eigenvalues)
        results['edge_states_count'].append(n_edge)
        results['phases'].append(phase)
    
    # 转换为数组
    for key in ['t1_values', 't2_values', 'gaps', 'edge_states_count']:
        results[key] = np.array(results[key])
    
    # 添加分析结果
    gaps_array = results['gaps']
    idx_min_gap = np.argmin(gaps_array)
    
    results['analysis'] = {
        'critical_point': float(delta_values[idx_min_gap]),
        'min_gap': float(gaps_array[idx_min_gap]),
        'min_gap_index': int(idx_min_gap),
        'topological_count': int(np.sum(delta_values < 0)),
        'trivial_count': int(np.sum(delta_values > 0)),
        'edge_states_found': int(np.max(results['edge_states_count'])) if len(results['edge_states_count']) > 0 else 0,
        'topological_with_edge_states': int(np.sum(np.array(results['edge_states_count']) > 0))
    }
    
    return results


def calculate_winding_number(H_k, k_values):
    """
    计算SSH模型的缠绕数（winding number）
    
    这是一个拓扑不变量，用于区分拓扑相和平庸相
    
    Parameters:
    -----------
    H_k : callable
        动量空间哈密顿量函数 H(k)
    k_values : list or array-like
        动量点
        
    Returns:
    --------
    winding : int
        缠绕数（0或±1）
        
    Notes:
    ------
    需要PythTB工具来完整计算
    这里提供简化版本
    """
    # 转换为numpy数组
    k_values = np.asarray(k_values)
    
    # 简化实现：通过符号判断
    # 完整实现需要Berry相位计算
    return 0  # 占位符


# ============================================================================
# 便捷函数
# ============================================================================

def quick_ssh_analysis(N=25, t_sum=2.0, delta_values=None, verbose=True):
    """
    快速SSH模型分析（便捷函数）
    
    Parameters:
    -----------
    N : int, default=25
        单胞数（总格点数2N=50）
    t_sum : float, default=2.0
        t₁ + t₂
    delta_values : array-like, optional
        δ值列表，默认从-0.8到0.8
    verbose : bool, default=True
        是否打印详细信息
        
    Returns:
    --------
    results : dict
        分析结果
        
    Examples:
    ---------
    >>> results = quick_ssh_analysis()
    >>> print(f"拓扑相变点在 δ = 0")
    """
    if delta_values is None:
        delta_values = np.linspace(-0.8, 0.8, 21)
    
    if verbose:
        print("="*80)
        print("SSH模型快速分析")
        print("="*80)
        print(f"系统参数：N={N} (总格点={2*N}), t₁+t₂={t_sum}")
        print(f"扫描δ范围：{delta_values[0]:.2f} 到 {delta_values[-1]:.2f}")
        print("="*80)
    
    results = scan_ssh_model(N, t_sum, delta_values, periodic=False)
    
    if verbose:
        print("\n结果总结：")
        print(f"{'δ':<10} {'t₁':<10} {'t₂':<10} {'能隙':<15} {'边缘态数':<10}")
        print("-"*55)
        for i in range(len(delta_values)):
            print(f"{results['delta_values'][i]:<10.2f} "
                  f"{results['t1_values'][i]:<10.3f} "
                  f"{results['t2_values'][i]:<10.3f} "
                  f"{results['gaps'][i]:<15.6f} "
                  f"{results['edge_states_count'][i]:<10}")
    
    return results


def plot_ssh_phase_transition(results, output_dir='./images', filename='ssh_topological_transition.png'):
    """
    绘制SSH模型的拓扑相变分析图
    
    包含4个子图：
    1. 能隙 vs δ
    2. 边缘态数量 vs δ
    3. 能谱演化
    4. 临界点附近的能隙闭合
    
    Parameters:
    -----------
    results : dict
        由scan_ssh_model返回的结果字典
    output_dir : str, default='./images'
        输出目录
    filename : str, default='ssh_topological_transition.png'
        输出文件名
        
    Returns:
    --------
    filepath : str
        保存的文件路径
        
    Examples:
    ---------
    >>> results = scan_ssh_model(N=25, t_sum=2.0, delta_values=np.linspace(-0.8, 0.8, 21))
    >>> plot_ssh_phase_transition(results)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib未安装，无法绘图")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    delta_values = results['delta_values']
    gaps = results['gaps']
    edge_states_count = results['edge_states_count']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 能隙 vs δ
    ax1 = axes[0, 0]
    ax1.plot(delta_values, gaps, 'o-', linewidth=2, markersize=6, color='blue')
    
    # 标记特殊点（如δ=-0.5）
    if -0.5 in delta_values or np.any(np.abs(delta_values - (-0.5)) < 0.01):
        idx_minus_05 = np.argmin(np.abs(delta_values - (-0.5)))
        ax1.plot(delta_values[idx_minus_05], gaps[idx_minus_05], 'o', 
                markersize=15, color='red', 
                markeredgecolor='darkred', markeredgewidth=2,
                label=f'$\\delta={delta_values[idx_minus_05]:.2f}$: gap={gaps[idx_minus_05]:.4f}')
    
    # 标记拓扑相变点
    ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, 
                label='拓扑相变点 $\\delta=0$')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax1.set_xlabel('拓扑参数 $\\delta = (t_1-t_2)/(t_1+t_2)$', fontsize=12)
    ax1.set_ylabel('能隙 $\\Delta E_{\\mathrm{gap}}$', fontsize=12)
    ax1.set_title('SSH模型的拓扑相变', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 添加相区标注
    if len(gaps) > 0:
        ax1.text(-0.4, max(gaps)*0.9, '拓扑相\n(边缘态)', 
                ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax1.text(0.4, max(gaps)*0.9, '平庸相\n(无边缘态)', 
                ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # 2. 边缘态数量 vs δ
    ax2 = axes[0, 1]
    ax2.plot(delta_values, edge_states_count, 's-', linewidth=2, markersize=6, color='orangered')
    ax2.axvline(x=0, color='green', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('拓扑参数 $\\delta$', fontsize=12)
    ax2.set_ylabel('边缘态数量', fontsize=12)
    ax2.set_title('边缘态数量随 $\\delta$ 的变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(edge_states_count) > 0:
        ax2.set_ylim([-0.5, max(edge_states_count) + 1])
    
    # 3. 能谱演化（选几个代表性的δ值）
    ax3 = axes[1, 0]
    
    delta_samples = [-0.5, -0.1, 0.0, 0.1, 0.5]
    colors = ['blue', 'cyan', 'green', 'orange', 'red']
    
    for delta_val, color in zip(delta_samples, colors):
        idx = np.argmin(np.abs(delta_values - delta_val))
        if 'eigenvalues_list' in results and idx < len(results['eigenvalues_list']):
            eigenvals = results['eigenvalues_list'][idx]
            
            # 绘制能谱
            n_states = len(eigenvals)
            ax3.scatter(np.ones(n_states) * delta_val, eigenvals, 
                       s=10, alpha=0.6, color=color, label=f'$\\delta={delta_val:.1f}$')
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax3.axvline(x=0, color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('拓扑参数 $\\delta$', fontsize=12)
    ax3.set_ylabel('本征能量', fontsize=12)
    ax3.set_title('能谱演化', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-2.5, 2.5])
    
    # 4. 能隙闭合放大图（δ接近0的区域）
    ax4 = axes[1, 1]
    
    # 选择 δ 在 -0.2 到 0.2 范围内的点
    mask = (delta_values >= -0.2) & (delta_values <= 0.2)
    if np.any(mask):
        ax4.plot(delta_values[mask], gaps[mask], 'o-', linewidth=2, markersize=8, color='purple')
    ax4.axvline(x=0, color='green', linestyle='--', linewidth=2, label='相变点')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax4.set_xlabel('拓扑参数 $\\delta$', fontsize=12)
    ax4.set_ylabel('能隙 $\\Delta E_{\\mathrm{gap}}$', fontsize=12)
    ax4.set_title('拓扑相变点附近的能隙闭合', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"图表已保存：{filepath}")
    
    plt.close()
    
    return filepath


def plot_ssh_edge_states(N, t1, t2, output_dir='./images', filename='ssh_edge_states.png', 
                        periodic=False, max_states=4):
    """
    绘制SSH模型的边缘态波函数
    
    展示拓扑相中边缘态的局域化特征
    
    Parameters:
    -----------
    N : int
        单胞数
    t1 : float
        单胞内跃迁
    t2 : float
        单胞间跃迁
    output_dir : str, default='./images'
        输出目录
    filename : str, default='ssh_edge_states.png'
        输出文件名
    periodic : bool, default=False
        边界条件
    max_states : int, default=4
        最多绘制的边缘态数量
        
    Returns:
    --------
    filepath : str or None
        保存的文件路径，如果没有边缘态则返回None
        
    Examples:
    ---------
    >>> # 拓扑相（t2 > t1）
    >>> plot_ssh_edge_states(N=25, t1=0.5, t2=1.5)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("⚠️  matplotlib未安装，无法绘图")
        return None
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建哈密顿量
    H = construct_ssh_hamiltonian(N, t1, t2, periodic=periodic)
    eigenvalues, eigenvectors = la.eigh(H)
    
    # 识别边缘态
    edge_indices, edge_energies = identify_edge_states(eigenvalues, eigenvectors)
    
    if len(edge_indices) == 0:
        print("未找到边缘态（可能是平庸相或周期性边界条件）")
        return None
    
    # 计算δ值
    delta = (t1 - t2) / (t1 + t2)
    
    # 绘制前几个边缘态
    num_plot = min(len(edge_indices), max_states)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(max_states):
        ax = axes[i]
        
        if i < len(edge_indices):
            idx_state = edge_indices[i]
            psi = eigenvectors[:, idx_state]
            E = eigenvalues[idx_state]
            
            positions = np.arange(len(psi))
            ax.plot(positions, np.abs(psi)**2, 'o-', linewidth=2, markersize=4)
            
            ax.set_xlabel('格点位置', fontsize=11)
            ax.set_ylabel('概率密度 $|\\psi|^2$', fontsize=11)
            ax.set_title(f'边缘态 {i+1}: $E={E:.4f}$', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.axis('off')
    
    plt.suptitle(f'SSH模型边缘态波函数 ($\\delta={delta:.2f}$, N={N}, 拓扑相)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"边缘态波函数图已保存：{filepath}")
    
    plt.close()
    
    return filepath


def example_usage():
    """示例：演示如何使用本模块"""
    print("="*80)
    print("拓扑系统模块 - 使用示例")
    print("="*80)
    
    # 示例1：构建SSH哈密顿量
    print("\n示例1：构建SSH哈密顿量")
    N = 25
    t1 = 0.5
    t2 = 1.5
    H = construct_ssh_hamiltonian(N, t1, t2, periodic=False)
    print(f"  单胞数：N = {N}")
    print(f"  总格点数：{2*N}")
    print(f"  t₁ = {t1}, t₂ = {t2}")
    print(f"  δ = {(t1-t2)/(t1+t2):.3f}")
    print(f"  哈密顿量形状：{H.shape}")
    
    # 示例2：计算能隙和边缘态
    print("\n示例2：计算能隙和边缘态")
    eigenvalues, eigenvectors = la.eigh(H)
    gap, E_v, E_c = calculate_ssh_gap(eigenvalues)
    print(f"  能隙：Δ = {gap:.6f}")
    print(f"  价带顶：E_v = {E_v:.6f}")
    print(f"  导带底：E_c = {E_c:.6f}")
    
    edge_indices, edge_energies = identify_edge_states(eigenvalues, eigenvectors)
    print(f"  边缘态数量：{len(edge_indices)}")
    if len(edge_indices) > 0:
        print(f"  边缘态能量：{edge_energies}")
    
    # 示例3：快速分析
    print("\n示例3：快速分析拓扑相变")
    delta_values = [-0.5, 0.0, 0.5]
    results = quick_ssh_analysis(N=25, delta_values=delta_values, verbose=False)
    
    print(f"  δ=-0.5 (拓扑相): 能隙={results['gaps'][0]:.4f}, "
          f"边缘态={results['edge_states_count'][0]}")
    print(f"  δ=0.0  (临界点): 能隙={results['gaps'][1]:.4f}")
    print(f"  δ=+0.5 (平庸相): 能隙={results['gaps'][2]:.4f}, "
          f"边缘态={results['edge_states_count'][2]}")
    
    print("\n✓ 示例完成！")


if __name__ == "__main__":
    example_usage()

