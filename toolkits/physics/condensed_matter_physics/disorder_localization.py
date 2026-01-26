#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
无序系统与Anderson局域化模块

提供Anderson局域化相关的计算工具，包括：
- 构建带随机势的哈密顿量
- 计算逆参与率（IPR）
- 计算局域长度
- 多样本平均

物理背景：
    Anderson局域化（1977年诺贝尔奖）描述了无序势场中波函数的空间局域性。
    在一维和二维系统中，任意弱的无序都会导致所有态局域化。
    
作者：SciTools Team
版本：1.0.0
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from typing import Tuple, List, Optional, Dict
import warnings


def construct_anderson_hamiltonian_1d(N, t=1.0, disorder_strength=0.0, periodic=False, seed=None):
    """
    构建一维Anderson模型哈密顿量
    
    哈密顿量：H = -t∑ᵢ(c†ᵢcᵢ₊₁ + h.c.) + ∑ᵢεᵢnᵢ
    其中εᵢ在[-W/2, W/2]均匀分布
    
    Parameters:
    -----------
    N : int
        系统格点数
    t : float, default=1.0
        跃迁积分（hopping amplitude）
    disorder_strength : float, default=0.0
        无序强度W（随机势的范围）
    periodic : bool, default=False
        是否使用周期性边界条件
    seed : int, optional
        随机数种子，用于可重复性
        
    Returns:
    --------
    H : scipy.sparse.csr_matrix or np.ndarray
        哈密顿量矩阵
    potential : np.ndarray
        随机势能场（用于记录）
        
    Examples:
    ---------
    >>> H, V = construct_anderson_hamiltonian_1d(100, t=1.0, disorder_strength=4.0)
    >>> eigenvalues, eigenvectors = solve_eigensystem(H)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成随机势场
    if disorder_strength > 0:
        potential = disorder_strength * (np.random.random(N) - 0.5)  # [-W/2, W/2]
    else:
        potential = np.zeros(N)
    
    # 构建哈密顿量
    H = sp.lil_matrix((N, N), dtype=np.float64)
    
    # 对角项（随机势）
    for i in range(N):
        H[i, i] = potential[i]
    
    # 跃迁项（最近邻）
    for i in range(N - 1):
        H[i, i+1] = -t
        H[i+1, i] = -t
    
    # 周期性边界条件
    if periodic:
        H[N-1, 0] = -t
        H[0, N-1] = -t
    
    return H.tocsr(), potential


def calculate_inverse_participation_ratio(wavefunction):
    """
    计算逆参与率（Inverse Participation Ratio, IPR）
    
    IPR = ∑ᵢ|ψᵢ|⁴
    
    物理意义：
    - 扩展态（extended state）：IPR ~ 1/N → 0（N→∞）
    - 局域态（localized state）：IPR ~ O(1)（与N无关）
    
    Parameters:
    -----------
    wavefunction : list or array-like
        波函数（本征态），一维数组
        
    Returns:
    --------
    ipr : float
        逆参与率
        
    Examples:
    ---------
    >>> psi = eigenvectors[:, 0]  # 某个本征态
    >>> ipr = calculate_inverse_participation_ratio(psi)
    >>> print(f"IPR = {ipr:.6f}")
    """
    # 转换为numpy数组并归一化（确保∑|ψᵢ|²=1）
    wavefunction = np.asarray(wavefunction).flatten()
    norm = np.sum(np.abs(wavefunction)**2)
    if norm > 0:
        wavefunction = wavefunction / np.sqrt(norm)
    
    # 计算IPR
    ipr = np.sum(np.abs(wavefunction)**4)
    
    return ipr


def calculate_localization_length(wavefunction, method='exponential_fit'):
    """
    计算局域长度（Localization Length）
    
    对于局域态，波函数衰减：|ψ(r)| ~ exp(-|r-r₀|/ξ)
    其中ξ是局域长度
    
    Parameters:
    -----------
    wavefunction : list or array-like
        波函数（本征态）
    method : str, default='exponential_fit'
        计算方法：'exponential_fit', 'second_moment'
        
    Returns:
    --------
    xi : float
        局域长度
        
    Notes:
    ------
    对于扩展态，局域长度发散（ξ→∞）
    """
    # 转换为numpy数组
    wavefunction = np.asarray(wavefunction).flatten()
    N = len(wavefunction)
    
    # 找到波函数峰值位置
    center = np.argmax(np.abs(wavefunction))
    
    if method == 'exponential_fit':
        # 指数拟合方法
        distances = np.arange(N)
        log_amplitude = np.log(np.abs(wavefunction) + 1e-12)
        
        # 只拟合距离中心较近的点
        mask = (distances > center) & (log_amplitude > -10)
        if np.sum(mask) < 3:
            return np.inf  # 扩展态
        
        try:
            # 线性拟合 log|ψ| vs r
            coeffs = np.polyfit(distances[mask] - center, log_amplitude[mask], 1)
            xi = -1.0 / coeffs[0] if coeffs[0] < 0 else np.inf
        except:
            xi = np.inf
            
    elif method == 'second_moment':
        # 二次矩方法
        positions = np.arange(N)
        prob_density = np.abs(wavefunction)**2
        
        mean_pos = np.sum(positions * prob_density)
        variance = np.sum((positions - mean_pos)**2 * prob_density)
        xi = np.sqrt(variance)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return xi


def calculate_participation_ratio(wavefunction):
    """
    计算参与率（Participation Ratio, PR）
    
    PR = 1/IPR = 1/(∑ᵢ|ψᵢ|⁴)
    
    物理意义：
    - 表示波函数"参与"的有效格点数
    - 扩展态：PR ~ N
    - 局域态：PR ~ O(1)
    
    Parameters:
    -----------
    wavefunction : list or array-like
        波函数
        
    Returns:
    --------
    pr : float
        参与率
    """
    # calculate_inverse_participation_ratio内部会转换为numpy数组
    ipr = calculate_inverse_participation_ratio(wavefunction)
    return 1.0 / ipr if ipr > 0 else 0.0


def analyze_localization_disorder_average(N, t, disorder_strengths, num_samples=10, 
                                          energy_window=None, periodic=False):
    """
    分析Anderson局域化：对多个无序样本求平均
    
    对给定的无序强度列表，生成多个随机样本并计算平均IPR
    
    Parameters:
    -----------
    N : int
        系统格点数
    t : float
        跃迁积分
    disorder_strengths : list or array-like
        无序强度列表（W值）
    num_samples : int, default=10
        每个W值的随机样本数
    energy_window : tuple, optional
        能量窗口(E_min, E_max)，只统计此范围内的态
        如果为None，使用能带中心附近的态
    periodic : bool, default=False
        边界条件
        
    Returns:
    --------
    results : dict
        包含以下键：
        - 'disorder_strengths': W值列表
        - 'mean_ipr': 平均IPR（对样本和能量窗口内的态）
        - 'std_ipr': IPR标准差
        - 'mean_pr': 平均参与率
        - 'num_states_analyzed': 分析的态数量
        
    Examples:
    ---------
    >>> W_values = [0, 1, 2, 4, 8]
    >>> results = analyze_localization_disorder_average(
    ...     N=100, t=1.0, disorder_strengths=W_values, num_samples=10
    ... )
    >>> print(f"W=4时的平均IPR: {results['mean_ipr'][3]:.6f}")
    """
    # 转换为numpy数组
    disorder_strengths = np.asarray(disorder_strengths)
    
    results = {
        'disorder_strengths': disorder_strengths.copy(),
        'mean_ipr': [],
        'std_ipr': [],
        'mean_pr': [],
        'num_states_analyzed': []
    }
    
    for W in disorder_strengths:
        ipr_all_samples = []
        
        print(f"  处理 W = {W:.1f}:")
        
        for sample in range(num_samples):
            # 构建哈密顿量
            H, potential = construct_anderson_hamiltonian_1d(
                N, t=t, disorder_strength=W, periodic=periodic, seed=sample
            )
            
            # 求解本征态
            if N <= 500:  # 小系统用密集矩阵
                H_dense = H.toarray()
                eigenvalues, eigenvectors = la.eigh(H_dense)
            else:  # 大系统用稀疏矩阵
                k = min(int(N * 0.5), 200)  # 求解一半的态
                try:
                    eigenvalues, eigenvectors = spla.eigsh(H, k=k, which='SA')
                except:
                    # 如果稀疏求解失败，回退到密集矩阵
                    H_dense = H.toarray()
                    eigenvalues, eigenvectors = la.eigh(H_dense)
            
            # 选择能量窗口内的态
            if energy_window is None:
                # 默认选择能带中心附近的态（约1/4）
                n_states = len(eigenvalues)
                center_idx = n_states // 2
                window_size = max(n_states // 4, 10)
                start_idx = max(0, center_idx - window_size // 2)
                end_idx = min(n_states, center_idx + window_size // 2)
                state_indices = range(start_idx, end_idx)
            else:
                E_min, E_max = energy_window
                state_indices = np.where((eigenvalues >= E_min) & (eigenvalues <= E_max))[0]
            
            # 计算这些态的IPR
            for idx in state_indices:
                psi = eigenvectors[:, idx]
                ipr = calculate_inverse_participation_ratio(psi)
                ipr_all_samples.append(ipr)
            
            if sample == 0:
                print(f"    样本 {sample+1}: 分析 {len(state_indices)} 个态")
        
        # 统计结果
        ipr_array = np.array(ipr_all_samples)
        results['mean_ipr'].append(np.mean(ipr_array))
        results['std_ipr'].append(np.std(ipr_array))
        results['mean_pr'].append(np.mean(1.0 / ipr_array))
        results['num_states_analyzed'].append(len(ipr_all_samples))
        
        print(f"    → 平均IPR = {results['mean_ipr'][-1]:.6f} ± {results['std_ipr'][-1]:.6f}")
    
    # 转换为数组
    for key in ['mean_ipr', 'std_ipr', 'mean_pr', 'num_states_analyzed']:
        results[key] = np.array(results[key])
    
    return results


def calculate_dos_anderson(N, t, disorder_strength, num_samples=10, num_bins=100):
    """
    计算Anderson模型的态密度（Density of States, DOS）
    
    对多个随机样本求平均，得到无序平均的态密度
    
    Parameters:
    -----------
    N : int
        系统格点数
    t : float
        跃迁积分
    disorder_strength : float
        无序强度
    num_samples : int, default=10
        样本数
    num_bins : int, default=100
        能量直方图的bin数
        
    Returns:
    --------
    energy_bins : np.ndarray
        能量bin中心
    dos : np.ndarray
        态密度
    """
    all_eigenvalues = []
    
    for sample in range(num_samples):
        H, _ = construct_anderson_hamiltonian_1d(
            N, t=t, disorder_strength=disorder_strength, seed=sample
        )
        
        if N <= 500:
            H_dense = H.toarray()
            eigenvalues = la.eigvalsh(H_dense)
        else:
            k = min(int(N * 0.8), 500)
            eigenvalues = spla.eigsh(H, k=k, which='SA', return_eigenvectors=False)
        
        all_eigenvalues.extend(eigenvalues)
    
    # 计算直方图
    hist, bin_edges = np.histogram(all_eigenvalues, bins=num_bins, density=True)
    energy_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return energy_bins, hist


def identify_mobility_edge(N, t, disorder_strength, num_samples=10, ipr_threshold=0.1):
    """
    识别迁移率边界（Mobility Edge）
    
    在某些系统中，能谱中部分态局域，部分态扩展，分界线称为迁移率边界。
    （注：1D系统中所有态都局域化，此函数主要用于教学和3D系统）
    
    Parameters:
    -----------
    N : int
        系统格点数
    t : float
        跃迁积分
    disorder_strength : float
        无序强度
    num_samples : int, default=10
        样本数
    ipr_threshold : float, default=0.1
        IPR阈值，IPR < threshold视为扩展态
        
    Returns:
    --------
    results : dict
        包含能量和对应的IPR
    """
    energy_ipr_pairs = []
    
    for sample in range(num_samples):
        H, _ = construct_anderson_hamiltonian_1d(
            N, t=t, disorder_strength=disorder_strength, seed=sample
        )
        
        if N <= 200:
            H_dense = H.toarray()
            eigenvalues, eigenvectors = la.eigh(H_dense)
        else:
            k = min(int(N * 0.5), 200)
            eigenvalues, eigenvectors = spla.eigsh(H, k=k, which='SA')
        
        for i, E in enumerate(eigenvalues):
            psi = eigenvectors[:, i]
            ipr = calculate_inverse_participation_ratio(psi)
            energy_ipr_pairs.append((E, ipr))
    
    # 排序
    energy_ipr_pairs.sort(key=lambda x: x[0])
    
    energies = np.array([e for e, _ in energy_ipr_pairs])
    iprs = np.array([ipr for _, ipr in energy_ipr_pairs])
    
    return {
        'energies': energies,
        'iprs': iprs,
        'localized_mask': iprs > ipr_threshold,
        'extended_mask': iprs <= ipr_threshold
    }


# ============================================================================
# 便捷函数
# ============================================================================

def quick_anderson_analysis(N=100, t=1.0, W_values=[0, 1, 2, 4, 8], 
                            num_samples=10, verbose=True):
    """
    快速Anderson局域化分析（便捷函数）
    
    Parameters:
    -----------
    N : int, default=100
        格点数
    t : float, default=1.0
        跃迁积分
    W_values : list, default=[0,1,2,4,8]
        无序强度列表
    num_samples : int, default=10
        每个W的样本数
    verbose : bool, default=True
        是否打印详细信息
        
    Returns:
    --------
    results : dict
        分析结果
        
    Examples:
    ---------
    >>> results = quick_anderson_analysis(N=100, W_values=[0,2,4,8])
    >>> print(f"W=4时的IPR: {results['mean_ipr'][2]:.6f}")
    """
    if verbose:
        print("="*80)
        print("Anderson局域化快速分析")
        print("="*80)
        print(f"系统参数：N={N}, t={t}")
        print(f"扫描无序强度：W = {W_values}")
        print(f"每个W的样本数：{num_samples}")
        print("="*80)
    
    results = analyze_localization_disorder_average(
        N=N, t=t, 
        disorder_strengths=W_values, 
        num_samples=num_samples,
        periodic=False
    )
    
    if verbose:
        print("\n结果总结：")
        print(f"{'W':<10} {'平均IPR':<15} {'标准差':<15} {'平均PR':<15}")
        print("-"*55)
        for i, W in enumerate(results['disorder_strengths']):
            print(f"{W:<10.1f} {results['mean_ipr'][i]:<15.6f} "
                  f"{results['std_ipr'][i]:<15.6f} {results['mean_pr'][i]:<15.2f}")
    
    return results


def example_usage():
    """示例：演示如何使用本模块"""
    print("="*80)
    print("无序系统与Anderson局域化模块 - 使用示例")
    print("="*80)
    
    # 示例1：构建Anderson哈密顿量
    print("\n示例1：构建Anderson哈密顿量")
    N = 50
    t = 1.0
    W = 2.0
    H, potential = construct_anderson_hamiltonian_1d(N, t, W, seed=42)
    print(f"  系统大小：N = {N}")
    print(f"  哈密顿量形状：{H.shape}")
    print(f"  随机势范围：[{potential.min():.3f}, {potential.max():.3f}]")
    
    # 示例2：计算单个态的IPR
    print("\n示例2：计算逆参与率（IPR）")
    if N <= 100:
        eigenvalues, eigenvectors = la.eigh(H.toarray())
        center_state = eigenvectors[:, N//2]
        ipr = calculate_inverse_participation_ratio(center_state)
        pr = calculate_participation_ratio(center_state)
        print(f"  能带中心态的IPR = {ipr:.6f}")
        print(f"  参与率 PR = {pr:.2f}")
        print(f"  参与格点数占比 = {pr/N*100:.1f}%")
    
    # 示例3：快速分析
    print("\n示例3：快速Anderson局域化分析")
    W_values = [0, 2, 4]
    results = quick_anderson_analysis(N=50, W_values=W_values, num_samples=5, verbose=False)
    
    print(f"  W=0 (无无序): IPR = {results['mean_ipr'][0]:.6f}")
    print(f"  W=2 (弱无序): IPR = {results['mean_ipr'][1]:.6f}")
    print(f"  W=4 (强无序): IPR = {results['mean_ipr'][2]:.6f}")
    
    print("\n✓ 示例完成！")


if __name__ == "__main__":
    example_usage()

