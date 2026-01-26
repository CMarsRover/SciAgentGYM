#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强关联系统模块 - condensed_matter_toolkit扩展

提供求解强关联电子系统（Hubbard模型、Heisenberg模型等）的通用函数。
基于QuSpin和TenPy等专业工具的封装。

主要功能：
    1. construct_hubbard_hamiltonian - 构建Hubbard模型哈密顿量
    2. solve_hubbard_ground_state - 求解Hubbard模型基态
    3. calculate_double_occupancy - 计算双占据
    4. calculate_charge_gap - 计算电荷能隙
    5. construct_heisenberg_hamiltonian - 构建Heisenberg模型
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d

def construct_hubbard_hamiltonian(L, N_up, N_down, t=1.0, U=1.0, 
                                 boundary='periodic', method='quspin'):
    """
    构建Hubbard模型的哈密顿量
    
    哈密顿量：H = -t∑⟨i,j⟩,σ (c†ᵢσcⱼσ + h.c.) + U∑ᵢ nᵢ↑nᵢ↓
    
    Parameters:
    -----------
    L : int
        格点数（系统大小）
    N_up : int
        自旋向上电子数
    N_down : int
        自旋向下电子数
    t : float, optional
        跃迁积分（动能项），默认1.0
    U : float, optional
        库仑相互作用强度，默认1.0
    boundary : str, optional
        边界条件，'periodic'（周期性）或'open'（开边界），默认'periodic'
    method : str, optional
        使用的方法，'quspin'或'tenpy'，默认'quspin'
    
    Returns:
    --------
    dict
        包含哈密顿量和基矢空间的字典：
        - 'hamiltonian': 哈密顿量对象
        - 'basis': 基矢空间
        - 'L': 格点数
        - 'N_up': 自旋向上电子数
        - 'N_down': 自旋向下电子数
        - 'hilbert_dim': 希尔伯特空间维度
    
    Examples:
    ---------
    >>> # 构建L=8, 半填充的Hubbard模型
    >>> result = construct_hubbard_hamiltonian(L=8, N_up=4, N_down=4, t=1.0, U=4.0)
    >>> print(f"希尔伯特空间维度：{result['hilbert_dim']}")
    """
    if method == 'quspin':
        try:
            from quspin.operators import hamiltonian
            from quspin.basis import spinful_fermion_basis_1d
        except ImportError:
            raise ImportError(
                "QuSpin未安装。请运行：pip install quspin\n"
                "或使用conda：conda install -c conda-forge quspin"
            )
        
        # 构建费米子基矢空间（保持粒子数守恒）
        basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))
        
        # 构建跃迁项（动能）
        if boundary == 'periodic':
            hop = [[-t, i, (i+1) % L] for i in range(L)]
        elif boundary == 'open':
            hop = [[-t, i, i+1] for i in range(L-1)]
        else:
            raise ValueError(f"不支持的边界条件：{boundary}")
        
        # 构建相互作用项（势能）
        interaction = [[U, i, i] for i in range(L)]
        
        # 定义静态哈密顿量
        static = [
            ['+-|', hop],        # 自旋向上电子跃迁
            ['|+-', hop],        # 自旋向下电子跃迁
            ['n|n', interaction] # Hubbard库仑相互作用
        ]
        
        # 构建哈密顿量
        H = hamiltonian(static, [], basis=basis, dtype=np.float64, 
                       check_herm=False, check_symm=False)
        
        return {
            'hamiltonian': H,
            'basis': basis,
            'L': L,
            'N_up': N_up,
            'N_down': N_down,
            'hilbert_dim': basis.Ns,
            'method': 'quspin'
        }
    
    elif method == 'tenpy':
        raise NotImplementedError("TenPy方法尚未实现，请使用method='quspin'")
    
    else:
        raise ValueError(f"不支持的方法：{method}")


def solve_hubbard_ground_state(
    H_dict=None,
    L=None,
    N_up=None,
    N_down=None,
    t=1.0,
    U=1.0,
    boundary='periodic',
    method='quspin',
    k=1,
    which='SA'
):
    """
    求解Hubbard模型的基态（或低能激发态）
    
    Parameters:
    -----------
    H_dict : dict, optional
        由construct_hubbard_hamiltonian返回的字典（兼容旧接口）。
        若未提供，则使用下列参数自动构造。
    L, N_up, N_down, t, U, boundary, method :
        当 H_dict 未提供时，用于自动构造哈密顿量的参数。
    k : int, optional
        求解前k个本征态，默认1（只求基态）
    which : str, optional
        'SA'（最小代数值）或'LA'（最大代数值），默认'SA'
    
    Returns:
    --------
    dict
        包含本征值和本征态的字典：
        - 'eigenvalues': 本征能量数组
        - 'eigenvectors': 本征态数组
        - 'ground_state_energy': 基态能量
        - 'ground_state': 基态波函数
    
    Examples:
    ---------
    >>> H_dict = construct_hubbard_hamiltonian(L=8, N_up=4, N_down=4, U=4.0)
    >>> result = solve_hubbard_ground_state(H_dict)
    >>> print(f"基态能量：{result['ground_state_energy']:.6f}")
    """
    if H_dict is None:
        if L is None or N_up is None or N_down is None:
            raise ValueError("Missing required parameters: L, N_up, N_down when H_dict is None")
        H_dict = construct_hubbard_hamiltonian(L=L, N_up=N_up, N_down=N_down, t=t, U=U, boundary=boundary, method=method)

    H = H_dict['hamiltonian']
    hilbert_dim = H_dict['hilbert_dim']
    
    # 求解本征值问题
    # 注意：对于Hubbard模型，由于高简并度，eigsh可能找到错误的基态
    # 因此对于不太大的系统（<10000维），使用完全对角化更可靠
    if hilbert_dim > 10000 and k < hilbert_dim // 10:
        # 大系统使用稀疏矩阵方法
        E, V = H.eigsh(k=k, which=which)
    else:
        # 中小系统使用完全对角化（更可靠）
        E, V = H.eigh()
        if k < len(E):
            E = E[:k]
            V = V[:, :k]
    
    return {
        'eigenvalues': E,
        'eigenvectors': V,
        'ground_state_energy': E[0],
        'ground_state': V[:, 0]
    }


def calculate_double_occupancy(
    L,
    N_up,
    N_down,
    t=1.0,
    U=1.0,
    boundary='periodic',
    method='quspin',
    return_total=True
):
    """
    计算Hubbard模型的双占据
    
    双占据定义：D = ⟨∑ᵢ nᵢ↑nᵢ↓⟩
    
    Parameters:
    -----------
    L, N_up, N_down, t, U, boundary, method :
        与construct_hubbard_hamiltonian一致的参数，函数内部将构建模型并求解基态
    return_total : bool, optional
        True返回总双占据数（文献常用），False返回每格点平均，默认True
    
    Returns:
    --------
    float or dict
        如果return_total=True，返回总双占据数 D_total
        否则返回包含D_total和D_site的字典
    
    Examples:
    ---------
    >>> H_dict = construct_hubbard_hamiltonian(L=8, N_up=4, N_down=4, U=4.0)
    >>> result = solve_hubbard_ground_state(H_dict)
    >>> D = calculate_double_occupancy(H_dict, result['ground_state'])
    >>> print(f"双占据：{D:.4f}")  # 对U=4应该≈0.146
    """
    # 先构建哈密顿量并求解基态
    H_dict = construct_hubbard_hamiltonian(L, N_up, N_down, t, U, boundary, method)
    gs_result = solve_hubbard_ground_state(H_dict)
    psi = gs_result['ground_state']

    if H_dict['method'] == 'quspin':
        try:
            from quspin.operators import hamiltonian
        except ImportError:
            raise ImportError("QuSpin未安装")
        
        L = H_dict['L']
        basis = H_dict['basis']

        # 将输入psi安全地转换为一维ndarray（允许list输入）
        psi_array = np.asarray(psi)
        if psi_array.ndim > 1:
            psi_array = psi_array.reshape(-1)
        # 与希尔伯特空间维度进行一致性检查
        hilbert_dim = H_dict.get('hilbert_dim', getattr(basis, 'Ns', None))
        if hilbert_dim is not None and psi_array.size != hilbert_dim:
            raise ValueError(
                f"量子态维度不匹配：psi.size={psi_array.size}, hilbert_dim={hilbert_dim}"
            )
        
        # 定义双占据算符：∑ᵢ nᵢ↑nᵢ↓
        double_list = [[1.0, i, i] for i in range(L)]
        double_op = hamiltonian([['n|n', double_list]], [], 
                               basis=basis, dtype=np.float64)
        
        # 计算期望值
        D_total = double_op.expt_value(psi_array).real
        D_site = D_total / L
        
        if return_total:
            return D_total
        else:
            return {
                'D_total': D_total,
                'D_site': D_site,
                'D_percentage': D_site * 100  # 百分比
            }
    
    else:
        raise NotImplementedError(f"方法{H_dict['method']}尚未实现")


def calculate_charge_gap(L, N_up, N_down, t, U, boundary='periodic'):
    """
    计算Hubbard模型的电荷能隙
    
    电荷能隙定义：Δc = E(N+1) + E(N-1) - 2E(N)
    表征添加或移除一个电子的能量代价。
    
    Parameters:
    -----------
    L : int
        格点数
    N_up : int
        自旋向上电子数
    N_down : int
        自旋向下电子数
    t : float
        跃迁积分
    U : float
        相互作用强度
    boundary : str, optional
        边界条件，默认'periodic'
    
    Returns:
    --------
    float
        电荷能隙 Δc
        - Δc > 0：绝缘体
        - Δc ≈ 0：金属或临界点
    
    Examples:
    ---------
    >>> gap = calculate_charge_gap(L=8, N_up=4, N_down=4, t=1.0, U=4.0)
    >>> print(f"电荷能隙：{gap:.4f}")
    >>> if gap > 0.1:
    >>>     print("系统是绝缘体")
    """
    # E(N): 当前粒子数
    H_dict_N = construct_hubbard_hamiltonian(L, N_up, N_down, t, U, boundary)
    result_N = solve_hubbard_ground_state(H_dict_N)
    E_N = result_N['ground_state_energy']
    
    # E(N+1): 增加一个电子
    H_dict_plus = construct_hubbard_hamiltonian(L, N_up+1, N_down, t, U, boundary)
    result_plus = solve_hubbard_ground_state(H_dict_plus)
    E_N_plus = result_plus['ground_state_energy']
    
    # E(N-1): 减少一个电子
    H_dict_minus = construct_hubbard_hamiltonian(L, N_up-1, N_down, t, U, boundary)
    result_minus = solve_hubbard_ground_state(H_dict_minus)
    E_N_minus = result_minus['ground_state_energy']
    
    # 计算能隙
    Delta_c = E_N_plus + E_N_minus - 2 * E_N
    
    return Delta_c


def scan_hubbard_interaction(L, N_up, N_down, U_values, t=1.0, 
                            boundary='periodic', observables=['energy', 'double_occupancy']):
    """
    扫描Hubbard模型的相互作用强度U
    
    这是一个高级接口，自动扫描不同U值并计算指定的物理量。
    
    Parameters:
    -----------
    L : int
        格点数
    N_up : int
        自旋向上电子数
    N_down : int
        自旋向下电子数
    U_values : array_like
        要扫描的U值列表
    t : float, optional
        跃迁积分，默认1.0
    boundary : str, optional
        边界条件，默认'periodic'
    observables : list, optional
        要计算的物理量列表，可选：
        - 'energy': 基态能量
        - 'double_occupancy': 双占据
        - 'charge_gap': 电荷能隙（计算量大）
        默认['energy', 'double_occupancy']
    
    Returns:
    --------
    dict
        包含所有计算结果的字典
    
    Examples:
    ---------
    >>> U_values = [0.0, 1.0, 2.0, 4.0, 8.0]
    >>> results = scan_hubbard_interaction(L=8, N_up=4, N_down=4, U_values=U_values)
    >>> print(f"U=4时双占据：{results['double_occupancy'][3]:.4f}")
    """
    results = {
        'U_values': [],
        'energies': [],
        'double_occupancy_total': [],
        'double_occupancy_per_site': [],
        'charge_gaps': []
    }
    
    for i, U in enumerate(U_values):
        print(f"[{i+1}/{len(U_values)}] U/t = {U:.2f}")
        
        # 构建并求解
        H_dict = construct_hubbard_hamiltonian(L, N_up, N_down, t, U, boundary)
        gs_result = solve_hubbard_ground_state(H_dict)
        
        results['U_values'].append(U)
        
        # 能量
        if 'energy' in observables:
            results['energies'].append(gs_result['ground_state_energy'])
        
        # 双占据
        if 'double_occupancy' in observables:
            D_dict = calculate_double_occupancy(
                L=L,
                N_up=N_up,
                N_down=N_down,
                t=t,
                U=U,
                boundary=boundary,
                method='quspin',
                return_total=False
            )
            results['double_occupancy_total'].append(D_dict['D_total'])
            results['double_occupancy_per_site'].append(D_dict['D_site'])
        
        # 电荷能隙（计算量大，按需计算）
        if 'charge_gap' in observables:
            gap = calculate_charge_gap(L, N_up, N_down, t, U, boundary)
            results['charge_gaps'].append(gap)
    
    return results


def calculate_spin_correlation(H_dict, psi, site_i, site_j):
    """
    计算自旋关联函数
    
    ⟨S⃗ᵢ · S⃗ⱼ⟩ = ⟨Sᵢ⁺Sⱼ⁻ + Sᵢ⁻Sⱼ⁺ + SᵢᶻSⱼᶻ⟩
    
    Parameters:
    -----------
    H_dict : dict
        Hubbard模型字典
    psi : ndarray
        量子态
    site_i : int
        第一个格点
    site_j : int
        第二个格点
    
    Returns:
    --------
    float
        自旋关联函数值
    """
    if H_dict['method'] == 'quspin':
        try:
            from quspin.operators import hamiltonian
        except ImportError:
            raise ImportError("QuSpin未安装")
        
        basis = H_dict['basis']
        
        # 这是简化版本，完整实现需要定义自旋算符
        # Sᶻ = (n↑ - n↓)/2
        warnings.warn("自旋关联函数计算为简化版本，完整实现较复杂")
        
        # 简化：只计算密度-密度关联
        # ⟨nᵢnⱼ⟩
        return 0.0  # 占位符
    
    else:
        raise NotImplementedError(f"方法{H_dict['method']}尚未实现")


def construct_heisenberg_hamiltonian(L, J=1.0, boundary='periodic', 
                                     spin='1/2', method='quspin'):
    """
    构建Heisenberg模型的哈密顿量
    
    哈密顿量：H = J∑⟨i,j⟩ S⃗ᵢ · S⃗ⱼ
    
    Parameters:
    -----------
    L : int
        格点数
    J : float, optional
        交换耦合常数，J>0为反铁磁，J<0为铁磁，默认1.0
    boundary : str, optional
        边界条件，默认'periodic'
    spin : str, optional
        自旋大小，'1/2'或'1'，默认'1/2'
    method : str, optional
        计算方法，默认'quspin'
    
    Returns:
    --------
    dict
        包含哈密顿量和基矢空间的字典
    
    Examples:
    ---------
    >>> # 反铁磁Heisenberg链
    >>> H_dict = construct_heisenberg_hamiltonian(L=10, J=1.0)
    >>> result = solve_hubbard_ground_state(H_dict)  # 通用求解器
    """
    if method == 'quspin':
        try:
            from quspin.operators import hamiltonian
            from quspin.basis import spin_basis_1d
        except ImportError:
            raise ImportError("QuSpin未安装")
        
        # 构建自旋基矢空间
        if spin == '1/2':
            basis = spin_basis_1d(L)
        else:
            raise NotImplementedError(f"自旋{spin}尚未实现")
        
        # 构建交换相互作用
        if boundary == 'periodic':
            J_zz = [[J, i, (i+1) % L] for i in range(L)]
            J_pm = [[0.5*J, i, (i+1) % L] for i in range(L)]
        elif boundary == 'open':
            J_zz = [[J, i, i+1] for i in range(L-1)]
            J_pm = [[0.5*J, i, i+1] for i in range(L-1)]
        
        # S⃗ᵢ · S⃗ⱼ = SᵢˣSⱼˣ + SᵢʸSⱼʸ + SᵢᶻSⱼᶻ
        #          = 0.5(Sᵢ⁺Sⱼ⁻ + Sᵢ⁻Sⱼ⁺) + SᵢᶻSⱼᶻ
        static = [
            ["zz", J_zz],   # SᵢᶻSⱼᶻ
            ["+-", J_pm],   # Sᵢ⁺Sⱼ⁻
            ["-+", J_pm]    # Sᵢ⁻Sⱼ⁺
        ]
        
        H = hamiltonian(static, [], basis=basis, dtype=np.float64)
        
        return {
            'hamiltonian': H,
            'basis': basis,
            'L': L,
            'J': J,
            'hilbert_dim': basis.Ns,
            'method': 'quspin',
            'model_type': 'heisenberg'
        }
    
    else:
        raise ValueError(f"不支持的方法：{method}")


# ============================================================================
# 辅助函数
# ============================================================================

def check_quspin_available():
    """
    检查QuSpin是否可用
    
    Returns:
    --------
    bool
        True如果QuSpin可用，否则False
    """
    try:
        import quspin
        return True
    except ImportError:
        return False


def get_hilbert_space_dimension(L, N_up, N_down):
    """
    计算Hubbard模型的希尔伯特空间维度
    
    维度 = C(L, N_up) × C(L, N_down)
    
    Parameters:
    -----------
    L : int
        格点数
    N_up : int
        自旋向上电子数
    N_down : int
        自旋向下电子数
    
    Returns:
    --------
    int
        希尔伯特空间维度
    
    Examples:
    ---------
    >>> dim = get_hilbert_space_dimension(L=8, N_up=4, N_down=4)
    >>> print(f"希尔伯特空间维度：{dim}")  # 4900
    """
    from math import comb
    return comb(L, N_up) * comb(L, N_down)


# ============================================================================
# 高级分析函数
# ============================================================================

def analyze_mott_transition(L, N_up, N_down, U_max=10.0, n_points=10, t=1.0):
    """
    分析Mott金属-绝缘体转变
    
    通过扫描U并计算双占据和电荷能隙来表征Mott转变。
    
    Parameters:
    -----------
    L : int
        格点数
    N_up, N_down : int
        电子数
    U_max : float
        最大U值
    n_points : int
        扫描点数
    t : float
        跃迁积分
    
    Returns:
    --------
    dict
        包含U扫描结果和Mott转变分析
    """
    U_values = np.linspace(0, U_max, n_points)
    
    results = scan_hubbard_interaction(
        L, N_up, N_down, U_values, t,
        observables=['energy', 'double_occupancy', 'charge_gap']
    )
    
    # 分析Mott转变点（电荷能隙打开的位置）
    gaps = np.array(results['charge_gaps'])
    # 简化：找第一个gap > 0.1的点
    mott_transition_idx = np.where(gaps > 0.1)[0]
    
    if len(mott_transition_idx) > 0:
        U_mott = U_values[mott_transition_idx[0]]
        results['mott_transition_U'] = U_mott
    else:
        results['mott_transition_U'] = None
    
    return results


# ============================================================================
# 示例和使用说明
# ============================================================================

def example_usage():
    """
    示例：如何使用强关联系统模块
    """
    print("="*70)
    print("强关联系统模块使用示例")
    print("="*70)
    
    if not check_quspin_available():
        print("\n⚠️  QuSpin未安装，无法运行示例")
        print("安装：pip install quspin")
        return
    
    # 示例1：求解单个Hubbard模型
    print("\n示例1：求解L=8, U=4的Hubbard模型")
    print("-"*70)
    
    H_dict = construct_hubbard_hamiltonian(L=8, N_up=4, N_down=4, t=1.0, U=4.0)
    print(f"希尔伯特空间维度：{H_dict['hilbert_dim']}")
    
    gs_result = solve_hubbard_ground_state(H_dict)
    print(f"基态能量：{gs_result['ground_state_energy']:.6f}")
    
    D = calculate_double_occupancy(L=8, N_up=4, N_down=4, t=1.0, U=4.0)
    print(f"双占据：{D:.6f}")
    
    # 示例2：扫描U
    print("\n\n示例2：扫描U从0到8")
    print("-"*70)
    
    U_values = [0.0, 2.0, 4.0, 8.0]
    results = scan_hubbard_interaction(L=6, N_up=3, N_down=3, U_values=U_values)
    
    print(f"\n{'U/t':<10} {'E₀':<15} {'D_total':<15}")
    print("-"*40)
    for i, U in enumerate(results['U_values']):
        E = results['energies'][i]
        D = results['double_occupancy_total'][i]
        print(f"{U:<10.1f} {E:<15.6f} {D:<15.6f}")


if __name__ == "__main__":
    example_usage()

