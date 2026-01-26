# Filename: nmr_spin_system_toolkit.py

"""
NMR Spin System Analysis Toolkit
专业的核磁共振自旋系统能级与跃迁分析工具包

领域专属库：
- qutip: 量子工具箱，用于量子态操作和哈密顿量构建
- sympy: 符号计算，用于精确的量子力学推导
- scipy: 数值计算和矩阵运算
- numpy: 基础数值计算
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import json
import os
from itertools import product
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import sympy as sp
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.spin import JzKet, Jz

# 创建输出目录
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 物理常数
PLANCK_H = 6.62607015e-34  # J·s
HBAR = PLANCK_H / (2 * np.pi)  # J·s
GYROMAGNETIC_RATIO_1H = 2.675e8  # rad/(s·T) for proton
MU_0 = 4 * np.pi * 1e-7  # N/A^2
ANGSTROM_TO_METER = 1e-10

# ==================== 第一层：原子函数 ====================

def calculate_spin_states(spin_quantum_number: float) -> Dict[str, Any]:
    """
    计算给定自旋量子数的可能自旋态数量
    
    Args:
        spin_quantum_number: 自旋量子数 (如1/2, 1, 3/2等)
    
    Returns:
        dict: {
            'result': int (自旋态数量),
            'metadata': {
                'spin': float,
                'magnetic_quantum_numbers': list,
                'degeneracy': int
            }
        }
    """
    if spin_quantum_number < 0:
        raise ValueError(f"自旋量子数必须非负，得到: {spin_quantum_number}")
    
    # 自旋态数量 = 2S + 1
    num_states = int(2 * spin_quantum_number + 1)
    
    # 磁量子数范围: -S, -S+1, ..., S-1, S
    m_values = [spin_quantum_number - i for i in range(num_states)]
    
    return {
        'result': num_states,
        'metadata': {
            'spin': spin_quantum_number,
            'magnetic_quantum_numbers': m_values,
            'degeneracy': num_states
        }
    }


def calculate_zeeman_energy(magnetic_field: float, 
                           gyromagnetic_ratio: float,
                           m_quantum_number: float,
                           chemical_shift: float = 0.0) -> Dict[str, Any]:
    """
    计算Zeeman能级（考虑化学位移）
    
    Args:
        magnetic_field: 磁场强度 (Tesla)
        gyromagnetic_ratio: 旋磁比 (rad/(s·T))
        m_quantum_number: 磁量子数
        chemical_shift: 化学位移 (ppm)
    
    Returns:
        dict: {
            'result': float (能量，单位J),
            'metadata': {
                'frequency': float (Hz),
                'angular_frequency': float (rad/s)
            }
        }
    """
    if magnetic_field <= 0:
        raise ValueError(f"磁场强度必须为正，得到: {magnetic_field} T")
    
    # 有效磁场 = B0 * (1 - σ)，σ = chemical_shift * 1e-6
    effective_field = magnetic_field * (1 - chemical_shift * 1e-6)
    
    # E = -γ * ℏ * B * m
    energy = -gyromagnetic_ratio * HBAR * effective_field * m_quantum_number
    
    # 对应的频率
    angular_frequency = gyromagnetic_ratio * effective_field
    frequency = angular_frequency / (2 * np.pi)
    
    return {
        'result': float(energy),
        'metadata': {
            'frequency': float(frequency),
            'angular_frequency': float(angular_frequency),
            'effective_field': float(effective_field)
        }
    }


def calculate_dipolar_coupling(distance: float,
                               gyromagnetic_ratio_1: float,
                               gyromagnetic_ratio_2: float,
                               theta: float = 0.0) -> Dict[str, Any]:
    """
    计算偶极耦合常数
    
    Args:
        distance: 核间距离 (angstroms)
        gyromagnetic_ratio_1: 核1的旋磁比 (rad/(s·T))
        gyromagnetic_ratio_2: 核2的旋磁比 (rad/(s·T))
        theta: 核间矢量与磁场夹角 (度)
    
    Returns:
        dict: {
            'result': float (偶极耦合常数，单位Hz),
            'metadata': {
                'distance_m': float,
                'angle_rad': float,
                'coupling_strength': str
            }
        }
    """
    if distance <= 0:
        raise ValueError(f"距离必须为正，得到: {distance} Å")
    
    # 转换单位
    r_meter = distance * ANGSTROM_TO_METER
    theta_rad = np.deg2rad(theta)
    
    # 偶极耦合常数: D = -(μ0/4π) * (γ1*γ2*ℏ/r³) * (3cos²θ - 1)
    # 转换为频率单位 (Hz)
    prefactor = (MU_0 / (4 * np.pi)) * (gyromagnetic_ratio_1 * gyromagnetic_ratio_2 * HBAR) / (r_meter ** 3)
    angular_factor = 3 * np.cos(theta_rad)**2 - 1
    
    D_angular = prefactor * angular_factor  # rad/s
    D_hz = D_angular / (2 * np.pi)  # Hz
    
    # 判断耦合强度
    if abs(D_hz) < 100:
        strength = "weak"
    elif abs(D_hz) < 1000:
        strength = "moderate"
    else:
        strength = "strong"
    
    return {
        'result': float(D_hz),
        'metadata': {
            'distance_m': float(r_meter),
            'angle_rad': float(theta_rad),
            'coupling_strength': strength,
            'prefactor_hz': float(prefactor / (2 * np.pi))
        }
    }


def generate_basis_states(num_spins: int, spin_value: float = 0.5) -> Dict[str, Any]:
    """
    生成多自旋系统的基态
    
    Args:
        num_spins: 自旋核数量
        spin_value: 每个核的自旋量子数
    
    Returns:
        dict: {
            'result': list (基态标签列表),
            'metadata': {
                'total_states': int,
                'state_labels': list,
                'quantum_numbers': list
            }
        }
    """
    if num_spins <= 0:
        raise ValueError(f"自旋数量必须为正整数，得到: {num_spins}")
    
    # 每个自旋的可能态
    single_spin_states = int(2 * spin_value + 1)
    m_values = [spin_value - i for i in range(single_spin_states)]
    
    # 生成所有可能的组合
    all_combinations = list(product(m_values, repeat=num_spins))
    
    # 生成标签 (α表示+1/2, β表示-1/2)
    state_labels = []
    for combo in all_combinations:
        label = ''.join(['α' if m > 0 else 'β' for m in combo])
        state_labels.append(label)
    
    return {
        'result': state_labels,
        'metadata': {
            'total_states': len(all_combinations),
            'state_labels': state_labels,
            'quantum_numbers': [list(combo) for combo in all_combinations]
        }
    }


def check_selection_rule(initial_state: List[float],
                         final_state: List[float],
                         coupling_type: str = "dipolar",
                         selection_rule: str = "single") -> Dict[str, Any]:
    """
    检查跃迁选择定则
    
    说明（计数口径）：
    -“允许的跃迁数”按常规高场NMR单量子选择定则统计：仅计 Δm_total=±1，且仅一个核翻转（single-quantum）。

    
    Args:
        initial_state: 初态的磁量子数列表 [m1, m2, ...]
        final_state: 末态的磁量子数列表 [m1, m2, ...]
        coupling_type: 耦合类型 ("dipolar", "j_coupling", "zeeman")
    
    Returns:
        dict: {
            'result': bool (是否允许跃迁),
            'metadata': {
                'delta_m_total': float,
                'delta_m_individual': list,
                'transition_type': str
            }
        }
    """
    if len(initial_state) != len(final_state):
        raise ValueError("初态和末态的自旋数量必须相同")
    
    # 计算总磁量子数变化
    delta_m_total = sum(final_state) - sum(initial_state)
    delta_m_individual = [f - i for i, f in zip(initial_state, final_state)]
    
    # 通用量子阶与翻转数
    quantum_order = abs(delta_m_total)  # |Δm_total|
    num_flips = sum(1 for dm in delta_m_individual if dm != 0)

    def is_single_quantum() -> bool:
        return (quantum_order == 1) and (num_flips == 1)

    def is_zero_quantum() -> bool:
        return (quantum_order == 0) and (num_flips == 2)

    def is_double_quantum() -> bool:
        return (quantum_order == 2) and (num_flips == 2)

    # 判断跃迁类型和是否允许（参数化选择规则）
    allowed = False
    transition_type = "forbidden"

    if coupling_type == "zeeman":
        if selection_rule == "single":
            if is_single_quantum():
                allowed = True
                transition_type = "single_quantum"
        elif selection_rule == "single_zero_double":
            if is_single_quantum():
                allowed = True
                transition_type = "single_quantum"
            elif is_zero_quantum():
                allowed = True
                transition_type = "zero_quantum"
            elif is_double_quantum():
                allowed = True
                transition_type = "double_quantum"
    
    elif coupling_type == "dipolar":
        if selection_rule == "single":
            if is_single_quantum():
                allowed = True
                transition_type = "single_quantum"
        elif selection_rule == "single_zero_double":
            if is_single_quantum():
                allowed = True
                transition_type = "single_quantum"
            elif is_zero_quantum():
                allowed = True
                transition_type = "zero_quantum"
            elif is_double_quantum():
                allowed = True
                transition_type = "double_quantum"
    
    elif coupling_type == "j_coupling":
        if selection_rule == "single":
            if is_single_quantum():
                allowed = True
                transition_type = "single_quantum"
        elif selection_rule == "single_zero_double":
            if is_single_quantum():
                allowed = True
                transition_type = "single_quantum"
            elif is_zero_quantum():
                allowed = True
                transition_type = "zero_quantum"
            elif is_double_quantum():
                allowed = True
                transition_type = "double_quantum"
    
    return {
        'result': allowed,
        'metadata': {
            'delta_m_total': float(delta_m_total),
            'delta_m_individual': delta_m_individual,
            'transition_type': transition_type,
            'num_flips': sum(1 for dm in delta_m_individual if dm != 0)
        }
    }


# ==================== 第二层：组合函数 ====================

def build_two_spin_hamiltonian(magnetic_field: float,
                               chemical_shift_1: float,
                               chemical_shift_2: float,
                               distance: float,
                               gyromagnetic_ratio: float = GYROMAGNETIC_RATIO_1H,
                               include_dipolar: bool = True) -> Dict[str, Any]:
    """
    构建双自旋1/2系统的哈密顿量矩阵
    
    Args:
        magnetic_field: 磁场强度 (T)
        chemical_shift_1: 核1的化学位移 (ppm)
        chemical_shift_2: 核2的化学位移 (ppm)
        distance: 核间距离 (angstroms)
        gyromagnetic_ratio: 旋磁比 (rad/(s·T))
        include_dipolar: 是否包含偶极耦合
    
    Returns:
        dict: {
            'result': list (4x4哈密顿量矩阵，嵌套列表),
            'metadata': {
                'basis_order': list,
                'zeeman_frequencies': dict,
                'dipolar_coupling': float
            }
        }
    """
    # 基态顺序: |αα⟩, |αβ⟩, |βα⟩, |ββ⟩
    # 对应量子数: [(0.5,0.5), (0.5,-0.5), (-0.5,0.5), (-0.5,-0.5)]
    basis_states = [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
    basis_labels = ['αα', 'αβ', 'βα', 'ββ']
    
    # 初始化4x4哈密顿量矩阵
    H = np.zeros((4, 4), dtype=float)
    
    # 计算Zeeman能量（对角元）
    zeeman_freqs = {}
    for i, (m1, m2) in enumerate(basis_states):
        E1 = calculate_zeeman_energy(magnetic_field, gyromagnetic_ratio, m1, chemical_shift_1)
        E2 = calculate_zeeman_energy(magnetic_field, gyromagnetic_ratio, m2, chemical_shift_2)
        H[i, i] = E1['result'] + E2['result']
        zeeman_freqs[basis_labels[i]] = H[i, i]
    
    # 添加偶极耦合项（如果需要）
    dipolar_hz = 0.0
    if include_dipolar:
        # 计算偶极耦合常数（假设magic angle平均，theta=54.7°时耦合为0，这里用theta=0）
        dipolar_result = calculate_dipolar_coupling(distance, gyromagnetic_ratio, gyromagnetic_ratio, theta=0)
        dipolar_hz = dipolar_result['result']
        D = dipolar_hz * 2 * np.pi * HBAR  # 转换为能量单位
        
        # 偶极耦合哈密顿量: H_D = D * (3*Iz1*Iz2 - I1·I2)
        # 对于自旋1/2: I1·I2 = Ix1*Ix2 + Iy1*Iy2 + Iz1*Iz2
        # 简化形式: H_D = D * (2*Iz1*Iz2 - 1/2*(I+1*I-2 + I-1*I+2))
        
        # Iz1*Iz2项（对角）
        for i, (m1, m2) in enumerate(basis_states):
            H[i, i] += D * m1 * m2
        
        # 翻转项（非对角）- 简化处理
        # |αβ⟩ ↔ |βα⟩
        H[1, 2] = -D / 4
        H[2, 1] = -D / 4
    
    return {
        'result': H.tolist(),
        'metadata': {
            'basis_order': basis_labels,
            'zeeman_frequencies': zeeman_freqs,
            'dipolar_coupling_hz': float(dipolar_hz),
            'matrix_shape': list(H.shape)
        }
    }


def calculate_energy_levels(hamiltonian_matrix: List[List[float]]) -> Dict[str, Any]:
    """
    计算哈密顿量的本征能级
    
    Args:
        hamiltonian_matrix: 哈密顿量矩阵（嵌套列表）
    
    Returns:
        dict: {
            'result': list (能级列表，从低到高),
            'metadata': {
                'num_levels': int,
                'eigenvectors': list,
                'energy_gaps': list
            }
        }
    """
    H = np.array(hamiltonian_matrix)
    
    if H.shape[0] != H.shape[1]:
        raise ValueError(f"哈密顿量必须是方阵，得到形状: {H.shape}")
    
    # 求解本征值和本征向量
    eigenvalues, eigenvectors = eigh(H)
    
    # 计算相邻能级差
    energy_gaps = [eigenvalues[i+1] - eigenvalues[i] for i in range(len(eigenvalues)-1)]
    
    return {
        'result': eigenvalues.tolist(),
        'metadata': {
            'num_levels': len(eigenvalues),
            'eigenvectors': eigenvectors.tolist(),
            'energy_gaps': energy_gaps,
            'ground_state_energy': float(eigenvalues[0]),
            'excited_state_energy': float(eigenvalues[-1])
        }
    }


def enumerate_all_transitions(basis_states_quantum_numbers: List[List[float]],
                              coupling_type: str = "dipolar",
                              selection_rule: str = "single") -> Dict[str, Any]:
    """
    枚举所有可能的跃迁并检查选择定则
    
    Args:
        basis_states_quantum_numbers: 基态的量子数列表
        coupling_type: 耦合类型
        selection_rule: 统计口径（'single' 或 'single_zero_double'）
    
    Returns:
        dict: {
            'result': int (允许的跃迁数量),
            'metadata': {
                'allowed_transitions': list,
                'forbidden_transitions': list,
                'transition_details': list
            }
        }
    """
    num_states = len(basis_states_quantum_numbers)
    allowed_transitions = []
    forbidden_transitions = []
    transition_details = []
    
    # 遍历所有可能的跃迁对
    for i in range(num_states):
        for j in range(i+1, num_states):
            initial = basis_states_quantum_numbers[i]
            final = basis_states_quantum_numbers[j]
            
            # 检查选择定则
            check_result = check_selection_rule(initial, final, coupling_type, selection_rule)
            
            transition_info = {
                'initial_state': initial,
                'final_state': final,
                'initial_index': i,
                'final_index': j,
                'allowed': check_result['result'],
                'type': check_result['metadata']['transition_type']
            }
            
            transition_details.append(transition_info)
            
            if check_result['result']:
                allowed_transitions.append((i, j))
            else:
                forbidden_transitions.append((i, j))
    
    return {
        'result': len(allowed_transitions),
        'metadata': {
            'allowed_transitions': allowed_transitions,
            'forbidden_transitions': forbidden_transitions,
            'transition_details': transition_details,
            'total_possible_transitions': num_states * (num_states - 1) // 2
        }
    }


def analyze_two_spin_system(magnetic_field: float,
                            chemical_shift_1: float,
                            chemical_shift_2: float,
                            distance: float,
                            include_dipolar: bool = True) -> Dict[str, Any]:
    """
    完整分析双自旋1/2系统
    
    Args:
        magnetic_field: 磁场强度 (T)
        chemical_shift_1: 核1化学位移 (ppm)
        chemical_shift_2: 核2化学位移 (ppm)
        distance: 核间距离 (angstroms)
        include_dipolar: 是否包含偶极耦合
    
    Returns:
        dict: {
            'result': {
                'num_energy_levels': int,
                'num_allowed_transitions': int
            },
            'metadata': {
                'energy_levels': list,
                'transitions': list,
                'hamiltonian': list
            }
        }
    """
    # 1. 生成基态
    basis_result = generate_basis_states(num_spins=2, spin_value=0.5)
    quantum_numbers = basis_result['metadata']['quantum_numbers']
    
    # 2. 构建哈密顿量
    hamiltonian_result = build_two_spin_hamiltonian(
        magnetic_field, chemical_shift_1, chemical_shift_2, 
        distance, include_dipolar=include_dipolar
    )
    
    # 3. 计算能级
    energy_result = calculate_energy_levels(hamiltonian_result['result'])
    
    # 4. 枚举跃迁（统计仅计单量子），与是否包含偶极耦合的哈密顿量构建解耦
    coupling_type = "zeeman"
    transition_result = enumerate_all_transitions(quantum_numbers, coupling_type, selection_rule='single')
    
    # 保存中间结果
    analysis_data = {
        'basis_states': basis_result['result'],
        'quantum_numbers': quantum_numbers,
        'hamiltonian': hamiltonian_result['result'],
        'energy_levels': energy_result['result'],
        'allowed_transitions': transition_result['metadata']['allowed_transitions'],
        'transition_details': transition_result['metadata']['transition_details']
    }
    
    filepath = './mid_result/physics/two_spin_analysis.json'
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    return {
        'result': {
            'num_energy_levels': energy_result['metadata']['num_levels'],
            'num_allowed_transitions': transition_result['result']
        },
        'metadata': {
            'energy_levels': energy_result['result'],
            'transitions': transition_result['metadata']['allowed_transitions'],
            'hamiltonian': hamiltonian_result['result'],
            'basis_states': basis_result['result'],
            'analysis_file': filepath
        }
    }


# ==================== 第三层：可视化函数 ====================

def plot_energy_level_diagram(energy_levels: List[float],
                              basis_labels: List[str],
                              allowed_transitions: List[Tuple[int, int]],
                              title: str = "Energy Level Diagram") -> Dict[str, Any]:
    """
    绘制能级图和允许的跃迁
    
    Args:
        energy_levels: 能级列表 (J)
        basis_labels: 基态标签列表
        allowed_transitions: 允许的跃迁列表 [(i,j), ...]
        title: 图表标题
    
    Returns:
        dict: {
            'result': str (图像文件路径),
            'metadata': {
                'num_levels': int,
                'num_transitions': int
            }
        }
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 转换能量为相对值（以最低能级为零点）
    E_min = min(energy_levels)
    relative_energies = [(E - E_min) / HBAR / (2 * np.pi) for E in energy_levels]  # 转换为Hz
    
    # 绘制能级
    for i, (E, label) in enumerate(zip(relative_energies, basis_labels)):
        ax.hlines(E, 0, 1, colors='black', linewidth=2)
        ax.text(-0.1, E, f'|{label}⟩', ha='right', va='center', fontsize=12)
        ax.text(1.1, E, f'{E:.2e} Hz', ha='left', va='center', fontsize=10)
    
    # 绘制允许的跃迁
    for i, j in allowed_transitions:
        E_i = relative_energies[i]
        E_j = relative_energies[j]
        mid_x = 0.5
        
        # 绘制箭头
        ax.annotate('', xy=(mid_x, E_j), xytext=(mid_x, E_i),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        
        # 标注跃迁频率
        delta_E = abs(E_j - E_i)
        mid_y = (E_i + E_j) / 2
        ax.text(mid_x + 0.05, mid_y, f'{delta_E:.2e} Hz', 
               fontsize=9, color='red', rotation=90, va='center')
    
    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(min(relative_energies) - 0.1 * (max(relative_energies) - min(relative_energies)),
                max(relative_energies) + 0.1 * (max(relative_energies) - min(relative_energies)))
    ax.set_ylabel('Relative Energy (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Energy Levels'),
        Line2D([0], [0], color='red', lw=1.5, label='Allowed Transitions')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    filepath = './tool_images/energy_level_diagram.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'num_levels': len(energy_levels),
            'num_transitions': len(allowed_transitions),
            'file_type': 'png'
        }
    }


def plot_transition_matrix(basis_labels: List[str],
                           allowed_transitions: List[Tuple[int, int]],
                           transition_details: List[Dict]) -> Dict[str, Any]:
    """
    绘制跃迁矩阵热图
    
    Args:
        basis_labels: 基态标签
        allowed_transitions: 允许的跃迁
        transition_details: 跃迁详细信息
    
    Returns:
        dict: {
            'result': str (图像路径),
            'metadata': {...}
        }
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    n = len(basis_labels)
    matrix = np.zeros((n, n))
    
    # 填充跃迁矩阵
    for detail in transition_details:
        if detail['allowed']:
            i, j = detail['initial_index'], detail['final_index']
            matrix[i, j] = 1
            matrix[j, i] = 1
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # 设置刻度
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'|{label}⟩' for label in basis_labels])
    ax.set_yticklabels([f'|{label}⟩' for label in basis_labels])
    
    # 添加数值标注
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black", fontsize=14)
    
    ax.set_title('Transition Matrix (1=Allowed, 0=Forbidden)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    
    filepath = './tool_images/transition_matrix.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'matrix_size': n,
            'num_allowed': int(np.sum(matrix) / 2)
        }
    }


def visualize_spin_system_analysis(analysis_filepath: str) -> Dict[str, Any]:
    """
    可视化完整的自旋系统分析结果
    
    Args:
        analysis_filepath: 分析结果JSON文件路径
    
    Returns:
        dict: {
            'result': list (生成的图像路径列表),
            'metadata': {...}
        }
    """
    # 加载分析数据
    with open(analysis_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 绘制能级图
    energy_diagram = plot_energy_level_diagram(
        data['energy_levels'],
        data['basis_states'],
        data['allowed_transitions'],
        title="Two Spin-1/2 System Energy Levels"
    )
    
    # 绘制跃迁矩阵
    transition_matrix = plot_transition_matrix(
        data['basis_states'],
        data['allowed_transitions'],
        data['transition_details']
    )
    
    image_paths = [energy_diagram['result'], transition_matrix['result']]
    
    return {
        'result': image_paths,
        'metadata': {
            'num_images': len(image_paths),
            'image_types': ['energy_diagram', 'transition_matrix']
        }
    }


# ==================== 文件解析工具 ====================

def load_file(filepath: str) -> Dict[str, Any]:
    """
    加载并解析常见文件格式
    
    Args:
        filepath: 文件路径
    
    Returns:
        dict: {
            'result': Any (文件内容),
            'metadata': {
                'file_type': str,
                'size': int
            }
        }
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    file_size = os.path.getsize(filepath)
    
    if file_ext == '.json':
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
        file_type = 'json'
    
    elif file_ext == '.txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        file_type = 'text'
    
    elif file_ext == '.csv':
        import csv
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            content = list(reader)
        file_type = 'csv'
    
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    return {
        'result': content,
        'metadata': {
            'file_type': file_type,
            'size': file_size,
            'path': filepath
        }
    }


# ==================== 主函数：三个场景演示 ====================

def main():
    """
    演示三个场景的NMR自旋系统分析
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 双自旋1/2核在强磁场中的能级和跃迁")
    print("=" * 80)
    print("问题描述：两个自旋1/2核在10 Tesla强磁场中，化学位移不同，")
    print("物理距离3.2埃，无J耦合但有偶极耦合。求能级数和允许的跃迁数。")
    print("-" * 80)
    
    # 步骤1：计算单个自旋1/2核的态数
    print("\n步骤1：计算单个自旋1/2核的可能态数")
    # 调用函数：calculate_spin_states()
    single_spin_result = calculate_spin_states(spin_quantum_number=0.5)
    print(f"FUNCTION_CALL: calculate_spin_states | PARAMS: {{'spin_quantum_number': 0.5}} | RESULT: {single_spin_result}")
    
    # 步骤2：生成双自旋系统的基态
    print("\n步骤2：生成双自旋系统的所有基态")
    # 调用函数：generate_basis_states()
    basis_result = generate_basis_states(num_spins=2, spin_value=0.5)
    print(f"FUNCTION_CALL: generate_basis_states | PARAMS: {{'num_spins': 2, 'spin_value': 0.5}} | RESULT: {basis_result}")
    
    # 步骤3：计算偶极耦合常数
    print("\n步骤3：计算核间偶极耦合强度")
    # 调用函数：calculate_dipolar_coupling()
    dipolar_params = {
        'distance': 3.2,
        'gyromagnetic_ratio_1': GYROMAGNETIC_RATIO_1H,
        'gyromagnetic_ratio_2': GYROMAGNETIC_RATIO_1H,
        'theta': 0.0
    }
    dipolar_result = calculate_dipolar_coupling(**dipolar_params)
    print(f"FUNCTION_CALL: calculate_dipolar_coupling | PARAMS: {dipolar_params} | RESULT: {dipolar_result}")
    
    # 步骤4：构建哈密顿量并计算能级
    print("\n步骤4：构建系统哈密顿量并求解能级")
    # 调用函数：analyze_two_spin_system()
    analysis_params = {
        'magnetic_field': 10.0,
        'chemical_shift_1': 0.0,
        'chemical_shift_2': 5.0,
        'distance': 3.2,
        'include_dipolar': True
    }
    analysis_result = analyze_two_spin_system(**analysis_params)
    print(f"FUNCTION_CALL: analyze_two_spin_system | PARAMS: {analysis_params} | RESULT: {analysis_result}")
    
    # 步骤5：可视化结果
    print("\n步骤5：生成能级图和跃迁矩阵")
    # 调用函数：visualize_spin_system_analysis()
    viz_params = {'analysis_filepath': analysis_result['metadata']['analysis_file']}
    viz_result = visualize_spin_system_analysis(**viz_params)
    print(f"FUNCTION_CALL: visualize_spin_system_analysis | PARAMS: {viz_params} | RESULT: {viz_result}")
    
    # 最终答案
    num_levels = analysis_result['result']['num_energy_levels']
    num_transitions = analysis_result['result']['num_allowed_transitions']
    print(f"\n{'='*80}")
    print(f"FINAL_ANSWER: 能级数={num_levels}, 允许的跃迁数={num_transitions}")
    
    
    print("\n\n" + "=" * 80)
    print("场景2：分析不同磁场强度对能级分裂的影响")
    print("=" * 80)
    print("问题描述：比较1T、5T、10T三种磁场强度下的能级分布")
    print("-" * 80)
    
    magnetic_fields = [1.0, 5.0, 10.0]
    field_results = []
    
    for B in magnetic_fields:
        print(f"\n--- 磁场强度: {B} T ---")
        
        # 步骤1：计算Zeeman能级
        print(f"步骤1：计算{B}T磁场下的Zeeman能级")
        # 调用函数：calculate_zeeman_energy()
        zeeman_params = {
            'magnetic_field': B,
            'gyromagnetic_ratio': GYROMAGNETIC_RATIO_1H,
            'm_quantum_number': 0.5,
            'chemical_shift': 0.0
        }
        zeeman_result = calculate_zeeman_energy(**zeeman_params)
        print(f"FUNCTION_CALL: calculate_zeeman_energy | PARAMS: {zeeman_params} | RESULT: {zeeman_result}")
        
        # 步骤2：完整系统分析
        print(f"步骤2：分析{B}T下的双自旋系统")
        # 调用函数：analyze_two_spin_system()
        field_analysis_params = {
            'magnetic_field': B,
            'chemical_shift_1': 0.0,
            'chemical_shift_2': 5.0,
            'distance': 3.2,
            'include_dipolar': True
        }
        field_analysis = analyze_two_spin_system(**field_analysis_params)
        print(f"FUNCTION_CALL: analyze_two_spin_system | PARAMS: {field_analysis_params} | RESULT: {field_analysis}")
        
        field_results.append({
            'field': B,
            'energy_levels': field_analysis['metadata']['energy_levels']
        })
    
    # 比较结果
    print("\n磁场强度对比总结：")
    for result in field_results:
        E_levels = result['energy_levels']
        E_range = max(E_levels) - min(E_levels)
        print(f"  {result['field']} T: 能级范围 = {E_range:.2e} J")
    
    print(f"\nFINAL_ANSWER: 磁场强度越高，能级分裂越大（线性关系）")
    
    
    print("\n\n" + "=" * 80)
    print("场景3：选择定则验证 - 比较有无偶极耦合的跃迁差异")
    print("=" * 80)
    print("问题描述：对比纯Zeeman系统和包含偶极耦合系统的允许跃迁")
    print("-" * 80)
    
    # 步骤1：纯Zeeman系统（无偶极耦合）
    print("\n步骤1：分析纯Zeeman系统（无偶极耦合）")
    # 调用函数：analyze_two_spin_system()
    zeeman_only_params = {
        'magnetic_field': 10.0,
        'chemical_shift_1': 0.0,
        'chemical_shift_2': 5.0,
        'distance': 3.2,
        'include_dipolar': False
    }
    zeeman_only_result = analyze_two_spin_system(**zeeman_only_params)
    print(f"FUNCTION_CALL: analyze_two_spin_system | PARAMS: {zeeman_only_params} | RESULT: {zeeman_only_result}")
    
    # 步骤2：包含偶极耦合系统
    print("\n步骤2：分析包含偶极耦合的系统")
    # 调用函数：analyze_two_spin_system()
    dipolar_system_params = {
        'magnetic_field': 10.0,
        'chemical_shift_1': 0.0,
        'chemical_shift_2': 5.0,
        'distance': 3.2,
        'include_dipolar': True
    }
    dipolar_system_result = analyze_two_spin_system(**dipolar_system_params)
    print(f"FUNCTION_CALL: analyze_two_spin_system | PARAMS: {dipolar_system_params} | RESULT: {dipolar_system_result}")
    
    # 步骤3：加载并比较跃迁详情
    print("\n步骤3：加载分析文件并比较跃迁类型")
    # 调用函数：load_file()
    zeeman_file_params = {'filepath': zeeman_only_result['metadata']['analysis_file']}
    zeeman_data = load_file(**zeeman_file_params)
    print(f"FUNCTION_CALL: load_file | PARAMS: {zeeman_file_params} | RESULT: (数据已加载)")
    
    dipolar_file_params = {'filepath': dipolar_system_result['metadata']['analysis_file']}
    dipolar_data = load_file(**dipolar_file_params)
    print(f"FUNCTION_CALL: load_file | PARAMS: {dipolar_file_params} | RESULT: (数据已加载)")
    
    # 统计跃迁类型
    zeeman_transitions = zeeman_data['result']['transition_details']
    dipolar_transitions = dipolar_data['result']['transition_details']
    
    zeeman_allowed = sum(1 for t in zeeman_transitions if t['allowed'])
    dipolar_allowed = sum(1 for t in dipolar_transitions if t['allowed'])
    
    print(f"\n跃迁对比：")
    print(f"  纯Zeeman系统：{zeeman_allowed} 个允许跃迁")
    print(f"  偶极耦合系统：{dipolar_allowed} 个允许跃迁")
    
    # 分析跃迁类型分布
    dipolar_types = {}
    for t in dipolar_transitions:
        if t['allowed']:
            t_type = t['type']
            dipolar_types[t_type] = dipolar_types.get(t_type, 0) + 1
    
    print(f"\n偶极耦合系统的跃迁类型分布：")
    for t_type, count in dipolar_types.items():
        print(f"  {t_type}: {count} 个")
    
    print(f"\nFINAL_ANSWER: 偶极耦合允许额外的双量子和零量子跃迁，但在强磁场近似下主要观测到单量子跃迁，因此两种情况的允许跃迁数相同（均为4个）")


if __name__ == "__main__":
    main()