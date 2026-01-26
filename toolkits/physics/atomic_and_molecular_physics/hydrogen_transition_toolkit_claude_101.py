# Filename: hydrogen_transition_toolkit.py

"""
Hydrogen Atom Dipole Transition Analysis Toolkit
专业的氢原子偶极跃迁分析工具包

功能模块：
1. 量子态表示与验证
2. 偶极跃迁选择定则检查
3. 跃迁路径搜索与分析
4. 跃迁概率计算（基于Wigner-Eckart定理）
5. 能级图与跃迁路径可视化

依赖库：
- sympy: 符号计算（Wigner 3j符号、Clebsch-Gordan系数）
- scipy.special: 特殊函数（球谐函数、径向波函数）
- numpy: 数值计算
- matplotlib: 可视化
"""

import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial
from sympy.physics.wigner import wigner_3j, clebsch_gordan
import sympy as sp
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os
from itertools import product

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
os.makedirs('./mid_result/quantum_physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础量子态操作
# ============================================================================

def validate_quantum_state(n: int, l: int, m: int) -> Dict:
    """
    验证量子态的物理有效性
    
    参数:
        n: 主量子数 (n ≥ 1)
        l: 角量子数 (0 ≤ l < n)
        m: 磁量子数 (-l ≤ m ≤ l)
    
    返回:
        {'result': bool, 'metadata': {'valid': bool, 'reason': str}}
    """
    if not isinstance(n, int) or not isinstance(l, int) or not isinstance(m, int):
        return {
            'result': False,
            'metadata': {
                'valid': False,
                'reason': 'Quantum numbers must be integers'
            }
        }
    
    if n < 1:
        return {
            'result': False,
            'metadata': {
                'valid': False,
                'reason': f'Principal quantum number n={n} must be ≥ 1'
            }
        }
    
    if l < 0 or l >= n:
        return {
            'result': False,
            'metadata': {
                'valid': False,
                'reason': f'Angular quantum number l={l} must satisfy 0 ≤ l < n={n}'
            }
        }
    
    if abs(m) > l:
        return {
            'result': False,
            'metadata': {
                'valid': False,
                'reason': f'Magnetic quantum number m={m} must satisfy |m| ≤ l={l}'
            }
        }
    
    return {
        'result': True,
        'metadata': {
            'valid': True,
            'reason': 'Valid quantum state',
            'state_notation': f'|{n},{l},{m}⟩',
            'spectroscopic_notation': f'{n}{_l_to_spectroscopic(l)}'
        }
    }


def _l_to_spectroscopic(l: int) -> str:
    """将角量子数转换为光谱学符号"""
    spectroscopic = {0: 's', 1: 'p', 2: 'd', 3: 'f', 4: 'g', 5: 'h'}
    return spectroscopic.get(l, f'l={l}')


def check_dipole_selection_rule(n1: int, l1: int, m1: int, 
                                  n2: int, l2: int, m2: int) -> Dict:
    """
    检查偶极跃迁选择定则
    
    选择定则：
        Δl = ±1 (角量子数必须改变1)
        Δm = 0, ±1 (磁量子数改变0或±1)
        Δn: 无限制
    
    参数:
        n1, l1, m1: 初态量子数
        n2, l2, m2: 末态量子数
    
    返回:
        {'result': bool, 'metadata': {...}}
    """
    # 验证两个态的有效性
    state1_valid = validate_quantum_state(n1, l1, m1)
    state2_valid = validate_quantum_state(n2, l2, m2)
    
    if not state1_valid['result'] or not state2_valid['result']:
        return {
            'result': False,
            'metadata': {
                'allowed': False,
                'reason': 'Invalid quantum state(s)',
                'initial_state': state1_valid['metadata'],
                'final_state': state2_valid['metadata']
            }
        }
    
    delta_l = l2 - l1
    delta_m = m2 - m1
    
    # 检查选择定则
    l_allowed = (delta_l == 1 or delta_l == -1)
    m_allowed = (delta_m in [-1, 0, 1])
    
    allowed = l_allowed and m_allowed
    
    return {
        'result': allowed,
        'metadata': {
            'allowed': allowed,
            'initial_state': f'|{n1},{l1},{m1}⟩',
            'final_state': f'|{n2},{l2},{m2}⟩',
            'delta_l': delta_l,
            'delta_m': delta_m,
            'l_rule_satisfied': l_allowed,
            'm_rule_satisfied': m_allowed,
            'reason': _get_selection_rule_reason(l_allowed, m_allowed, delta_l, delta_m)
        }
    }


def _get_selection_rule_reason(l_allowed: bool, m_allowed: bool, 
                                delta_l: int, delta_m: int) -> str:
    """生成选择定则检查的详细原因"""
    if l_allowed and m_allowed:
        return f'Allowed: Δl={delta_l} (±1), Δm={delta_m} (0,±1)'
    elif not l_allowed:
        return f'Forbidden: Δl={delta_l} violates Δl=±1 rule'
    else:
        return f'Forbidden: Δm={delta_m} violates Δm=0,±1 rule'


def calculate_transition_matrix_element(n1: int, l1: int, m1: int,
                                        n2: int, l2: int, m2: int,
                                        polarization: str = 'z') -> Dict:
    """
    计算偶极跃迁矩阵元 ⟨n2,l2,m2|r|n1,l1,m1⟩
    
    使用Wigner-Eckart定理分解：
    ⟨n2,l2,m2|r_q|n1,l1,m1⟩ = ⟨n2,l2||r||n1,l1⟩ × CG系数
    
    参数:
        n1, l1, m1: 初态量子数
        n2, l2, m2: 末态量子数
        polarization: 偏振方向 ('z', 'x', 'y', 'circular+', 'circular-')
    
    返回:
        {'result': float, 'metadata': {...}}
    """
    # 检查选择定则
    selection = check_dipole_selection_rule(n1, l1, m1, n2, l2, m2)
    
    if not selection['result']:
        return {
            'result': 0.0,
            'metadata': {
                'matrix_element': 0.0,
                'reason': 'Forbidden by selection rules',
                'selection_rule_check': selection['metadata']
            }
        }
    
    # 确定偏振对应的q值
    q_map = {
        'z': 0,           # π偏振
        'circular+': 1,   # σ+偏振
        'circular-': -1,  # σ-偏振
    }
    
    if polarization not in q_map:
        return {
            'result': 0.0,
            'metadata': {
                'error': f'Invalid polarization: {polarization}',
                'valid_polarizations': list(q_map.keys())
            }
        }
    
    q = q_map[polarization]
    
    # 检查m选择定则
    if m2 - m1 != q:
        return {
            'result': 0.0,
            'metadata': {
                'matrix_element': 0.0,
                'reason': f'Δm={m2-m1} does not match polarization q={q}'
            }
        }
    
    # 计算约化矩阵元（使用氢原子解析公式）
    reduced_me = _calculate_reduced_matrix_element(n1, l1, n2, l2)
    
    # 计算Clebsch-Gordan系数
    # ⟨l2,m2|l1,m1;1,q⟩
    cg_coeff = float(clebsch_gordan(l1, 1, l2, m1, q, m2))
    
    # 完整矩阵元
    matrix_element = reduced_me * cg_coeff
    
    return {
        'result': float(matrix_element),
        'metadata': {
            'matrix_element': float(matrix_element),
            'reduced_matrix_element': float(reduced_me),
            'clebsch_gordan_coefficient': float(cg_coeff),
            'polarization': polarization,
            'q': q,
            'transition': f'|{n1},{l1},{m1}⟩ → |{n2},{l2},{m2}⟩'
        }
    }


def _calculate_reduced_matrix_element(n1: int, l1: int, n2: int, l2: int) -> float:
    """
    计算约化矩阵元 ⟨n2,l2||r||n1,l1⟩
    
    使用氢原子径向积分的解析公式（单位：玻尔半径a0）
    """
    # 简化计算：使用数值积分方法
    # 对于氢原子，可以使用已知的解析结果
    
    # 这里使用近似公式（基于量子力学教材）
    # 实际应用中可以查表或使用更精确的数值积分
    
    if abs(l2 - l1) != 1:
        return 0.0
    
    # 使用简化的振子强度公式
    # f ~ (n^2 - n'^2) for large n
    # 这里使用更精确的数值方法
    
    # 归一化常数
    norm1 = np.sqrt((2.0/n1)**3 * factorial(n1-l1-1) / (2*n1*factorial(n1+l1)))
    norm2 = np.sqrt((2.0/n2)**3 * factorial(n2-l2-1) / (2*n2*factorial(n2+l2)))
    
    # 径向积分的数值估计（使用高斯-拉盖尔求积）
    # 这是一个简化版本，实际计算需要更精确的方法
    radial_integral = _numerical_radial_integral(n1, l1, n2, l2)
    
    reduced_me = norm1 * norm2 * radial_integral
    
    return reduced_me


def _numerical_radial_integral(n1: int, l1: int, n2: int, l2: int) -> float:
    """数值计算径向积分"""
    # 使用高斯-拉盖尔求积
    from scipy.integrate import quad
    
    def radial_wavefunction(r, n, l):
        """氢原子径向波函数 R_nl(r)"""
        rho = 2 * r / n
        norm = np.sqrt((2.0/n)**3 * factorial(n-l-1) / (2*n*factorial(n+l)))
        laguerre = genlaguerre(n-l-1, 2*l+1)(rho)
        return norm * np.exp(-rho/2) * rho**l * laguerre
    
    def integrand(r):
        R1 = radial_wavefunction(r, n1, l1)
        R2 = radial_wavefunction(r, n2, l2)
        return R1 * r * R2 * r**2  # r^3 因子来自 r 算符和 r^2 dr
    
    # 积分范围：0到足够大的值（约10倍最大主量子数）
    r_max = 10 * max(n1, n2)
    result, _ = quad(integrand, 0, r_max, limit=100)
    
    return result


def calculate_transition_probability(n1: int, l1: int, m1: int,
                                     n2: int, l2: int, m2: int) -> Dict:
    """
    计算跃迁概率（相对强度）
    
    跃迁概率正比于 |⟨f|r|i⟩|^2 对所有可能偏振的求和
    
    参数:
        n1, l1, m1: 初态量子数
        n2, l2, m2: 末态量子数
    
    返回:
        {'result': float, 'metadata': {...}}
    """
    # 对所有可能的偏振求和
    polarizations = ['z', 'circular+', 'circular-']
    
    total_probability = 0.0
    contributions = {}
    
    for pol in polarizations:
        me_result = calculate_transition_matrix_element(n1, l1, m1, n2, l2, m2, pol)
        me = me_result['result']
        prob = abs(me)**2
        total_probability += prob
        contributions[pol] = {
            'matrix_element': float(me),
            'probability': float(prob)
        }
    
    return {
        'result': float(total_probability),
        'metadata': {
            'total_probability': float(total_probability),
            'contributions_by_polarization': contributions,
            'transition': f'|{n1},{l1},{m1}⟩ → |{n2},{l2},{m2}⟩',
            'note': 'Relative probability (not normalized)'
        }
    }


# ============================================================================
# 第二层：组合函数 - 跃迁路径分析
# ============================================================================

def find_all_intermediate_states(n_initial: int, l_initial: int, m_initial: int,
                                 n_final: int, l_final: int, m_final: int) -> Dict:
    """
    寻找所有可能的中间态（单步偶极跃迁）
    
    参数:
        n_initial, l_initial, m_initial: 初态量子数
        n_final, l_final, m_final: 末态量子数
    
    返回:
        {'result': List[Tuple], 'metadata': {...}}
    """
    # 验证初末态
    initial_valid = validate_quantum_state(n_initial, l_initial, m_initial)
    final_valid = validate_quantum_state(n_final, l_final, m_final)
    
    if not initial_valid['result'] or not final_valid['result']:
        return {
            'result': [],
            'metadata': {
                'error': 'Invalid initial or final state',
                'initial_state_check': initial_valid['metadata'],
                'final_state_check': final_valid['metadata']
            }
        }
    
    # 搜索中间态：需要满足两次偶极跃迁；并加入物理约束：能量单调下降
    # 第一步：initial → intermediate；第二步：intermediate → final
    intermediate_states = []

    # 氢原子能级 E_n = -13.6/n^2，能量单调下降对应 n_final < n_inter < n_initial
    n_lower = min(n_initial, n_final)
    n_upper = max(n_initial, n_final)
    # 我们期望从更高能级（较大能量，较小 n）向更低能级（更负能量，较大 n）衰变，
    # 但为兼容一般性，这里采用严格限制：n_final < n_inter < n_initial 当 n_initial > n_final
    if n_initial > n_final:
        n_start = n_final + 1
        n_end = n_initial - 1
    else:
        # 若初末态不满足能量降低（非常规情形），保持原有宽松搜索但仍不越界
        n_start = 1
        n_end = max(n_initial, n_final) + 2

    for n_inter in range(max(1, n_start), max(1, n_end) + 1):
        for l_inter in range(0, n_inter):
            for m_inter in range(-l_inter, l_inter + 1):
                # 检查第一步跃迁
                step1 = check_dipole_selection_rule(
                    n_initial, l_initial, m_initial,
                    n_inter, l_inter, m_inter
                )
                
                # 检查第二步跃迁
                step2 = check_dipole_selection_rule(
                    n_inter, l_inter, m_inter,
                    n_final, l_final, m_final
                )
                
                if step1['result'] and step2['result']:
                    intermediate_states.append((n_inter, l_inter, m_inter))
    
    return {
        'result': intermediate_states,
        'metadata': {
            'count': len(intermediate_states),
            'initial_state': f'|{n_initial},{l_initial},{m_initial}⟩',
            'final_state': f'|{n_final},{l_final},{m_final}⟩',
            'intermediate_states': [
                f'|{n},{l},{m}⟩' for n, l, m in intermediate_states
            ]
        }
    }


def calculate_two_step_transition_probability(n1: int, l1: int, m1: int,
                                              n2: int, l2: int, m2: int,
                                              n3: int, l3: int, m3: int) -> Dict:
    """
    计算两步跃迁的总概率
    
    对于级联跃迁 |1⟩ → |2⟩ → |3⟩：
    - 如果中间态|2⟩是稳定的，概率是两步概率的乘积
    - 考虑分支比：P(1→2→3) = P(1→2) × BR(2→3)
    
    参数:
        n1, l1, m1: 初态
        n2, l2, m2: 中间态
        n3, l3, m3: 末态
    
    返回:
        {'result': float, 'metadata': {...}}
    """
    # 第一步跃迁概率
    prob1_result = calculate_transition_probability(n1, l1, m1, n2, l2, m2)
    prob1 = prob1_result['result']
    
    # 第二步跃迁概率
    prob2_result = calculate_transition_probability(n2, l2, m2, n3, l3, m3)
    prob2 = prob2_result['result']
    
    # 计算中间态的所有可能衰变通道（用于分支比）
    all_decay_channels = _find_all_decay_channels(n2, l2, m2, n3)
    
    # 计算分支比
    total_decay_prob = sum(ch['probability'] for ch in all_decay_channels)
    
    if total_decay_prob > 0:
        branching_ratio = prob2 / total_decay_prob
    else:
        branching_ratio = 0.0
    
    # 两步跃迁总概率
    total_prob = prob1 * branching_ratio
    
    return {
        'result': float(total_prob),
        'metadata': {
            'total_probability': float(total_prob),
            'step1_probability': float(prob1),
            'step2_probability': float(prob2),
            'branching_ratio': float(branching_ratio),
            'transition_path': f'|{n1},{l1},{m1}⟩ → |{n2},{l2},{m2}⟩ → |{n3},{l3},{m3}⟩',
            'decay_channels_from_intermediate': len(all_decay_channels)
        }
    }


def _find_all_decay_channels(n: int, l: int, m: int, n_min: int = 1) -> List[Dict]:
    """
    找到给定态的所有可能衰变通道（能量降低的跃迁）
    
    参数:
        n, l, m: 当前态
        n_min: 最低能级（通常是1）
    
    返回:
        List of decay channels with probabilities
    """
    channels = []
    
    # 只考虑能量降低的跃迁 (n' < n)
    for n_final in range(n_min, n):
        for l_final in range(0, n_final):
            for m_final in range(-l_final, l_final + 1):
                selection = check_dipole_selection_rule(n, l, m, n_final, l_final, m_final)
                
                if selection['result']:
                    prob_result = calculate_transition_probability(n, l, m, n_final, l_final, m_final)
                    channels.append({
                        'final_state': (n_final, l_final, m_final),
                        'probability': prob_result['result']
                    })
    
    return channels


def analyze_all_transition_paths(n_initial: int, l_initial: int, m_initial: int,
                                 n_final: int, l_final: int, m_final: int) -> Dict:
    """
    分析所有可能的两步跃迁路径及其概率
    
    参数:
        n_initial, l_initial, m_initial: 初态
        n_final, l_final, m_final: 末态
    
    返回:
        {'result': List[Dict], 'metadata': {...}}
    """
    # 找到所有中间态
    intermediate_result = find_all_intermediate_states(
        n_initial, l_initial, m_initial,
        n_final, l_final, m_final
    )
    
    intermediate_states = intermediate_result['result']
    
    if not intermediate_states:
        return {
            'result': [],
            'metadata': {
                'message': 'No valid two-step transition paths found',
                'initial_state': f'|{n_initial},{l_initial},{m_initial}⟩',
                'final_state': f'|{n_final},{l_final},{m_final}⟩',
                'total_paths': 0
            }
        }
    
    # 计算每条路径的概率
    paths = []
    for n_inter, l_inter, m_inter in intermediate_states:
        prob_result = calculate_two_step_transition_probability(
            n_initial, l_initial, m_initial,
            n_inter, l_inter, m_inter,
            n_final, l_final, m_final
        )
        
        paths.append({
            'intermediate_state': (n_inter, l_inter, m_inter),
            'probability': prob_result['result'],
            'path_notation': f'|{n_initial},{l_initial},{m_initial}⟩ → |{n_inter},{l_inter},{m_inter}⟩ → |{n_final},{l_final},{m_final}⟩',
            'details': prob_result['metadata']
        })
    
    # 按概率排序
    paths.sort(key=lambda x: x['probability'], reverse=True)
    
    # 归一化概率
    total_prob = sum(p['probability'] for p in paths)
    
    if total_prob > 0:
        for path in paths:
            path['normalized_probability'] = path['probability'] / total_prob
    
    # 保存详细结果到文件
    output_file = './mid_result/quantum_physics/transition_paths_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'initial_state': f'|{n_initial},{l_initial},{m_initial}⟩',
            'final_state': f'|{n_final},{l_final},{m_final}⟩',
            'total_paths': len(paths),
            'paths': paths
        }, f, indent=2, ensure_ascii=False)
    
    return {
        'result': paths,
        'metadata': {
            'total_paths': len(paths),
            'total_probability': float(total_prob),
            'initial_state': f'|{n_initial},{l_initial},{m_initial}⟩',
            'final_state': f'|{n_final},{l_final},{m_final}⟩',
            'output_file': output_file,
            'most_probable_path': paths[0]['path_notation'] if paths else None
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_energy_levels_and_transitions(n_initial: int, l_initial: int, m_initial: int,
                                           n_final: int, l_final: int, m_final: int,
                                           paths: List[Dict]) -> Dict:
    """
    可视化氢原子能级图和跃迁路径
    
    参数:
        n_initial, l_initial, m_initial: 初态
        n_final, l_final, m_final: 末态
        paths: 跃迁路径列表（来自analyze_all_transition_paths）
    
    返回:
        {'result': str, 'metadata': {...}}
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 氢原子能级公式：E_n = -13.6 eV / n^2
    def energy(n):
        return -13.6 / n**2
    
    # 收集所有涉及的能级
    all_n = {n_initial, n_final}
    for path in paths:
        n_inter, _, _ = path['intermediate_state']
        all_n.add(n_inter)
    
    # 绘制能级线
    n_levels = sorted(all_n)
    level_positions = {}
    
    for i, n in enumerate(n_levels):
        E = energy(n)
        # 为不同的l值设置不同的x位置
        for l in range(n):
            x_pos = i * 2.0 + l * 0.3
            ax.hlines(E, x_pos - 0.4, x_pos + 0.4, colors='black', linewidth=2)
            
            # 标注能级
            spectro = _l_to_spectroscopic(l)
            ax.text(x_pos, E + 0.3, f'{n}{spectro}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            level_positions[(n, l)] = (x_pos, E)
    
    # 绘制跃迁箭头
    # 只绘制概率最高的前3条路径
    top_paths = paths[:min(3, len(paths))]
    
    colors = ['red', 'blue', 'green']
    
    for idx, path in enumerate(top_paths):
        n_inter, l_inter, m_inter = path['intermediate_state']
        prob = path.get('normalized_probability', path['probability'])
        
        color = colors[idx % len(colors)]
        alpha = 0.3 + 0.7 * prob  # 透明度反映概率
        
        # 第一步跃迁
        if (n_initial, l_initial) in level_positions and (n_inter, l_inter) in level_positions:
            x1, y1 = level_positions[(n_initial, l_initial)]
            x2, y2 = level_positions[(n_inter, l_inter)]
            
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=alpha))
        
        # 第二步跃迁
        if (n_inter, l_inter) in level_positions and (n_final, l_final) in level_positions:
            x2, y2 = level_positions[(n_inter, l_inter)]
            x3, y3 = level_positions[(n_final, l_final)]
            
            ax.annotate('', xy=(x3, y3), xytext=(x2, y2),
                       arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=alpha))
        
        # 添加概率标签
        label = f'Path {idx+1}: P={prob:.3f}'
        ax.plot([], [], color=color, linewidth=2, label=label)
    
    # 标注初态和末态
    if (n_initial, l_initial) in level_positions:
        x, y = level_positions[(n_initial, l_initial)]
        ax.plot(x, y, 'go', markersize=15, label='Initial State')
    
    if (n_final, l_final) in level_positions:
        x, y = level_positions[(n_final, l_final)]
        ax.plot(x, y, 'rs', markersize=15, label='Final State')
    
    ax.set_ylabel('Energy (eV)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Quantum States', fontsize=14, fontweight='bold')
    ax.set_title(f'Hydrogen Atom Dipole Transitions\n|{n_initial},{l_initial},{m_initial}⟩ → |{n_final},{l_final},{m_final}⟩',
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置y轴范围
    ax.set_ylim([energy(max(n_levels)) - 2, 0.5])
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/hydrogen_transition_energy_levels.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'image/png',
            'description': 'Energy level diagram with transition paths',
            'paths_shown': len(top_paths),
            'total_paths': len(paths)
        }
    }


def visualize_transition_probabilities(paths: List[Dict]) -> Dict:
    """
    可视化跃迁概率分布（柱状图）
    
    参数:
        paths: 跃迁路径列表
    
    返回:
        {'result': str, 'metadata': {...}}
    """
    if not paths:
        return {
            'result': None,
            'metadata': {'error': 'No paths to visualize'}
        }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 提取数据
    path_labels = []
    probabilities = []
    normalized_probs = []
    
    for i, path in enumerate(paths):
        n_inter, l_inter, m_inter = path['intermediate_state']
        label = f'|{n_inter},{l_inter},{m_inter}⟩'
        path_labels.append(label)
        probabilities.append(path['probability'])
        normalized_probs.append(path.get('normalized_probability', 0))
    
    # 左图：原始概率
    ax1.bar(range(len(paths)), probabilities, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Intermediate State', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Relative Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Transition Probabilities (Relative)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(paths)))
    ax1.set_xticklabels(path_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：归一化概率
    ax2.bar(range(len(paths)), normalized_probs, color='coral', alpha=0.7)
    ax2.set_xlabel('Intermediate State', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Transition Probabilities (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(paths)))
    ax2.set_xticklabels(path_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    # 在柱子上标注数值
    for i, (p, np) in enumerate(zip(probabilities, normalized_probs)):
        ax1.text(i, p, f'{p:.2e}', ha='center', va='bottom', fontsize=8)
        ax2.text(i, np, f'{np:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/transition_probabilities.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'image/png',
            'description': 'Bar chart of transition probabilities',
            'total_paths': len(paths)
        }
    }


def create_transition_summary_report(n_initial: int, l_initial: int, m_initial: int,
                                    n_final: int, l_final: int, m_final: int,
                                    paths: List[Dict]) -> Dict:
    """
    生成跃迁分析的完整报告（文本文件）
    
    参数:
        n_initial, l_initial, m_initial: 初态
        n_final, l_final, m_final: 末态
        paths: 跃迁路径列表
    
    返回:
        {'result': str, 'metadata': {...}}
    """
    filepath = './mid_result/quantum_physics/transition_summary_report.txt'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HYDROGEN ATOM DIPOLE TRANSITION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Initial State: |{n_initial},{l_initial},{m_initial}⟩ ({n_initial}{_l_to_spectroscopic(l_initial)})\n")
        f.write(f"Final State:   |{n_final},{l_final},{m_final}⟩ ({n_final}{_l_to_spectroscopic(l_final)})\n")
        f.write(f"Energy Change: ΔE = {-13.6/n_final**2 - (-13.6/n_initial**2):.4f} eV\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("SELECTION RULES CHECK\n")
        f.write("-" * 80 + "\n")
        f.write("Dipole Selection Rules:\n")
        f.write("  • Δl = ±1 (angular momentum must change by 1)\n")
        f.write("  • Δm = 0, ±1 (magnetic quantum number change)\n")
        f.write("  • Δn: no restriction\n\n")
        
        delta_l = l_final - l_initial
        delta_m = m_final - m_initial
        
        f.write(f"Direct Transition Check:\n")
        f.write(f"  Δl = {delta_l} (required: ±1)\n")
        f.write(f"  Δm = {delta_m} (required: 0, ±1)\n")
        
        if abs(delta_l) == 1 and abs(delta_m) <= 1:
            f.write("  ✓ Direct transition ALLOWED\n\n")
        else:
            f.write("  ✗ Direct transition FORBIDDEN\n")
            f.write("  → Two-step transition required\n\n")
        
        f.write("-" * 80 + "\n")
        f.write(f"TWO-STEP TRANSITION PATHS (Total: {len(paths)})\n")
        f.write("-" * 80 + "\n\n")
        
        if not paths:
            f.write("No valid two-step transition paths found.\n\n")
        else:
            for i, path in enumerate(paths, 1):
                n_inter, l_inter, m_inter = path['intermediate_state']
                prob = path.get('normalized_probability', path['probability'])
                
                f.write(f"Path {i}:\n")
                f.write(f"  Route: |{n_initial},{l_initial},{m_initial}⟩ → |{n_inter},{l_inter},{m_inter}⟩ → |{n_final},{l_final},{m_final}⟩\n")
                f.write(f"  Spectroscopic: {n_initial}{_l_to_spectroscopic(l_initial)} → {n_inter}{_l_to_spectroscopic(l_inter)} → {n_final}{_l_to_spectroscopic(l_final)}\n")
                f.write(f"  Normalized Probability: {prob:.6f}\n")
                
                if 'details' in path:
                    details = path['details']
                    f.write(f"  Step 1 Probability: {details.get('step1_probability', 0):.6e}\n")
                    f.write(f"  Step 2 Probability: {details.get('step2_probability', 0):.6e}\n")
                    f.write(f"  Branching Ratio: {details.get('branching_ratio', 0):.6f}\n")
                
                f.write("\n")
        
        f.write("-" * 80 + "\n")
        f.write("MOST PROBABLE PATH\n")
        f.write("-" * 80 + "\n")
        
        if paths:
            most_probable = paths[0]
            n_inter, l_inter, m_inter = most_probable['intermediate_state']
            prob = most_probable.get('normalized_probability', most_probable['probability'])
            
            f.write(f"Intermediate State: |{n_inter},{l_inter},{m_inter}⟩ ({n_inter}{_l_to_spectroscopic(l_inter)})\n")
            f.write(f"Transition Route: {most_probable['path_notation']}\n")
            f.write(f"Probability: {prob:.6f} ({prob*100:.2f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"FILE_GENERATED: text | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'text/plain',
            'description': 'Comprehensive transition analysis report',
            'total_paths': len(paths)
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示氢原子偶极跃迁分析工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - |3,0,0⟩ → |1,0,0⟩ 的两步偶极跃迁")
    print("=" * 80)
    print("问题描述：氢原子从 |3,0,0⟩ 态通过两次偶极跃迁衰变到 |1,0,0⟩ 态")
    print("要求：找出跃迁路径和对应的概率")
    print("-" * 80)
    
    # 步骤1：验证初末态
    print("\n步骤1：验证量子态的有效性")
    initial_state = (3, 0, 0)
    final_state = (1, 0, 0)
    
    initial_valid = validate_quantum_state(*initial_state)
    final_valid = validate_quantum_state(*final_state)
    
    print(f"FUNCTION_CALL: validate_quantum_state | PARAMS: {initial_state} | RESULT: {initial_valid}")
    print(f"FUNCTION_CALL: validate_quantum_state | PARAMS: {final_state} | RESULT: {final_valid}")
    
    # 步骤2：检查直接跃迁是否允许
    print("\n步骤2：检查直接跃迁选择定则")
    direct_transition = check_dipole_selection_rule(*initial_state, *final_state)
    print(f"FUNCTION_CALL: check_dipole_selection_rule | PARAMS: {initial_state + final_state} | RESULT: {direct_transition}")
    
    if not direct_transition['result']:
        print("→ 直接跃迁被禁止，需要两步跃迁")
    
    # 步骤3：寻找所有可能的中间态
    print("\n步骤3：寻找所有可能的中间态")
    intermediate_states = find_all_intermediate_states(*initial_state, *final_state)
    print(f"FUNCTION_CALL: find_all_intermediate_states | PARAMS: {initial_state + final_state} | RESULT: Found {intermediate_states['metadata']['count']} intermediate states")
    print(f"中间态列表: {intermediate_states['metadata']['intermediate_states']}")
    
    # 步骤4：分析所有跃迁路径及概率
    print("\n步骤4：计算所有跃迁路径的概率")
    paths_analysis = analyze_all_transition_paths(*initial_state, *final_state)
    print(f"FUNCTION_CALL: analyze_all_transition_paths | PARAMS: {initial_state + final_state} | RESULT: {paths_analysis['metadata']}")
    
    paths = paths_analysis['result']
    
    # 输出最可能的路径
    print("\n最可能的跃迁路径：")
    if paths:
        most_probable = paths[0]
        n_inter, l_inter, m_inter = most_probable['intermediate_state']
        prob = most_probable.get('normalized_probability', most_probable['probability'])
        
        print(f"  路径: |3,0,0⟩ → |{n_inter},{l_inter},{m_inter}⟩ → |1,0,0⟩")
        print(f"  光谱符号: 3s → {n_inter}{_l_to_spectroscopic(l_inter)} → 1s")
        print(f"  归一化概率: {prob:.6f} ({prob*100:.2f}%)")
        
        # 验证答案
        expected_intermediate = (2, 1, 0)
        expected_prob = 1.0 / 3.0
        
        if (n_inter, l_inter, m_inter) == expected_intermediate:
            print(f"\n✓ 中间态匹配标准答案: |2,1,0⟩ (2p)")
        
        print(f"\n标准答案概率: {expected_prob:.6f}")
        print(f"计算得到概率: {prob:.6f}")
        print(f"相对误差: {abs(prob - expected_prob)/expected_prob * 100:.2f}%")
    
    # 步骤5：生成可视化
    print("\n步骤5：生成能级图和跃迁路径可视化")
    energy_plot = visualize_energy_levels_and_transitions(*initial_state, *final_state, paths)
    print(f"FUNCTION_CALL: visualize_energy_levels_and_transitions | RESULT: {energy_plot}")
    
    prob_plot = visualize_transition_probabilities(paths)
    print(f"FUNCTION_CALL: visualize_transition_probabilities | RESULT: {prob_plot}")
    
    # 步骤6：生成完整报告
    print("\n步骤6：生成跃迁分析报告")
    report = create_transition_summary_report(*initial_state, *final_state, paths)
    print(f"FUNCTION_CALL: create_transition_summary_report | RESULT: {report}")
    
    # 最终答案
    if paths:
        most_probable = paths[0]
        n_inter, l_inter, m_inter = most_probable['intermediate_state']
        prob = most_probable.get('normalized_probability', most_probable['probability'])
        
        answer = f"|3,0,0⟩ → |{n_inter},{l_inter},{m_inter}⟩ → |1,0,0⟩, probability = {prob:.6f}"
        print(f"\nFINAL_ANSWER: {answer}")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景2：分析 |4,0,0⟩ → |2,0,0⟩ 的两步跃迁")
    print("=" * 80)
    print("问题描述：4s态到2s态的跃迁路径分析")
    print("-" * 80)
    
    initial_state_2 = (4, 0, 0)
    final_state_2 = (2, 0, 0)
    
    print("\n步骤1：验证量子态")
    valid_2 = validate_quantum_state(*initial_state_2)
    print(f"FUNCTION_CALL: validate_quantum_state | PARAMS: {initial_state_2} | RESULT: {valid_2}")
    
    print("\n步骤2：检查直接跃迁")
    direct_2 = check_dipole_selection_rule(*initial_state_2, *final_state_2)
    print(f"FUNCTION_CALL: check_dipole_selection_rule | PARAMS: {initial_state_2 + final_state_2} | RESULT: {direct_2}")
    
    print("\n步骤3：寻找中间态并计算概率")
    paths_2 = analyze_all_transition_paths(*initial_state_2, *final_state_2)
    print(f"FUNCTION_CALL: analyze_all_transition_paths | PARAMS: {initial_state_2 + final_state_2} | RESULT: Found {paths_2['metadata']['total_paths']} paths")
    
    if paths_2['result']:
        print("\n前3个最可能的路径：")
        for i, path in enumerate(paths_2['result'][:3], 1):
            n_i, l_i, m_i = path['intermediate_state']
            prob = path.get('normalized_probability', path['probability'])
            print(f"  {i}. |4,0,0⟩ → |{n_i},{l_i},{m_i}⟩ → |2,0,0⟩, P = {prob:.4f}")
    
    print("\n步骤4：生成可视化")
    vis_2 = visualize_energy_levels_and_transitions(*initial_state_2, *final_state_2, paths_2['result'])
    print(f"FUNCTION_CALL: visualize_energy_levels_and_transitions | RESULT: {vis_2}")
    
    if paths_2['result']:
        most_prob_2 = paths_2['result'][0]
        n_i, l_i, m_i = most_prob_2['intermediate_state']
        prob_2 = most_prob_2.get('normalized_probability', most_prob_2['probability'])
        answer_2 = f"|4,0,0⟩ → |{n_i},{l_i},{m_i}⟩ → |2,0,0⟩, probability = {prob_2:.6f}"
        print(f"\nFINAL_ANSWER: {answer_2}")
    
    
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("场景3：比较不同磁量子数的跃迁概率 - |3,1,1⟩ → |1,0,0⟩")
    print("=" * 80)
    print("问题描述：分析磁量子数对跃迁路径和概率的影响")
    print("-" * 80)
    
    initial_state_3 = (3, 1, 1)
    final_state_3 = (1, 0, 0)
    
    print("\n步骤1：验证量子态")
    valid_3 = validate_quantum_state(*initial_state_3)
    print(f"FUNCTION_CALL: validate_quantum_state | PARAMS: {initial_state_3} | RESULT: {valid_3}")
    
    print("\n步骤2：分析跃迁路径")
    paths_3 = analyze_all_transition_paths(*initial_state_3, *final_state_3)
    print(f"FUNCTION_CALL: analyze_all_transition_paths | PARAMS: {initial_state_3 + final_state_3} | RESULT: Found {paths_3['metadata']['total_paths']} paths")
    
    print("\n步骤3：比较不同中间态的概率")
    if paths_3['result']:
        print("\n所有可能的跃迁路径：")
        for i, path in enumerate(paths_3['result'], 1):
            n_i, l_i, m_i = path['intermediate_state']
            prob = path.get('normalized_probability', path['probability'])
            print(f"  {i}. |3,1,1⟩ → |{n_i},{l_i},{m_i}⟩ → |1,0,0⟩")
            print(f"     Probability: {prob:.6f} ({prob*100:.2f}%)")
    
    print("\n步骤4：生成概率分布图")
    prob_vis_3 = visualize_transition_probabilities(paths_3['result'])
    print(f"FUNCTION_CALL: visualize_transition_probabilities | RESULT: {prob_vis_3}")
    
    print("\n步骤5：生成详细报告")
    report_3 = create_transition_summary_report(*initial_state_3, *final_state_3, paths_3['result'])
    print(f"FUNCTION_CALL: create_transition_summary_report | RESULT: {report_3}")
    
    if paths_3['result']:
        most_prob_3 = paths_3['result'][0]
        n_i, l_i, m_i = most_prob_3['intermediate_state']
        prob_3 = most_prob_3.get('normalized_probability', most_prob_3['probability'])
        answer_3 = f"|3,1,1⟩ → |{n_i},{l_i},{m_i}⟩ → |1,0,0⟩, probability = {prob_3:.6f}"
        print(f"\nFINAL_ANSWER: {answer_3}")
    
    print("\n" + "=" * 80)
    print("所有场景演示完成")
    print("=" * 80)


if __name__ == "__main__":
    main()