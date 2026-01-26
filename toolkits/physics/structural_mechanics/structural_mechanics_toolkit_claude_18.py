# Filename: structural_mechanics_toolkit.py

"""
结构力学工具包 - 悬臂梁挠度分析
Structural Mechanics Toolkit - Cantilever Beam Deflection Analysis

本工具包用于分析复杂悬臂梁系统的挠度、弯矩和剪力分布
适用于能量收集装置等工程应用

主要功能：
1. 单段悬臂梁挠度计算
2. 分支悬臂梁系统分析
3. 刚性节点力学分析
4. 挠度曲线可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import json
import os
from scipy.integrate import odeint
from scipy.optimize import fsolve

# 配置matplotlib字体，避免中文乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建结果保存目录
os.makedirs('./mid_result/structural_mechanics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# ============================================================================
# 第一层：原子函数 - 基础力学计算
# ============================================================================

def calculate_cantilever_deflection_point_load(
    force: float,
    length: float,
    flexural_rigidity: float,
    position: float
) -> Dict[str, Union[float, dict]]:
    """
    计算悬臂梁在集中力作用下某点的挠度
    
    基于欧拉-伯努利梁理论：
    w(x) = (F/(6EI)) * x^2 * (3L - x)  当力作用在自由端时
    
    Parameters:
    -----------
    force : float
        集中力大小 (N)，向下为正
    length : float
        从固定端到力作用点的距离 (m)
    flexural_rigidity : float
        抗弯刚度 EI (N·m²)
    position : float
        计算挠度的位置，从固定端测量 (m)
        
    Returns:
    --------
    dict : {
        'result': float,  # 挠度值 (m)，向下为正
        'metadata': {
            'force': float,
            'length': float,
            'EI': float,
            'position': float,
            'formula': str
        }
    }
    
    Raises:
    -------
    ValueError: 参数不合法时
    """
    # 参数验证
    if flexural_rigidity <= 0:
        raise ValueError(f"抗弯刚度必须为正值，当前值: {flexural_rigidity}")
    if length <= 0:
        raise ValueError(f"梁长必须为正值，当前值: {length}")
    if position < 0 or position > length:
        raise ValueError(f"位置必须在 [0, {length}] 范围内，当前值: {position}")
    
    # 计算挠度
    x = position
    L = length
    EI = flexural_rigidity
    F = force
    
    # 悬臂梁自由端受集中力的挠度公式
    deflection = (F / (6 * EI)) * x**2 * (3 * L - x)
    
    return {
        'result': deflection,
        'metadata': {
            'force': force,
            'length': length,
            'EI': flexural_rigidity,
            'position': position,
            'formula': 'w(x) = (F/(6EI)) * x^2 * (3L - x)',
            'unit': 'm'
        }
    }


def calculate_cantilever_slope_point_load(
    force: float,
    length: float,
    flexural_rigidity: float,
    position: float
) -> Dict[str, Union[float, dict]]:
    """
    计算悬臂梁在集中力作用下某点的转角
    
    基于欧拉-伯努利梁理论：
    θ(x) = dw/dx = (F/(2EI)) * x * (2L - x)
    
    Parameters:
    -----------
    force : float
        集中力大小 (N)
    length : float
        从固定端到力作用点的距离 (m)
    flexural_rigidity : float
        抗弯刚度 EI (N·m²)
    position : float
        计算转角的位置 (m)
        
    Returns:
    --------
    dict : {
        'result': float,  # 转角 (rad)
        'metadata': {...}
    }
    """
    if flexural_rigidity <= 0:
        raise ValueError(f"抗弯刚度必须为正值: {flexural_rigidity}")
    if length <= 0:
        raise ValueError(f"梁长必须为正值: {length}")
    if position < 0 or position > length:
        raise ValueError(f"位置超出范围 [0, {length}]: {position}")
    
    x = position
    L = length
    EI = flexural_rigidity
    F = force
    
    # 转角公式（挠度的一阶导数）
    slope = (F / (2 * EI)) * x * (2 * L - x)
    
    return {
        'result': slope,
        'metadata': {
            'force': force,
            'length': length,
            'EI': flexural_rigidity,
            'position': position,
            'formula': 'θ(x) = (F/(2EI)) * x * (2L - x)',
            'unit': 'rad'
        }
    }


def calculate_moment_at_position(
    force: float,
    force_position: float,
    calc_position: float
) -> Dict[str, Union[float, dict]]:
    """
    计算悬臂梁某点的弯矩
    
    Parameters:
    -----------
    force : float
        集中力大小 (N)
    force_position : float
        力作用点距固定端的距离 (m)
    calc_position : float
        计算弯矩的位置 (m)
        
    Returns:
    --------
    dict : {
        'result': float,  # 弯矩 (N·m)
        'metadata': {...}
    }
    """
    if calc_position < 0 or calc_position > force_position:
        raise ValueError(f"计算位置必须在 [0, {force_position}] 范围内")
    
    # 悬臂梁弯矩：M(x) = -F * (L - x)
    moment = -force * (force_position - calc_position)
    
    return {
        'result': moment,
        'metadata': {
            'force': force,
            'force_position': force_position,
            'calc_position': calc_position,
            'formula': 'M(x) = -F * (L - x)',
            'unit': 'N·m'
        }
    }


def superpose_deflections(
    deflection_list: List[float]
) -> Dict[str, Union[float, dict]]:
    """
    叠加多个挠度值（线性叠加原理）
    
    Parameters:
    -----------
    deflection_list : List[float]
        挠度值列表 (m)
        
    Returns:
    --------
    dict : {
        'result': float,  # 总挠度 (m)
        'metadata': {...}
    }
    """
    if not deflection_list:
        raise ValueError("挠度列表不能为空")
    
    total_deflection = sum(deflection_list)
    
    return {
        'result': total_deflection,
        'metadata': {
            'individual_deflections': deflection_list,
            'count': len(deflection_list),
            'principle': 'Linear superposition',
            'unit': 'm'
        }
    }


# ============================================================================
# 第二层：组合函数 - 复杂结构分析
# ============================================================================

def analyze_rigid_joint_forces(
    weight_c: float,
    weight_d: float,
    length_bc: float,
    length_bd: float,
    ei_bc: float,
    ei_bd: float
) -> Dict[str, Union[dict, dict]]:
    """
    分析刚性节点B处的力和力矩
    
    对于对称结构（BC和BD平行且参数相同），刚性节点将力和力矩
    传递给主梁AB
    
    Parameters:
    -----------
    weight_c : float
        C端重物重量 (N)
    weight_d : float
        D端重物重量 (N)
    length_bc : float
        BC梁长度 (m)
    length_bd : float
        BD梁长度 (m)
    ei_bc : float
        BC梁抗弯刚度 (N·m²)
    ei_bd : float
        BD梁抗弯刚度 (N·m²)
        
    Returns:
    --------
    dict : {
        'result': {
            'total_force': float,  # 总竖向力 (N)
            'total_moment': float,  # 总弯矩 (N·m)
            'force_c': float,
            'force_d': float
        },
        'metadata': {...}
    }
    """
    # 参数验证
    if any(x <= 0 for x in [length_bc, length_bd, ei_bc, ei_bd]):
        raise ValueError("长度和刚度必须为正值")
    
    # 计算每个分支传递到节点B的力
    force_c = weight_c
    force_d = weight_d
    
    # 总竖向力
    total_force = force_c + force_d
    
    # 计算每个分支在节点B处产生的弯矩
    # 对于悬臂梁，自由端受力在固定端产生的弯矩为 M = F * L
    moment_from_c = weight_c * length_bc
    moment_from_d = weight_d * length_bd
    
    # 总弯矩（两个分支的弯矩叠加）
    total_moment = moment_from_c + moment_from_d
    
    return {
        'result': {
            'total_force': total_force,
            'total_moment': total_moment,
            'force_c': force_c,
            'force_d': force_d,
            'moment_from_c': moment_from_c,
            'moment_from_d': moment_from_d
        },
        'metadata': {
            'weight_c': weight_c,
            'weight_d': weight_d,
            'length_bc': length_bc,
            'length_bd': length_bd,
            'ei_bc': ei_bc,
            'ei_bd': ei_bd,
            'analysis_method': 'Rigid joint force equilibrium'
        }
    }


def calculate_branched_cantilever_deflection(
    weight: float,
    length_ab: float,
    length_branch: float,
    ei_ab: float,
    ei_branch: float,
    num_branches: int = 2
) -> Dict[str, Union[float, dict]]:
    """
    计算分支悬臂梁系统末端的挠度
    
    系统结构：
    - 主梁AB：长度length_ab，刚度ei_ab，固定端在A
    - 分支梁：从B点延伸，长度length_branch，刚度ei_branch
    - 每个分支末端承受重量weight
    
    分析方法：
    1. 计算分支梁自身的挠度
    2. 计算主梁在节点B处的挠度和转角
    3. 叠加得到分支末端总挠度
    
    Parameters:
    -----------
    weight : float
        每个分支末端的重量 (N)
    length_ab : float
        主梁AB长度 (m)
    length_branch : float
        分支梁长度 (m)
    ei_ab : float
        主梁抗弯刚度 (N·m²)
    ei_branch : float
        分支梁抗弯刚度 (N·m²)
    num_branches : int
        分支数量，默认2
        
    Returns:
    --------
    dict : {
        'result': float,  # 分支末端总挠度 (m)
        'metadata': {
            'deflection_branch': float,  # 分支自身挠度
            'deflection_at_b': float,    # B点挠度
            'slope_at_b': float,         # B点转角
            'additional_deflection': float  # 由B点运动引起的附加挠度
        }
    }
    """
    # 参数验证
    if any(x <= 0 for x in [length_ab, length_branch, ei_ab, ei_branch]):
        raise ValueError("长度和刚度必须为正值")
    if weight < 0:
        raise ValueError(f"重量不能为负: {weight}")
    if num_branches <= 0:
        raise ValueError(f"分支数量必须为正整数: {num_branches}")
    
    # 步骤1：计算分支梁自身的挠度（将分支视为独立悬臂梁）
    branch_deflection_result = calculate_cantilever_deflection_point_load(
        force=weight,
        length=length_branch,
        flexural_rigidity=ei_branch,
        position=length_branch
    )
    deflection_branch = branch_deflection_result['result']
    
    # 步骤2：分析刚性节点B处的力和力矩
    # 所有分支的力和力矩作用在主梁AB上
    total_force_at_b = num_branches * weight
    total_moment_at_b = num_branches * weight * length_branch
    
    # 步骤3：计算主梁在B点（自由端）的挠度
    # 主梁受到集中力和集中力矩的共同作用
    
    # 3a. 集中力引起的挠度
    deflection_b_force = calculate_cantilever_deflection_point_load(
        force=total_force_at_b,
        length=length_ab,
        flexural_rigidity=ei_ab,
        position=length_ab
    )['result']
    
    # 3b. 集中力矩引起的挠度
    # 悬臂梁自由端受集中力矩M的挠度：w = M*L^2 / (2EI)
    deflection_b_moment = (total_moment_at_b * length_ab**2) / (2 * ei_ab)
    
    # 3c. B点总挠度
    deflection_at_b = deflection_b_force + deflection_b_moment
    
    # 步骤4：计算主梁在B点的转角
    # 4a. 集中力引起的转角
    slope_b_force = calculate_cantilever_slope_point_load(
        force=total_force_at_b,
        length=length_ab,
        flexural_rigidity=ei_ab,
        position=length_ab
    )['result']
    
    # 4b. 集中力矩引起的转角
    # 悬臂梁自由端受集中力矩M的转角：θ = M*L / (EI)
    slope_b_moment = (total_moment_at_b * length_ab) / ei_ab
    
    # 4c. B点总转角
    slope_at_b = slope_b_force + slope_b_moment
    
    # 步骤5：计算分支末端由于B点运动产生的附加挠度
    # 由于B点的转角，分支末端会有额外的竖向位移
    additional_deflection = slope_at_b * length_branch
    
    # 步骤6：叠加所有挠度分量
    total_deflection = deflection_branch + deflection_at_b + additional_deflection
    
    return {
        'result': total_deflection,
        'metadata': {
            'deflection_branch': deflection_branch,
            'deflection_at_b': deflection_at_b,
            'deflection_b_force': deflection_b_force,
            'deflection_b_moment': deflection_b_moment,
            'slope_at_b': slope_at_b,
            'slope_b_force': slope_b_force,
            'slope_b_moment': slope_b_moment,
            'additional_deflection': additional_deflection,
            'total_force_at_b': total_force_at_b,
            'total_moment_at_b': total_moment_at_b,
            'num_branches': num_branches,
            'unit': 'm'
        }
    }


def calculate_deflection_to_weight_ratio(
    length_ab: float,
    length_branch: float,
    ei_ab: float,
    ei_branch: float,
    num_branches: int = 2,
    unit_weight: float = 1.0
) -> Dict[str, Union[float, dict]]:
    """
    计算挠度与重量的比值 f_D/W
    
    这是能量收集装置的关键性能参数
    
    Parameters:
    -----------
    length_ab : float
        主梁长度 (m)
    length_branch : float
        分支梁长度 (m)
    ei_ab : float
        主梁抗弯刚度 (N·m²)
    ei_branch : float
        分支梁抗弯刚度 (N·m²)
    num_branches : int
        分支数量
    unit_weight : float
        单位重量，用于计算比值，默认1.0 N
        
    Returns:
    --------
    dict : {
        'result': float,  # f_D/W 比值 (m/N)
        'metadata': {...}
    }
    """
    # 计算单位重量下的挠度
    deflection_result = calculate_branched_cantilever_deflection(
        weight=unit_weight,
        length_ab=length_ab,
        length_branch=length_branch,
        ei_ab=ei_ab,
        ei_branch=ei_branch,
        num_branches=num_branches
    )
    
    deflection = deflection_result['result']
    ratio = deflection / unit_weight
    
    return {
        'result': ratio,
        'metadata': {
            'deflection': deflection,
            'weight': unit_weight,
            'ratio': ratio,
            'unit': 'm/N',
            'deflection_components': deflection_result['metadata']
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def plot_deflection_curve(
    length_ab: float,
    length_branch: float,
    ei_ab: float,
    ei_branch: float,
    weight: float,
    num_branches: int = 2,
    num_points: int = 50
) -> Dict[str, Union[str, dict]]:
    """
    绘制悬臂梁系统的挠度曲线
    
    Parameters:
    -----------
    length_ab : float
        主梁长度 (m)
    length_branch : float
        分支梁长度 (m)
    ei_ab : float
        主梁抗弯刚度 (N·m²)
    ei_branch : float
        分支梁抗弯刚度 (N·m²)
    weight : float
        分支末端重量 (N)
    num_branches : int
        分支数量
    num_points : int
        绘图点数
        
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': {...}
    }
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 计算主梁AB的挠度曲线
    x_ab = np.linspace(0, length_ab, num_points)
    deflection_ab = []
    
    total_force_at_b = num_branches * weight
    total_moment_at_b = num_branches * weight * length_branch
    
    for x in x_ab:
        # 集中力引起的挠度
        d_force = calculate_cantilever_deflection_point_load(
            force=total_force_at_b,
            length=length_ab,
            flexural_rigidity=ei_ab,
            position=x
        )['result']
        
        # 集中力矩引起的挠度
        d_moment = (total_moment_at_b * x**2) / (2 * ei_ab)
        
        deflection_ab.append(d_force + d_moment)
    
    # 绘制主梁挠度
    ax1.plot(x_ab, deflection_ab, 'b-', linewidth=2, label='Main Beam AB')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Position along AB (m)', fontsize=12)
    ax1.set_ylabel('Deflection (m)', fontsize=12)
    ax1.set_title('Deflection Curve of Main Beam AB', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.invert_yaxis()  # 向下为正
    
    # 计算分支梁的挠度曲线（相对于B点）
    x_branch = np.linspace(0, length_branch, num_points)
    deflection_branch = []
    
    # B点的挠度和转角
    deflection_at_b = deflection_ab[-1]
    slope_b_force = calculate_cantilever_slope_point_load(
        force=total_force_at_b,
        length=length_ab,
        flexural_rigidity=ei_ab,
        position=length_ab
    )['result']
    slope_b_moment = (total_moment_at_b * length_ab) / ei_ab
    slope_at_b = slope_b_force + slope_b_moment
    
    for x in x_branch:
        # 分支自身挠度
        d_self = calculate_cantilever_deflection_point_load(
            force=weight,
            length=length_branch,
            flexural_rigidity=ei_branch,
            position=x
        )['result']
        
        # B点运动引起的附加挠度
        d_additional = slope_at_b * x
        
        # 总挠度（相对于固定端A）
        total_d = deflection_at_b + d_self + d_additional
        deflection_branch.append(total_d)
    
    # 绘制分支梁挠度
    x_branch_global = length_ab + x_branch
    ax2.plot(x_branch_global, deflection_branch, 'r-', linewidth=2, 
             label=f'Branch Beam (×{num_branches})')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=length_ab, color='g', linestyle='--', alpha=0.5, 
                label='Joint B')
    ax2.set_xlabel('Position from A (m)', fontsize=12)
    ax2.set_ylabel('Deflection (m)', fontsize=12)
    ax2.set_title('Deflection Curve of Branch Beam', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/cantilever_deflection_curve.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG image',
            'max_deflection_ab': max(deflection_ab),
            'max_deflection_branch': max(deflection_branch),
            'num_points': num_points,
            'weight': weight,
            'num_branches': num_branches
        }
    }


def plot_parameter_sensitivity(
    length_ab: float,
    length_branch: float,
    ei_ab: float,
    ei_branch: float,
    num_branches: int = 2
) -> Dict[str, Union[str, dict]]:
    """
    绘制挠度比值对参数的敏感性分析图
    
    分析参数变化对 f_D/W 的影响
    
    Parameters:
    -----------
    length_ab : float
        主梁基准长度 (m)
    length_branch : float
        分支梁基准长度 (m)
    ei_ab : float
        主梁基准刚度 (N·m²)
    ei_branch : float
        分支梁基准刚度 (N·m²)
    num_branches : int
        分支数量
        
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': {...}
    }
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 参数1：主梁长度变化
    length_ab_range = np.linspace(0.5 * length_ab, 1.5 * length_ab, 30)
    ratios_lab = []
    for l in length_ab_range:
        ratio = calculate_deflection_to_weight_ratio(
            length_ab=l,
            length_branch=length_branch,
            ei_ab=ei_ab,
            ei_branch=ei_branch,
            num_branches=num_branches
        )['result']
        ratios_lab.append(ratio)
    
    axes[0, 0].plot(length_ab_range, ratios_lab, 'b-', linewidth=2)
    axes[0, 0].axvline(x=length_ab, color='r', linestyle='--', 
                       label=f'Baseline: {length_ab}m')
    axes[0, 0].set_xlabel('Main Beam Length (m)', fontsize=11)
    axes[0, 0].set_ylabel('f_D/W (m/N)', fontsize=11)
    axes[0, 0].set_title('Sensitivity to Main Beam Length', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 参数2：分支梁长度变化
    length_branch_range = np.linspace(0.5 * length_branch, 1.5 * length_branch, 30)
    ratios_lbr = []
    for l in length_branch_range:
        ratio = calculate_deflection_to_weight_ratio(
            length_ab=length_ab,
            length_branch=l,
            ei_ab=ei_ab,
            ei_branch=ei_branch,
            num_branches=num_branches
        )['result']
        ratios_lbr.append(ratio)
    
    axes[0, 1].plot(length_branch_range, ratios_lbr, 'g-', linewidth=2)
    axes[0, 1].axvline(x=length_branch, color='r', linestyle='--',
                       label=f'Baseline: {length_branch}m')
    axes[0, 1].set_xlabel('Branch Beam Length (m)', fontsize=11)
    axes[0, 1].set_ylabel('f_D/W (m/N)', fontsize=11)
    axes[0, 1].set_title('Sensitivity to Branch Beam Length', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 参数3：主梁刚度变化
    ei_ab_range = np.linspace(0.5 * ei_ab, 1.5 * ei_ab, 30)
    ratios_eiab = []
    for ei in ei_ab_range:
        ratio = calculate_deflection_to_weight_ratio(
            length_ab=length_ab,
            length_branch=length_branch,
            ei_ab=ei,
            ei_branch=ei_branch,
            num_branches=num_branches
        )['result']
        ratios_eiab.append(ratio)
    
    axes[1, 0].plot(ei_ab_range, ratios_eiab, 'm-', linewidth=2)
    axes[1, 0].axvline(x=ei_ab, color='r', linestyle='--',
                       label=f'Baseline: {ei_ab}N·m²')
    axes[1, 0].set_xlabel('Main Beam Stiffness EI (N·m²)', fontsize=11)
    axes[1, 0].set_ylabel('f_D/W (m/N)', fontsize=11)
    axes[1, 0].set_title('Sensitivity to Main Beam Stiffness', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 参数4：分支梁刚度变化
    ei_branch_range = np.linspace(0.5 * ei_branch, 1.5 * ei_branch, 30)
    ratios_eibr = []
    for ei in ei_branch_range:
        ratio = calculate_deflection_to_weight_ratio(
            length_ab=length_ab,
            length_branch=length_branch,
            ei_ab=ei_ab,
            ei_branch=ei,
            num_branches=num_branches
        )['result']
        ratios_eibr.append(ratio)
    
    axes[1, 1].plot(ei_branch_range, ratios_eibr, 'c-', linewidth=2)
    axes[1, 1].axvline(x=ei_branch, color='r', linestyle='--',
                       label=f'Baseline: {ei_branch}N·m²')
    axes[1, 1].set_xlabel('Branch Beam Stiffness EI (N·m²)', fontsize=11)
    axes[1, 1].set_ylabel('f_D/W (m/N)', fontsize=11)
    axes[1, 1].set_title('Sensitivity to Branch Beam Stiffness', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/parameter_sensitivity_analysis.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG image',
            'parameters_analyzed': ['length_ab', 'length_branch', 'ei_ab', 'ei_branch'],
            'baseline_values': {
                'length_ab': length_ab,
                'length_branch': length_branch,
                'ei_ab': ei_ab,
                'ei_branch': ei_branch
            }
        }
    }


def visualize_beam_structure(
    length_ab: float,
    length_branch: float,
    weight: float,
    num_branches: int = 2
) -> Dict[str, Union[str, dict]]:
    """
    可视化悬臂梁结构示意图
    
    Parameters:
    -----------
    length_ab : float
        主梁长度 (m)
    length_branch : float
        分支梁长度 (m)
    weight : float
        末端重量 (N)
    num_branches : int
        分支数量
        
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': {...}
    }
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制固定端A
    ax.plot([0, 0], [-0.3, 0.3], 'k-', linewidth=8, label='Fixed Support A')
    # 固定端符号
    for y in np.linspace(-0.3, 0.3, 7):
        ax.plot([0, -0.1], [y, y-0.05], 'k-', linewidth=1)
    
    # 绘制主梁AB
    ax.plot([0, length_ab], [0, 0], 'b-', linewidth=6, label='Main Beam AB (2EI)')
    ax.plot(length_ab, 0, 'go', markersize=15, label='Rigid Joint B')
    
    # 绘制分支梁
    branch_offset = 0.15
    if num_branches == 2:
        # 上分支（BC）
        ax.plot([length_ab, length_ab + length_branch], 
                [branch_offset, branch_offset], 
                'r-', linewidth=4, label='Branch Beam BC (EI)')
        ax.plot(length_ab + length_branch, branch_offset, 'rs', 
                markersize=12, label=f'Weight C: {weight}N')
        
        # 下分支（BD）
        ax.plot([length_ab, length_ab + length_branch], 
                [-branch_offset, -branch_offset], 
                'r-', linewidth=4, label='Branch Beam BD (EI)')
        ax.plot(length_ab + length_branch, -branch_offset, 'rs', 
                markersize=12, label=f'Weight D: {weight}N')
        
        # 绘制重物
        rect_width = 0.15
        rect_height = 0.1
        from matplotlib.patches import Rectangle
        rect_c = Rectangle((length_ab + length_branch - rect_width/2, 
                           branch_offset - rect_height/2),
                          rect_width, rect_height, 
                          facecolor='orange', edgecolor='black', linewidth=2)
        rect_d = Rectangle((length_ab + length_branch - rect_width/2, 
                           -branch_offset - rect_height/2),
                          rect_width, rect_height, 
                          facecolor='orange', edgecolor='black', linewidth=2)
        ax.add_patch(rect_c)
        ax.add_patch(rect_d)
        
        # 绘制重力箭头
        arrow_length = 0.2
        ax.arrow(length_ab + length_branch, branch_offset - rect_height/2, 
                0, -arrow_length, head_width=0.08, head_length=0.05, 
                fc='red', ec='red', linewidth=2)
        ax.arrow(length_ab + length_branch, -branch_offset - rect_height/2, 
                0, -arrow_length, head_width=0.08, head_length=0.05, 
                fc='red', ec='red', linewidth=2)
    
    # 标注尺寸
    # 主梁长度
    ax.annotate('', xy=(length_ab, -0.5), xytext=(0, -0.5),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    ax.text(length_ab/2, -0.6, f'l = {length_ab}m', 
            ha='center', fontsize=12, fontweight='bold')
    
    # 分支梁长度
    ax.annotate('', xy=(length_ab + length_branch, 0.5), 
                xytext=(length_ab, 0.5),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='black'))
    ax.text(length_ab + length_branch/2, 0.6, f'l = {length_branch}m', 
            ha='center', fontsize=12, fontweight='bold')
    
    # 标注节点
    ax.text(0, 0.4, 'A', fontsize=14, fontweight='bold', ha='center')
    ax.text(length_ab, -0.4, 'B', fontsize=14, fontweight='bold', ha='center')
    ax.text(length_ab + length_branch, branch_offset + 0.25, 'C', 
            fontsize=14, fontweight='bold', ha='center')
    ax.text(length_ab + length_branch, -branch_offset - 0.25, 'D', 
            fontsize=14, fontweight='bold', ha='center')
    
    ax.set_xlim(-0.5, length_ab + length_branch + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('Horizontal Position (m)', fontsize=12)
    ax.set_ylabel('Vertical Position (m)', fontsize=12)
    ax.set_title('Energy Harvesting Device - Cantilever Beam Structure', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/beam_structure_diagram.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG image',
            'structure': 'Branched cantilever beam',
            'num_branches': num_branches,
            'total_length': length_ab + length_branch
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    主函数：演示结构力学工具包的三个应用场景
    """
    
    print("=" * 80)
    print("结构力学工具包 - 悬臂梁挠度分析演示")
    print("Structural Mechanics Toolkit - Cantilever Beam Deflection Analysis")
    print("=" * 80)
    print()
    
    # ========================================================================
    # 场景1：解决原始问题 - 能量收集装置的挠度分析
    # ========================================================================
    print("=" * 80)
    print("场景1：能量收集装置的挠度与重量比值计算")
    print("Scenario 1: Deflection-to-Weight Ratio for Energy Harvesting Device")
    print("=" * 80)
    print("问题描述：")
    print("一个能量收集装置简化为悬臂梁模型：")
    print("- 主梁AB：长度l，抗弯刚度2EI，固定端在A")
    print("- 分支梁BC和BD：各长度l，抗弯刚度EI，从刚性节点B延伸")
    print("- 两个分支末端C和D各承受重量W")
    print("求：D端挠度与重量的比值 f_D/W")
    print("-" * 80)
    
    # 定义符号参数（使用具体数值进行演示）
    l = 1.0  # 梁长度 (m)
    E = 1.0  # 弹性模量 (Pa)，用于演示
    I = 1.0  # 惯性矩 (m^4)，用于演示
    EI = E * I  # 抗弯刚度
    W = 1.0  # 单位重量 (N)
    
    print(f"参数设置（用于演示）：")
    print(f"  梁长度 l = {l} m")
    print(f"  抗弯刚度单位 EI = {EI} N·m²")
    print(f"  主梁刚度 = 2EI = {2*EI} N·m²")
    print(f"  分支梁刚度 = EI = {EI} N·m²")
    print(f"  单位重量 W = {W} N")
    print()
    
    # 步骤1：可视化结构
    print("步骤1：绘制悬臂梁结构示意图")
    print("调用函数：visualize_beam_structure()")
    structure_result = visualize_beam_structure(
        length_ab=l,
        length_branch=l,
        weight=W,
        num_branches=2
    )
    print(f"FUNCTION_CALL: visualize_beam_structure | PARAMS: {{length_ab: {l}, length_branch: {l}, weight: {W}, num_branches: 2}} | RESULT: {structure_result}")
    print()
    
    # 步骤2：计算分支梁自身的挠度
    print("步骤2：计算分支梁BD自身的挠度（将BD视为独立悬臂梁）")
    print("调用函数：calculate_cantilever_deflection_point_load()")
    branch_deflection = calculate_cantilever_deflection_point_load(
        force=W,
        length=l,
        flexural_rigidity=EI,
        position=l
    )
    print(f"FUNCTION_CALL: calculate_cantilever_deflection_point_load | PARAMS: {{force: {W}, length: {l}, flexural_rigidity: {EI}, position: {l}}} | RESULT: {branch_deflection}")
    print(f"  分支梁自身挠度 = {branch_deflection['result']:.6e} m")
    print()
    
    # 步骤3：分析刚性节点B处的力和力矩
    print("步骤3：分析刚性节点B处的力和力矩")
    print("调用函数：analyze_rigid_joint_forces()")
    joint_forces = analyze_rigid_joint_forces(
        weight_c=W,
        weight_d=W,
        length_bc=l,
        length_bd=l,
        ei_bc=EI,
        ei_bd=EI
    )
    print(f"FUNCTION_CALL: analyze_rigid_joint_forces | PARAMS: {{weight_c: {W}, weight_d: {W}, length_bc: {l}, length_bd: {l}, ei_bc: {EI}, ei_bd: {EI}}} | RESULT: {joint_forces}")
    print(f"  节点B处总竖向力 = {joint_forces['result']['total_force']} N")
    print(f"  节点B处总弯矩 = {joint_forces['result']['total_moment']} N·m")
    print()
    
    # 步骤4：计算主梁在B点的挠度（集中力作用）
    print("步骤4：计算主梁在B点由集中力引起的挠度")
    print("调用函数：calculate_cantilever_deflection_point_load()")
    total_force = joint_forces['result']['total_force']
    deflection_b_force = calculate_cantilever_deflection_point_load(
        force=total_force,
        length=l,
        flexural_rigidity=2*EI,
        position=l
    )
    print(f"FUNCTION_CALL: calculate_cantilever_deflection_point_load | PARAMS: {{force: {total_force}, length: {l}, flexural_rigidity: {2*EI}, position: {l}}} | RESULT: {deflection_b_force}")
    print(f"  集中力引起的B点挠度 = {deflection_b_force['result']:.6e} m")
    print()
    
    # 步骤5：计算主梁在B点由集中力矩引起的挠度
    print("步骤5：计算主梁在B点由集中力矩引起的挠度")
    total_moment = joint_forces['result']['total_moment']
    deflection_b_moment = (total_moment * l**2) / (2 * 2 * EI)
    print(f"  使用公式：w = M*L²/(2EI)")
    print(f"  集中力矩引起的B点挠度 = {deflection_b_moment:.6e} m")
    print()
    
    # 步骤6：计算主梁在B点的转角
    print("步骤6：计算主梁在B点的转角")
    print("调用函数：calculate_cantilever_slope_point_load()")
    slope_b_force = calculate_cantilever_slope_point_load(
        force=total_force,
        length=l,
        flexural_rigidity=2*EI,
        position=l
    )
    print(f"FUNCTION_CALL: calculate_cantilever_slope_point_load | PARAMS: {{force: {total_force}, length: {l}, flexural_rigidity: {2*EI}, position: {l}}} | RESULT: {slope_b_force}")
    slope_b_moment = (total_moment * l) / (2 * EI)
    print(f"  集中力引起的转角 = {slope_b_force['result']:.6e} rad")
    print(f"  集中力矩引起的转角 = {slope_b_moment:.6e} rad")
    total_slope = slope_b_force['result'] + slope_b_moment
    print(f"  B点总转角 = {total_slope:.6e} rad")
    print()
    
    # 步骤7：计算D端总挠度
    print("步骤7：计算D端总挠度（叠加所有挠度分量）")
    print("调用函数：calculate_branched_cantilever_deflection()")
    total_deflection_result = calculate_branched_cantilever_deflection(
        weight=W,
        length_ab=l,
        length_branch=l,
        ei_ab=2*EI,
        ei_branch=EI,
        num_branches=2
    )
    print(f"FUNCTION_CALL: calculate_branched_cantilever_deflection | PARAMS: {{weight: {W}, length_ab: {l}, length_branch: {l}, ei_ab: {2*EI}, ei_branch: {EI}, num_branches: 2}} | RESULT: {total_deflection_result}")
    print(f"  分支自身挠度 = {total_deflection_result['metadata']['deflection_branch']:.6e} m")
    print(f"  B点挠度 = {total_deflection_result['metadata']['deflection_at_b']:.6e} m")
    print(f"  B点转角引起的附加挠度 = {total_deflection_result['metadata']['additional_deflection']:.6e} m")
    print(f"  D端总挠度 = {total_deflection_result['result']:.6e} m")
    print()
    
    # 步骤8：计算挠度与重量的比值
    print("步骤8：计算挠度与重量的比值 f_D/W")
    print("调用函数：calculate_deflection_to_weight_ratio()")
    ratio_result = calculate_deflection_to_weight_ratio(
        length_ab=l,
        length_branch=l,
        ei_ab=2*EI,
        ei_branch=EI,
        num_branches=2,
        unit_weight=W
    )
    print(f"FUNCTION_CALL: calculate_deflection_to_weight_ratio | PARAMS: {{length_ab: {l}, length_branch: {l}, ei_ab: {2*EI}, ei_branch: {EI}, num_branches: 2, unit_weight: {W}}} | RESULT: {ratio_result}")
    print()
    
    # 步骤9：验证理论公式
    print("步骤9：验证理论公式")
    print("根据结构力学理论，对于该分支悬臂梁系统：")
    print("f_D/W = 8l³/(3EI)")
    theoretical_ratio = (8 * l**3) / (3 * EI)
    print(f"  理论值 = {theoretical_ratio:.6e} m/N")
    print(f"  计算值 = {ratio_result['result']:.6e} m/N")
    print(f"  相对误差 = {abs(theoretical_ratio - ratio_result['result']) / theoretical_ratio * 100:.2e}%")
    print()
    
    # 步骤10：绘制挠度曲线
    print("步骤10：绘制挠度曲线")
    print("调用函数：plot_deflection_curve()")
    deflection_curve = plot_deflection_curve(
        length_ab=l,
        length_branch=l,
        ei_ab=2*EI,
        ei_branch=EI,
        weight=W,
        num_branches=2
    )
    print(f"FUNCTION_CALL: plot_deflection_curve | PARAMS: {{length_ab: {l}, length_branch: {l}, ei_ab: {2*EI}, ei_branch: {EI}, weight: {W}, num_branches: 2}} | RESULT: {deflection_curve}")
    print()
    
    print("-" * 80)
    print(f"场景1最终答案：f_D/W = {ratio_result['result']:.6e} m/N")
    print(f"理论公式表达：f_D/W = 8l³/(3EI)")
    print(f"FINAL_ANSWER: {ratio_result['result']:.6e} m/N (理论值: 8l³/(3EI) = {theoretical_ratio:.6e} m/N)")
    print()
    
    # ========================================================================
    # 场景2：参数敏感性分析
    # ========================================================================
    print("=" * 80)
    print("场景2：挠度比值对结构参数的敏感性分析")
    print("Scenario 2: Sensitivity Analysis of Deflection Ratio to Structural Parameters")
    print("=" * 80)
    print("问题描述：")
    print("分析主梁长度、分支梁长度、主梁刚度、分支梁刚度对挠度比值的影响")
    print("这对能量收集装置的优化设计具有重要意义")
    print("-" * 80)
    
    # 步骤1：绘制参数敏感性图
    print("步骤1：绘制参数敏感性分析图")
    print("调用函数：plot_parameter_sensitivity()")
    sensitivity_result = plot_parameter_sensitivity(
        length_ab=l,
        length_branch=l,
        ei_ab=2*EI,
        ei_branch=EI,
        num_branches=2
    )
    print(f"FUNCTION_CALL: plot_parameter_sensitivity | PARAMS: {{length_ab: {l}, length_branch: {l}, ei_ab: {2*EI}, ei_branch: {EI}, num_branches: 2}} | RESULT: {sensitivity_result}")
    print()
    
    # 步骤2：分析主梁长度的影响
    print("步骤2：定量分析主梁长度变化的影响")
    l_ab_test = [0.5, 1.0, 1.5]
    print(f"测试主梁长度：{l_ab_test} m")
    for l_test in l_ab_test:
        ratio_test = calculate_deflection_to_weight_ratio(
            length_ab=l_test,
            length_branch=l,
            ei_ab=2*EI,
            ei_branch=EI,
            num_branches=2
        )
        print(f"  l_AB = {l_test} m: f_D/W = {ratio_test['result']:.6e} m/N")
        print(f"    FUNCTION_CALL: calculate_deflection_to_weight_ratio | PARAMS: {{length_ab: {l_test}, ...}} | RESULT: {ratio_test}")
    print()
    
    # 步骤3：分析分支梁刚度的影响
    print("步骤3：定量分析分支梁刚度变化的影响")
    ei_branch_test = [0.5*EI, EI, 1.5*EI]
    print(f"测试分支梁刚度：{ei_branch_test} N·m²")
    for ei_test in ei_branch_test:
        ratio_test = calculate_deflection_to_weight_ratio(
            length_ab=l,
            length_branch=l,
            ei_ab=2*EI,
            ei_branch=ei_test,
            num_branches=2
        )
        print(f"  EI_branch = {ei_test} N·m²: f_D/W = {ratio_test['result']:.6e} m/N")
        print(f"    FUNCTION_CALL: calculate_deflection_to_weight_ratio | PARAMS: {{ei_branch: {ei_test}, ...}} | RESULT: {ratio_test}")
    print()
    
    print("-" * 80)
    print("场景2结论：")
    print("1. 挠度比值与主梁长度的三次方成正比")
    print("2. 挠度比值与分支梁长度的三次方成正比")
    print("3. 挠度比值与主梁刚度成反比")
    print("4. 挠度比值与分支梁刚度成反比")
    print("5. 主梁长度和刚度对挠度的影响最为显著")
    print(f"FINAL_ANSWER: 敏感性分析完成，结果保存在 {sensitivity_result['result']}")
    print()
    
    # ========================================================================
    # 场景3：不同分支数量的比较分析
    # ========================================================================
    print("=" * 80)
    print("场景3：不同分支数量对挠度的影响分析")
    print("Scenario 3: Effect of Number of Branches on Deflection")
    print("=" * 80)
    print("问题描述：")
    print("比较单分支、双分支、三分支悬臂梁系统的挠度特性")
    print("分析分支数量对能量收集效率的影响")
    print("-" * 80)
    
    # 步骤1：计算不同分支数量的挠度
    print("步骤1：计算不同分支数量下的挠度比值")
    branch_numbers = [1, 2, 3, 4]
    results_by_branches = []
    
    for n_branches in branch_numbers:
        ratio = calculate_deflection_to_weight_ratio(
            length_ab=l,
            length_branch=l,
            ei_ab=2*EI,
            ei_branch=EI,
            num_branches=n_branches,
            unit_weight=W
        )
        results_by_branches.append(ratio['result'])
        print(f"  {n_branches}个分支: f_D/W = {ratio['result']:.6e} m/N")
        print(f"    FUNCTION_CALL: calculate_deflection_to_weight_ratio | PARAMS: {{num_branches: {n_branches}, ...}} | RESULT: {ratio}")
    print()
    
    # 步骤2：绘制对比图
    print("步骤2：绘制分支数量对比图")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(branch_numbers, results_by_branches, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('Number of Branches', fontsize=12)
    ax.set_ylabel('f_D/W (m/N)', fontsize=12)
    ax.set_title('Effect of Number of Branches on Deflection Ratio', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(branch_numbers)
    
    for i, (n, r) in enumerate(zip(branch_numbers, results_by_branches)):
        ax.annotate(f'{r:.2e}', xy=(n, r), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    filepath_branches = './tool_images/branch_number_comparison.png'
    plt.savefig(filepath_branches, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {filepath_branches}")
    print()
    
    # 步骤3：分析分支数量的影响机制
    print("步骤3：分析分支数量的影响机制")
    print("对于n个分支的系统：")
    print("  - 节点B处总竖向力 = n × W")
    print("  - 节点B处总弯矩 = n × W × l")
    print("  - 主梁挠度随分支数量线性增加")
    print("  - 分支自身挠度不受分支数量影响")
    print()
    
    # 计算各分量的贡献
    for n_branches in [1, 2, 4]:
        deflection_detail = calculate_branched_cantilever_deflection(
            weight=W,
            length_ab=l,
            length_branch=l,
            ei_ab=2*EI,
            ei_branch=EI,
            num_branches=n_branches
        )
        print(f"{n_branches}个分支时的挠度分量：")
        print(f"  分支自身挠度: {deflection_detail['metadata']['deflection_branch']:.6e} m")
        print(f"  B点挠度: {deflection_detail['metadata']['deflection_at_b']:.6e} m")
        print(f"  附加挠度: {deflection_detail['metadata']['additional_deflection']:.6e} m")
        print(f"  总挠度: {deflection_detail['result']:.6e} m")
        print(f"    FUNCTION_CALL: calculate_branched_cantilever_deflection | PARAMS: {{num_branches: {n_branches}, ...}} | RESULT: {deflection_detail}")
        print()
    
    print("-" * 80)
    print("场景3结论：")
    print("1. 挠度比值随分支数量线性增加")
    print("2. 双分支系统（原问题）的挠度是单分支的约2倍")
    print("3. 增加分支数量可以提高能量收集装置的灵敏度")
    print("4. 但过多分支会降低结构刚度，需要权衡设计")
    print(f"FINAL_ANSWER: 分支数量分析完成，双分支系统 f_D/W = {results_by_branches[1]:.6e} m/N")
    print()
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("=" * 80)
    print("工具包演示总结")
    print("=" * 80)
    print("本工具包成功实现了：")
    print("1. 悬臂梁基础力学计算（挠度、转角、弯矩）")
    print("2. 分支悬臂梁系统的复杂分析")
    print("3. 刚性节点力学分析")
    print("4. 参数敏感性分析")
    print("5. 多场景可视化")
    print()
    print("核心结论：")
    print(f"能量收集装置的挠度与重量比值：f_D/W = 8l³/(3EI)")
    print(f"数值验证：f_D/W = {ratio_result['result']:.6e} m/N (l=1m, EI=1N·m²)")
    print()
    print("生成的文件：")
    print(f"  1. 结构示意图: {structure_result['result']}")
    print(f"  2. 挠度曲线: {deflection_curve['result']}")
    print(f"  3. 敏感性分析: {sensitivity_result['result']}")
    print(f"  4. 分支数量对比: {filepath_branches}")
    print("=" * 80)


if __name__ == "__main__":
    main()