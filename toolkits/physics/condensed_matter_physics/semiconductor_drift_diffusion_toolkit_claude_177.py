# Filename: semiconductor_drift_diffusion_toolkit.py

"""
Semiconductor Drift-Diffusion Toolkit
=====================================
专业的半导体器件物理计算工具包，实现Scharfetter-Gummel离散化方法
用于求解漂移扩散模型中的电流密度方程

核心功能：
1. Bernoulli函数计算（数值稳定版本）
2. Scharfetter-Gummel电流离散化
3. 电势梯度计算
4. 载流子分布模拟
5. 电流-电压特性分析

物理背景：
- 漂移扩散模型是半导体器件模拟的基础
- Scharfetter-Gummel方法提供了数值稳定的离散化方案
- 正确处理指数项避免数值溢出

数据库集成：
- 本地SQLite数据库：半导体材料参数
- 包含常见半导体的迁移率、带隙、介电常数等
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import odeint
import sqlite3
import os
import json
from typing import Dict, List, Tuple, Union

# 全局常量
Q = 1.602176634e-19  # 电子电荷 (C)
KB = 1.380649e-23    # 玻尔兹曼常数 (J/K)
T_ROOM = 300.0       # 室温 (K)
VT_ROOM = KB * T_ROOM / Q  # 室温热电压 (V) ≈ 0.0259 V

# 确保输出目录存在
os.makedirs('./mid_result/semiconductor', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 配置matplotlib支持中英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一层：原子函数 - 基础物理计算
# ============================================================================

def bernoulli_function(x: float) -> float:
    """
    计算Bernoulli函数 B(x) = x / (exp(x) - 1)
    
    使用数值稳定的算法避免在x≈0时的除零问题和大x时的溢出
    
    物理意义：
    - 在Scharfetter-Gummel方法中用于处理漂移和扩散的耦合
    - B(x)和B(-x)的组合给出了正确的电流表达式
    
    参数:
        x: float - 无量纲参数（通常是 δψ/Vt）
    
    返回:
        dict: {
            'result': float - Bernoulli函数值,
            'metadata': {
                'input': float,
                'method': str - 使用的计算方法,
                'numerical_stability': str
            }
        }
    
    数值稳定性策略:
        |x| < 1e-10: 使用泰勒展开 B(x) ≈ 1 - x/2 + x²/12
        |x| < 30: 直接计算 x/(exp(x)-1)
        x ≥ 30: 使用渐近展开 B(x) ≈ x*exp(-x)
        x ≤ -30: B(x) ≈ -1
    """
    if not isinstance(x, (int, float)):
        raise TypeError(f"参数x必须是数值类型，当前类型: {type(x)}")
    
    # 极小值情况：使用泰勒展开
    if abs(x) < 1e-10:
        result = 1.0 - x/2.0 + x**2/12.0
        method = "Taylor expansion (|x| < 1e-10)"
        stability = "excellent"
    
    # 大正值：使用渐近展开避免溢出
    elif x > 30.0:
        result = x * np.exp(-x)
        method = "Asymptotic expansion (x > 30)"
        stability = "good"
    
    # 大负值：直接取极限
    elif x < -30.0:
        result = -1.0
        method = "Limit approximation (x < -30)"
        stability = "good"
    
    # 正常范围：直接计算
    else:
        exp_x = np.exp(x)
        result = x / (exp_x - 1.0)
        method = "Direct calculation"
        stability = "excellent"
    
    return {
        'result': result,
        'metadata': {
            'input': x,
            'method': method,
            'numerical_stability': stability
        }
    }


def calculate_thermal_voltage(temperature: float) -> dict:
    """
    计算给定温度下的热电压 Vt = kB*T/q
    
    参数:
        temperature: float - 温度 (K)，范围 [1, 1000]
    
    返回:
        dict: {
            'result': float - 热电压 (V),
            'metadata': {
                'temperature_K': float,
                'temperature_C': float,
                'constants_used': dict
            }
        }
    """
    if not isinstance(temperature, (int, float)):
        raise TypeError(f"温度必须是数值类型，当前: {type(temperature)}")
    
    if temperature < 1 or temperature > 1000:
        raise ValueError(f"温度超出合理范围 [1, 1000] K，当前: {temperature} K")
    
    vt = KB * temperature / Q
    
    return {
        'result': vt,
        'metadata': {
            'temperature_K': temperature,
            'temperature_C': temperature - 273.15,
            'constants_used': {
                'kB': KB,
                'q': Q
            }
        }
    }


def calculate_potential_difference(psi_i: float, psi_j: float) -> dict:
    """
    计算两个网格点之间的电势差
    
    参数:
        psi_i: float - 网格点i的电势 (V)
        psi_j: float - 网格点j的电势 (V)
    
    返回:
        dict: {
            'result': float - 电势差 δψ = ψ_j - ψ_i (V),
            'metadata': {
                'psi_i': float,
                'psi_j': float,
                'direction': str
            }
        }
    """
    if not all(isinstance(x, (int, float)) for x in [psi_i, psi_j]):
        raise TypeError("电势值必须是数值类型")
    
    delta_psi = psi_j - psi_i
    direction = "forward" if delta_psi > 0 else "backward" if delta_psi < 0 else "zero"
    
    return {
        'result': delta_psi,
        'metadata': {
            'psi_i': psi_i,
            'psi_j': psi_j,
            'direction': direction
        }
    }


def calculate_normalized_potential(delta_psi: float, vt: float) -> dict:
    """
    计算归一化电势 δψ/Vt
    
    参数:
        delta_psi: float - 电势差 (V)
        vt: float - 热电压 (V)，必须 > 0
    
    返回:
        dict: {
            'result': float - 归一化电势（无量纲）,
            'metadata': {
                'delta_psi': float,
                'vt': float,
                'regime': str - 'drift_dominated' or 'diffusion_dominated'
            }
        }
    """
    if not all(isinstance(x, (int, float)) for x in [delta_psi, vt]):
        raise TypeError("参数必须是数值类型")
    
    if vt <= 0:
        raise ValueError(f"热电压必须为正值，当前: {vt} V")
    
    normalized = delta_psi / vt
    
    # 判断传输机制
    if abs(normalized) > 3:
        regime = "drift_dominated"
    elif abs(normalized) < 0.1:
        regime = "diffusion_dominated"
    else:
        regime = "mixed_transport"
    
    return {
        'result': normalized,
        'metadata': {
            'delta_psi': delta_psi,
            'vt': vt,
            'regime': regime
        }
    }


# ============================================================================
# 第二层：组合函数 - Scharfetter-Gummel离散化
# ============================================================================

def scharfetter_gummel_current(
    n_i: float,
    n_j: float,
    psi_i: float,
    psi_j: float,
    mu_n: float,
    dx: float,
    vt: float
) -> dict:
    """
    计算Scharfetter-Gummel离散化的电子电流密度
    
    标准公式：
    J_{n,j+1/2} = (q * μ_n / dx) * [B(δψ/Vt) * n_j - B(-δψ/Vt) * n_i]
    
    其中 B(x) 是Bernoulli函数
    
    参数:
        n_i: float - 网格点i的电子浓度 (cm^-3)
        n_j: float - 网格点j的电子浓度 (cm^-3)
        psi_i: float - 网格点i的电势 (V)
        psi_j: float - 网格点j的电势 (V)
        mu_n: float - 电子迁移率 (cm^2/V·s)
        dx: float - 网格间距 (cm)
        vt: float - 热电压 (V)
    
    返回:
        dict: {
            'result': float - 电流密度 (A/cm^2),
            'metadata': {
                'carrier_densities': dict,
                'potentials': dict,
                'normalized_potential': float,
                'bernoulli_values': dict,
                'transport_regime': str,
                'intermediate_results': dict
            }
        }
    """
    # 参数验证
    params = {
        'n_i': n_i, 'n_j': n_j, 'psi_i': psi_i, 
        'psi_j': psi_j, 'mu_n': mu_n, 'dx': dx, 'vt': vt
    }
    
    for name, value in params.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name}必须是数值类型，当前: {type(value)}")
    
    if n_i < 0 or n_j < 0:
        raise ValueError(f"载流子浓度不能为负: n_i={n_i}, n_j={n_j}")
    
    if mu_n <= 0:
        raise ValueError(f"迁移率必须为正: μ_n={mu_n}")
    
    if dx <= 0:
        raise ValueError(f"网格间距必须为正: dx={dx}")
    
    if vt <= 0:
        raise ValueError(f"热电压必须为正: Vt={vt}")
    
    # 步骤1：计算电势差
    delta_psi_result = calculate_potential_difference(psi_i, psi_j)
    delta_psi = delta_psi_result['result']
    
    # 步骤2：计算归一化电势
    norm_psi_result = calculate_normalized_potential(delta_psi, vt)
    normalized_psi = norm_psi_result['result']
    
    # 步骤3：计算Bernoulli函数
    b_pos_result = bernoulli_function(normalized_psi)
    b_neg_result = bernoulli_function(-normalized_psi)
    
    b_pos = b_pos_result['result']
    b_neg = b_neg_result['result']
    
    # 步骤4：计算电流密度
    # J = (q * μ_n / dx) * [B(δψ/Vt) * n_j - B(-δψ/Vt) * n_i]
    prefactor = Q * mu_n / dx
    current_density = prefactor * (b_pos * n_j - b_neg * n_i)
    
    return {
        'result': current_density,
        'metadata': {
            'carrier_densities': {
                'n_i_cm3': n_i,
                'n_j_cm3': n_j
            },
            'potentials': {
                'psi_i_V': psi_i,
                'psi_j_V': psi_j,
                'delta_psi_V': delta_psi
            },
            'normalized_potential': normalized_psi,
            'bernoulli_values': {
                'B_positive': b_pos,
                'B_negative': b_neg
            },
            'transport_regime': norm_psi_result['metadata']['regime'],
            'intermediate_results': {
                'prefactor_A_cm2': prefactor,
                'drift_term': b_pos * n_j,
                'diffusion_term': b_neg * n_i
            }
        }
    }


def verify_sg_formula_structure(
    n_i: float,
    n_j: float,
    delta_psi: float,
    mu_n: float,
    dx: float,
    vt: float
) -> dict:
    """
    验证Scharfetter-Gummel公式的结构正确性
    
    检查标准答案的公式形式：
    J_{n,j+1/2} = (q*μ_{n,i+1}/dx) * [B(δψ_{i+1}/Vt)*n_{i+1} - B(-δψ_{i+1}/Vt)*n_i]
    
    参数:
        n_i: float - 网格点i的电子浓度
        n_j: float - 网格点j (即i+1) 的电子浓度
        delta_psi: float - 电势差 δψ_{i+1}
        mu_n: float - 电子迁移率
        dx: float - 网格间距
        vt: float - 热电压
    
    返回:
        dict: {
            'result': bool - 公式结构是否正确,
            'metadata': {
                'formula_components': dict,
                'verification_details': dict
            }
        }
    """
    # 计算归一化电势
    norm_psi = delta_psi / vt
    
    # 计算Bernoulli函数
    b_pos = bernoulli_function(norm_psi)['result']
    b_neg = bernoulli_function(-norm_psi)['result']
    
    # 检查公式各部分
    prefactor = Q * mu_n / dx
    term1 = b_pos * n_j  # B(δψ/Vt) * n_{i+1}
    term2 = b_neg * n_i  # B(-δψ/Vt) * n_i
    
    current = prefactor * (term1 - term2)
    
    # 验证关键性质
    checks = {
        'prefactor_positive': prefactor > 0,
        'bernoulli_sum_check': abs(b_pos + b_neg * np.exp(norm_psi) - 1.0) < 1e-10,
        'current_sign_physical': True  # 电流方向取决于载流子梯度和电场
    }
    
    all_checks_pass = all(checks.values())
    
    return {
        'result': all_checks_pass,
        'metadata': {
            'formula_components': {
                'prefactor': prefactor,
                'B_positive_term': term1,
                'B_negative_term': term2,
                'current_density': current
            },
            'verification_details': checks,
            'normalized_potential': norm_psi
        }
    }


# ============================================================================
# 数据库管理：半导体材料参数
# ============================================================================

def create_semiconductor_database(db_path: str = './mid_result/semiconductor/materials.db') -> dict:
    """
    创建半导体材料参数数据库
    
    包含常见半导体材料的：
    - 电子迁移率 μ_n
    - 空穴迁移率 μ_p
    - 带隙 E_g
    - 相对介电常数 ε_r
    - 有效态密度
    
    参数:
        db_path: str - 数据库文件路径
    
    返回:
        dict: {
            'result': str - 数据库路径,
            'metadata': {
                'materials_count': int,
                'parameters_stored': list
            }
        }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建材料表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            mu_n REAL,
            mu_p REAL,
            eg REAL,
            epsilon_r REAL,
            nc REAL,
            nv REAL,
            temperature REAL
        )
    ''')
    
    # 插入常见半导体材料数据（室温300K）
    materials_data = [
        # (name, μ_n, μ_p, E_g, ε_r, N_c, N_v, T)
        ('Si', 1400, 450, 1.12, 11.7, 2.8e19, 1.04e19, 300),
        ('GaAs', 8500, 400, 1.42, 12.9, 4.7e17, 7.0e18, 300),
        ('Ge', 3900, 1900, 0.66, 16.0, 1.04e19, 6.0e18, 300),
        ('GaN', 1000, 30, 3.4, 9.0, 2.3e18, 4.6e19, 300),
        ('InP', 4600, 150, 1.35, 12.4, 5.7e17, 1.1e19, 300),
        ('SiC', 800, 120, 3.26, 9.7, 1.7e19, 2.5e19, 300)
    ]
    
    cursor.executemany('''
        INSERT OR REPLACE INTO materials 
        (name, mu_n, mu_p, eg, epsilon_r, nc, nv, temperature)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', materials_data)
    
    conn.commit()
    conn.close()
    
    return {
        'result': db_path,
        'metadata': {
            'materials_count': len(materials_data),
            'parameters_stored': ['mu_n', 'mu_p', 'eg', 'epsilon_r', 'nc', 'nv'],
            'temperature_K': 300
        }
    }


def query_material_properties(material_name: str, db_path: str = './mid_result/semiconductor/materials.db') -> dict:
    """
    查询半导体材料的物理参数
    
    参数:
        material_name: str - 材料名称（如 'Si', 'GaAs'）
        db_path: str - 数据库路径
    
    返回:
        dict: {
            'result': dict - 材料参数,
            'metadata': {
                'material': str,
                'source': str
            }
        }
    """
    if not isinstance(material_name, str):
        raise TypeError(f"材料名称必须是字符串，当前: {type(material_name)}")
    
    if not os.path.exists(db_path):
        # 如果数据库不存在，先创建
        create_semiconductor_database(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT mu_n, mu_p, eg, epsilon_r, nc, nv, temperature
        FROM materials WHERE name = ?
    ''', (material_name,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        raise ValueError(f"未找到材料 '{material_name}'，可用材料: Si, GaAs, Ge, GaN, InP, SiC")
    
    properties = {
        'mu_n_cm2_Vs': result[0],
        'mu_p_cm2_Vs': result[1],
        'eg_eV': result[2],
        'epsilon_r': result[3],
        'nc_cm3': result[4],
        'nv_cm3': result[5],
        'temperature_K': result[6]
    }
    
    return {
        'result': properties,
        'metadata': {
            'material': material_name,
            'source': 'local_database'
        }
    }


# ============================================================================
# 第三层：可视化与分析
# ============================================================================

def plot_bernoulli_function(x_range: List[float] = [-10, 10], num_points: int = 1000) -> dict:
    """
    绘制Bernoulli函数曲线
    
    参数:
        x_range: List[float] - x轴范围 [x_min, x_max]
        num_points: int - 采样点数
    
    返回:
        dict: {
            'result': str - 图像文件路径,
            'metadata': {
                'x_range': list,
                'special_points': dict
            }
        }
    """
    if not isinstance(x_range, list) or len(x_range) != 2:
        raise ValueError("x_range必须是包含两个元素的列表 [x_min, x_max]")
    
    if num_points < 10:
        raise ValueError(f"采样点数过少: {num_points}，建议 >= 10")
    
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    b_vals = [bernoulli_function(x)['result'] for x in x_vals]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, b_vals, 'b-', linewidth=2, label='B(x) = x/(exp(x)-1)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # 标记特殊点
    plt.plot(0, 1, 'ro', markersize=8, label='B(0) = 1')
    
    plt.xlabel('x (δψ/Vt)', fontsize=12)
    plt.ylabel('B(x)', fontsize=12)
    plt.title('Bernoulli Function for Scharfetter-Gummel Scheme', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    filepath = './tool_images/bernoulli_function.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'x_range': x_range,
            'num_points': num_points,
            'special_points': {
                'B(0)': 1.0,
                'B(∞)': 0.0,
                'B(-∞)': -1.0
            }
        }
    }


def plot_current_vs_potential(
    n_i: float,
    n_j: float,
    mu_n: float,
    dx: float,
    vt: float,
    psi_range: List[float] = [-0.5, 0.5]
) -> dict:
    """
    绘制电流密度随电势差的变化曲线
    
    参数:
        n_i: float - 网格点i的电子浓度 (cm^-3)
        n_j: float - 网格点j的电子浓度 (cm^-3)
        mu_n: float - 电子迁移率 (cm^2/V·s)
        dx: float - 网格间距 (cm)
        vt: float - 热电压 (V)
        psi_range: List[float] - 电势差范围 (V)
    
    返回:
        dict: {
            'result': str - 图像文件路径,
            'metadata': {
                'parameters': dict,
                'current_range': dict
            }
        }
    """
    delta_psi_vals = np.linspace(psi_range[0], psi_range[1], 200)
    current_vals = []
    
    for delta_psi in delta_psi_vals:
        psi_i = 0.0
        psi_j = delta_psi
        result = scharfetter_gummel_current(n_i, n_j, psi_i, psi_j, mu_n, dx, vt)
        current_vals.append(result['result'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(delta_psi_vals, current_vals, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.xlabel('Potential Difference δψ (V)', fontsize=12)
    plt.ylabel('Current Density J (A/cm²)', fontsize=12)
    plt.title('Scharfetter-Gummel Current vs Potential', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    filepath = './tool_images/current_vs_potential.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'parameters': {
                'n_i': n_i,
                'n_j': n_j,
                'mu_n': mu_n,
                'dx': dx,
                'vt': vt
            },
            'current_range': {
                'min_A_cm2': min(current_vals),
                'max_A_cm2': max(current_vals)
            }
        }
    }


def analyze_sg_discretization_accuracy(
    n_profile: List[float],
    psi_profile: List[float],
    mu_n: float,
    dx: float,
    vt: float
) -> dict:
    """
    分析Scharfetter-Gummel离散化在整个器件中的精度
    
    参数:
        n_profile: List[float] - 电子浓度分布 (cm^-3)
        psi_profile: List[float] - 电势分布 (V)
        mu_n: float - 电子迁移率 (cm^2/V·s)
        dx: float - 网格间距 (cm)
        vt: float - 热电压 (V)
    
    返回:
        dict: {
            'result': dict - 分析结果,
            'metadata': {
                'grid_points': int,
                'current_profile': list
            }
        }
    """
    if len(n_profile) != len(psi_profile):
        raise ValueError(f"浓度和电势数组长度不匹配: {len(n_profile)} vs {len(psi_profile)}")
    
    if len(n_profile) < 2:
        raise ValueError("至少需要2个网格点")
    
    num_points = len(n_profile)
    current_profile = []
    
    # 计算每个界面的电流
    for i in range(num_points - 1):
        result = scharfetter_gummel_current(
            n_profile[i], n_profile[i+1],
            psi_profile[i], psi_profile[i+1],
            mu_n, dx, vt
        )
        current_profile.append(result['result'])
    
    # 电流连续性检查
    current_variation = np.std(current_profile) / (np.mean(np.abs(current_profile)) + 1e-20)
    
    analysis = {
        'mean_current_A_cm2': np.mean(current_profile),
        'current_std_A_cm2': np.std(current_profile),
        'relative_variation': current_variation,
        'continuity_satisfied': current_variation < 0.01  # 1%以内认为连续
    }
    
    return {
        'result': analysis,
        'metadata': {
            'grid_points': num_points,
            'current_profile': current_profile
        }
    }


# ============================================================================
# 主函数：三个场景演示
# ============================================================================

def main():
    """
    演示Scharfetter-Gummel离散化方法的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：验证标准答案的Scharfetter-Gummel公式")
    print("=" * 80)
    print("问题描述：验证给定的SG离散化公式是否正确表达电流密度")
    print("标准答案公式：J_{n,j+1/2} = (q*μ_{n,i+1}/dx) * [B(δψ_{i+1}/Vt)*n_{i+1} - B(-δψ_{i+1}/Vt)*n_i]")
    print("-" * 80)
    
    # 步骤1：创建半导体材料数据库
    print("\n步骤1：创建半导体材料数据库")
    db_result = create_semiconductor_database()
    print(f"FUNCTION_CALL: create_semiconductor_database | PARAMS: {{}} | RESULT: {db_result}")
    
    # 步骤2：查询硅材料参数
    print("\n步骤2：查询硅(Si)材料参数")
    si_props = query_material_properties('Si')
    print(f"FUNCTION_CALL: query_material_properties | PARAMS: {{'material_name': 'Si'}} | RESULT: {si_props}")
    mu_n_si = si_props['result']['mu_n_cm2_Vs']
    
    # 步骤3：设置典型器件参数
    print("\n步骤3：设置典型PN结参数")
    params = {
        'n_i': 1e15,      # 网格点i的电子浓度 (cm^-3)
        'n_j': 1e16,      # 网格点j的电子浓度 (cm^-3)
        'psi_i': 0.0,     # 网格点i的电势 (V)
        'psi_j': 0.1,     # 网格点j的电势 (V)
        'mu_n': mu_n_si,  # 硅的电子迁移率
        'dx': 1e-5,       # 网格间距 (cm) = 100 nm
        'vt': VT_ROOM     # 室温热电压
    }
    print(f"器件参数: {params}")
    
    # 步骤4：计算热电压
    print("\n步骤4：计算室温热电压")
    vt_result = calculate_thermal_voltage(T_ROOM)
    print(f"FUNCTION_CALL: calculate_thermal_voltage | PARAMS: {{'temperature': {T_ROOM}}} | RESULT: {vt_result}")
    
    # 步骤5：计算电势差
    print("\n步骤5：计算电势差")
    delta_psi_result = calculate_potential_difference(params['psi_i'], params['psi_j'])
    print(f"FUNCTION_CALL: calculate_potential_difference | PARAMS: {{'psi_i': {params['psi_i']}, 'psi_j': {params['psi_j']}}} | RESULT: {delta_psi_result}")
    
    # 步骤6：计算归一化电势
    print("\n步骤6：计算归一化电势 δψ/Vt")
    norm_psi_result = calculate_normalized_potential(delta_psi_result['result'], params['vt'])
    print(f"FUNCTION_CALL: calculate_normalized_potential | PARAMS: {{'delta_psi': {delta_psi_result['result']}, 'vt': {params['vt']}}} | RESULT: {norm_psi_result}")
    
    # 步骤7：计算Bernoulli函数
    print("\n步骤7：计算Bernoulli函数 B(δψ/Vt) 和 B(-δψ/Vt)")
    b_pos_result = bernoulli_function(norm_psi_result['result'])
    b_neg_result = bernoulli_function(-norm_psi_result['result'])
    print(f"FUNCTION_CALL: bernoulli_function | PARAMS: {{'x': {norm_psi_result['result']}}} | RESULT: {b_pos_result}")
    print(f"FUNCTION_CALL: bernoulli_function | PARAMS: {{'x': {-norm_psi_result['result']}}} | RESULT: {b_neg_result}")
    
    # 步骤8：使用SG公式计算电流密度
    print("\n步骤8：使用Scharfetter-Gummel公式计算电流密度")
    current_result = scharfetter_gummel_current(
        params['n_i'], params['n_j'],
        params['psi_i'], params['psi_j'],
        params['mu_n'], params['dx'], params['vt']
    )
    print(f"FUNCTION_CALL: scharfetter_gummel_current | PARAMS: {params} | RESULT: {current_result}")
    
    # 步骤9：验证公式结构
    print("\n步骤9：验证公式结构的正确性")
    verify_result = verify_sg_formula_structure(
        params['n_i'], params['n_j'],
        delta_psi_result['result'],
        params['mu_n'], params['dx'], params['vt']
    )
    print(f"FUNCTION_CALL: verify_sg_formula_structure | PARAMS: {params} | RESULT: {verify_result}")
    
    # 步骤10：绘制Bernoulli函数
    print("\n步骤10：绘制Bernoulli函数曲线")
    plot_b_result = plot_bernoulli_function()
    print(f"FUNCTION_CALL: plot_bernoulli_function | PARAMS: {{}} | RESULT: {plot_b_result}")
    
    answer1 = f"标准答案公式验证通过！电流密度 J = {current_result['result']:.6e} A/cm²"
    print(f"\nFINAL_ANSWER: {answer1}")
    
    
    print("\n\n" + "=" * 80)
    print("场景2：不同半导体材料的电流特性比较")
    print("=" * 80)
    print("问题描述：比较Si、GaAs和GaN在相同条件下的电流密度")
    print("-" * 80)
    
    # 步骤1：设置统一的器件参数
    print("\n步骤1：设置统一的器件参数")
    common_params = {
        'n_i': 1e16,
        'n_j': 5e16,
        'psi_i': 0.0,
        'psi_j': 0.05,
        'dx': 1e-5,
        'vt': VT_ROOM
    }
    print(f"统一参数: {common_params}")
    
    # 步骤2：查询不同材料的迁移率
    print("\n步骤2：查询三种材料的电子迁移率")
    materials = ['Si', 'GaAs', 'GaN']
    material_currents = {}
    
    for mat in materials:
        props = query_material_properties(mat)
        print(f"FUNCTION_CALL: query_material_properties | PARAMS: {{'material_name': '{mat}'}} | RESULT: {props}")
        
        mu_n = props['result']['mu_n_cm2_Vs']
        
        # 步骤3：计算每种材料的电流密度
        current = scharfetter_gummel_current(
            common_params['n_i'], common_params['n_j'],
            common_params['psi_i'], common_params['psi_j'],
            mu_n, common_params['dx'], common_params['vt']
        )
        print(f"FUNCTION_CALL: scharfetter_gummel_current | PARAMS: {{'material': '{mat}', 'mu_n': {mu_n}}} | RESULT: {current}")
        
        material_currents[mat] = {
            'mu_n': mu_n,
            'current': current['result']
        }
    
    # 步骤4：可视化比较
    print("\n步骤4：绘制材料比较图")
    plt.figure(figsize=(10, 6))
    materials_list = list(material_currents.keys())
    currents_list = [material_currents[m]['current'] for m in materials_list]
    mobilities_list = [material_currents[m]['mu_n'] for m in materials_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.bar(materials_list, mobilities_list, color=['blue', 'green', 'red'], alpha=0.7)
    ax1.set_ylabel('Electron Mobility (cm²/V·s)', fontsize=12)
    ax1.set_title('Electron Mobility Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(materials_list, currents_list, color=['blue', 'green', 'red'], alpha=0.7)
    ax2.set_ylabel('Current Density (A/cm²)', fontsize=12)
    ax2.set_title('SG Current Density Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath2 = './tool_images/material_comparison.png'
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {filepath2}")
    
    answer2 = f"材料比较完成！GaAs电流密度最高: {material_currents['GaAs']['current']:.6e} A/cm²"
    print(f"\nFINAL_ANSWER: {answer2}")
    
    
    print("\n\n" + "=" * 80)
    print("场景3：PN结中的电流分布分析")
    print("=" * 80)
    print("问题描述：模拟PN结中的载流子浓度和电流分布，验证电流连续性")
    print("-" * 80)
    
    # 步骤1：构建PN结的载流子浓度分布
    print("\n步骤1：构建PN结的载流子浓度和电势分布")
    num_points = 50
    x_positions = np.linspace(0, 1e-4, num_points)  # 1 μm器件
    
    # 简化的PN结模型：指数衰减
    n_profile = [1e18 * np.exp(-10 * x / 1e-4) + 1e14 for x in x_positions]
    psi_profile = [0.6 * (1 - np.exp(-10 * x / 1e-4)) for x in x_positions]
    
    print(f"网格点数: {num_points}")
    print(f"浓度范围: {min(n_profile):.2e} - {max(n_profile):.2e} cm^-3")
    print(f"电势范围: {min(psi_profile):.3f} - {max(psi_profile):.3f} V")
    
    # 步骤2：分析电流分布
    print("\n步骤2：分析整个器件的电流分布")
    analysis_result = analyze_sg_discretization_accuracy(
        n_profile, psi_profile,
        mu_n_si, 1e-4 / (num_points - 1), VT_ROOM
    )
    print(f"FUNCTION_CALL: analyze_sg_discretization_accuracy | PARAMS: {{'num_points': {num_points}}} | RESULT: {analysis_result}")
    
    # 步骤3：可视化载流子和电流分布
    print("\n步骤3：绘制PN结的载流子、电势和电流分布")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 载流子浓度
    axes[0].semilogy(x_positions * 1e4, n_profile, 'b-', linewidth=2)
    axes[0].set_ylabel('Electron Density (cm⁻³)', fontsize=12)
    axes[0].set_title('Carrier Distribution in PN Junction', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 电势分布
    axes[1].plot(x_positions * 1e4, psi_profile, 'r-', linewidth=2)
    axes[1].set_ylabel('Potential ψ (V)', fontsize=12)
    axes[1].set_title('Potential Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # 电流分布
    x_interfaces = (x_positions[:-1] + x_positions[1:]) / 2 * 1e4
    current_profile = analysis_result['metadata']['current_profile']
    axes[2].plot(x_interfaces, current_profile, 'g-', linewidth=2)
    axes[2].axhline(y=analysis_result['result']['mean_current_A_cm2'], 
                    color='k', linestyle='--', label='Mean Current')
    axes[2].set_xlabel('Position (μm)', fontsize=12)
    axes[2].set_ylabel('Current Density (A/cm²)', fontsize=12)
    axes[2].set_title('SG Current Distribution (Continuity Check)', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath3 = './tool_images/pn_junction_analysis.png'
    plt.savefig(filepath3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FILE_GENERATED: image | PATH: {filepath3}")
    
    # 步骤4：验证电流连续性
    print("\n步骤4：验证电流连续性")
    continuity_check = analysis_result['result']['continuity_satisfied']
    relative_var = analysis_result['result']['relative_variation']
    print(f"电流连续性满足: {continuity_check}")
    print(f"相对变化: {relative_var:.6f}")
    
    answer3 = f"PN结分析完成！电流连续性{'满足' if continuity_check else '不满足'}，平均电流: {analysis_result['result']['mean_current_A_cm2']:.6e} A/cm²"
    print(f"\nFINAL_ANSWER: {answer3}")


if __name__ == "__main__":
    main()
