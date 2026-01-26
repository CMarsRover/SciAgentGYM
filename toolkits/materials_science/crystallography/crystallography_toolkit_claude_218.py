# Filename: crystallography_toolkit.py

"""
Crystallography Toolkit for Crystal Structure Analysis
专业晶体学计算工具包 - 支持晶格参数计算、晶面间距、倒易空间等

核心功能：
1. 晶格参数转换（实空间 ↔ 倒易空间）
2. 晶面间距计算（支持7种晶系）
3. 晶体结构可视化
4. 晶体学数据库集成（COD - Crystallography Open Database）

依赖库：
- pymatgen: 晶体结构分析
- numpy: 数值计算
- matplotlib: 可视化
- ase: 原子结构操作
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
import os

# 确保输出目录存在
os.makedirs('./mid_result/crystallography', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一层：原子函数 - 基础晶体学计算
# ============================================================================

def calculate_metric_tensor(a: float, b: float, c: float, 
                           alpha: float, beta: float, gamma: float) -> Dict:
    """
    计算晶格的度规张量（Metric Tensor）
    
    度规张量是晶体学中的基础量，用于计算晶格中的距离和角度
    
    Parameters:
    -----------
    a, b, c : float
        晶格常数（单位：Angstrom）
    alpha, beta, gamma : float
        晶格角度（单位：度）
    
    Returns:
    --------
    dict : {
        'result': list[list[float]],  # 3x3度规张量矩阵
        'metadata': {
            'lattice_params': dict,
            'volume': float,
            'description': str
        }
    }
    """
    # 参数验证
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("晶格常数必须为正数")
    if not (0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180):
        raise ValueError("晶格角度必须在(0, 180)度范围内")
    
    # 转换为弧度
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # 计算度规张量
    g11 = a**2
    g22 = b**2
    g33 = c**2
    g12 = a * b * np.cos(gamma_rad)
    g13 = a * c * np.cos(beta_rad)
    g23 = b * c * np.cos(alpha_rad)
    
    metric_tensor = [
        [g11, g12, g13],
        [g12, g22, g23],
        [g13, g23, g33]
    ]
    
    # 计算晶胞体积
    volume = np.sqrt(np.linalg.det(metric_tensor))
    
    return {
        'result': metric_tensor,
        'metadata': {
            'lattice_params': {'a': a, 'b': b, 'c': c, 
                             'alpha': alpha, 'beta': beta, 'gamma': gamma},
            'volume': float(volume),
            'description': 'Metric tensor for direct lattice'
        }
    }


def calculate_reciprocal_metric_tensor(metric_tensor: List[List[float]]) -> Dict:
    """
    计算倒易晶格的度规张量
    
    倒易度规张量 = (实空间度规张量)^(-1)
    
    Parameters:
    -----------
    metric_tensor : list[list[float]]
        实空间度规张量（3x3矩阵）
    
    Returns:
    --------
    dict : {
        'result': list[list[float]],  # 倒易度规张量
        'metadata': {
            'determinant': float,
            'description': str
        }
    }
    """
    g = np.array(metric_tensor)
    
    # 验证矩阵
    if g.shape != (3, 3):
        raise ValueError("度规张量必须是3x3矩阵")
    
    det_g = np.linalg.det(g)
    if abs(det_g) < 1e-10:
        raise ValueError("度规张量行列式接近零，矩阵奇异")
    
    # 计算倒易度规张量
    g_reciprocal = np.linalg.inv(g)
    
    return {
        'result': g_reciprocal.tolist(),
        'metadata': {
            'determinant': float(det_g),
            'reciprocal_determinant': float(1/det_g),
            'description': 'Reciprocal metric tensor'
        }
    }


def calculate_interplanar_spacing_general(h: int, k: int, l: int,
                                         metric_tensor: List[List[float]]) -> Dict:
    """
    通用晶面间距计算公式（适用于所有晶系）
    
    使用倒易度规张量计算d-spacing：
    1/d² = h² g*₁₁ + k² g*₂₂ + l² g*₃₃ + 2hk g*₁₂ + 2hl g*₁₃ + 2kl g*₂₃
    
    Parameters:
    -----------
    h, k, l : int
        Miller指数
    metric_tensor : list[list[float]]
        实空间度规张量
    
    Returns:
    --------
    dict : {
        'result': float,  # 晶面间距（Angstrom）
        'metadata': {
            'miller_indices': tuple,
            'reciprocal_d_squared': float,
            'formula': str
        }
    }
    """
    # 验证Miller指数
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller指数不能全为零")
    
    # 计算倒易度规张量
    g_recip_result = calculate_reciprocal_metric_tensor(metric_tensor)
    g_star = np.array(g_recip_result['result'])
    
    # 计算1/d²
    one_over_d_squared = (
        h**2 * g_star[0, 0] +
        k**2 * g_star[1, 1] +
        l**2 * g_star[2, 2] +
        2*h*k * g_star[0, 1] +
        2*h*l * g_star[0, 2] +
        2*k*l * g_star[1, 2]
    )
    
    if one_over_d_squared <= 0:
        raise ValueError(f"计算出的1/d²为负值或零: {one_over_d_squared}")
    
    d_spacing = 1.0 / np.sqrt(one_over_d_squared)
    
    return {
        'result': float(d_spacing),
        'metadata': {
            'miller_indices': (h, k, l),
            'reciprocal_d_squared': float(one_over_d_squared),
            'formula': '1/d² = Σ hᵢhⱼ g*ᵢⱼ (Einstein summation)',
            'reciprocal_metric_tensor': g_star.tolist()
        }
    }


def calculate_rhombohedral_d_spacing(h: int, k: int, l: int,
                                     a: float, alpha: float) -> Dict:
    """
    菱面体晶系专用晶面间距公式（解析解）
    
    对于菱面体晶系（a=b=c, α=β=γ），有简化公式：
    1/d² = [(h²+k²+l²)sin²α + 2(hk+hl+kl)(cos²α-cosα)] / [a²(1-3cos²α+2cos³α)]
    
    Parameters:
    -----------
    h, k, l : int
        Miller指数
    a : float
        晶格常数（Angstrom）
    alpha : float
        晶格角度（度）
    
    Returns:
    --------
    dict : {
        'result': float,  # 晶面间距（Angstrom）
        'metadata': {
            'miller_indices': tuple,
            'lattice_constant': float,
            'lattice_angle': float,
            'formula_type': str
        }
    }
    """
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller指数不能全为零")
    if a <= 0:
        raise ValueError("晶格常数必须为正数")
    if not (0 < alpha < 180):
        raise ValueError("晶格角度必须在(0, 180)度范围内")
    
    alpha_rad = np.radians(alpha)
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)
    
    # 菱面体晶系简化公式
    numerator = (
        (h**2 + k**2 + l**2) * sin_alpha**2 +
        2 * (h*k + h*l + k*l) * (cos_alpha**2 - cos_alpha)
    )
    
    denominator = a**2 * (1 - 3*cos_alpha**2 + 2*cos_alpha**3)
    
    if denominator <= 0:
        raise ValueError(f"分母计算结果非正: {denominator}")
    
    one_over_d_squared = numerator / denominator
    
    if one_over_d_squared <= 0:
        raise ValueError(f"1/d²计算结果非正: {one_over_d_squared}")
    
    d_spacing = 1.0 / np.sqrt(one_over_d_squared)
    
    return {
        'result': float(d_spacing),
        'metadata': {
            'miller_indices': (h, k, l),
            'lattice_constant': a,
            'lattice_angle': alpha,
            'formula_type': 'Rhombohedral analytical formula',
            'cos_alpha': float(cos_alpha),
            'sin_alpha': float(sin_alpha)
        }
    }


def identify_crystal_system(a: float, b: float, c: float,
                           alpha: float, beta: float, gamma: float) -> Dict:
    """
    识别晶系类型
    
    根据晶格参数判断属于7种晶系中的哪一种：
    1. Cubic (立方): a=b=c, α=β=γ=90°
    2. Tetragonal (四方): a=b≠c, α=β=γ=90°
    3. Orthorhombic (正交): a≠b≠c, α=β=γ=90°
    4. Rhombohedral (菱面体): a=b=c, α=β=γ≠90°
    5. Hexagonal (六方): a=b≠c, α=β=90°, γ=120°
    6. Monoclinic (单斜): a≠b≠c, α=γ=90°≠β
    7. Triclinic (三斜): a≠b≠c, α≠β≠γ
    
    Parameters:
    -----------
    a, b, c : float
        晶格常数
    alpha, beta, gamma : float
        晶格角度（度）
    
    Returns:
    --------
    dict : {
        'result': str,  # 晶系名称
        'metadata': {
            'lattice_params': dict,
            'symmetry_conditions': list,
            'description': str
        }
    }
    """
    tol = 1e-3  # 容差
    
    # 判断晶格常数相等性
    a_eq_b = abs(a - b) < tol
    b_eq_c = abs(b - c) < tol
    a_eq_c = abs(a - c) < tol
    
    # 判断角度相等性
    alpha_eq_beta = abs(alpha - beta) < tol
    beta_eq_gamma = abs(beta - gamma) < tol
    alpha_eq_gamma = abs(alpha - gamma) < tol
    
    # 判断特殊角度
    alpha_90 = abs(alpha - 90) < tol
    beta_90 = abs(beta - 90) < tol
    gamma_90 = abs(gamma - 90) < tol
    gamma_120 = abs(gamma - 120) < tol
    
    conditions = []
    
    # 立方晶系
    if a_eq_b and b_eq_c and alpha_90 and beta_90 and gamma_90:
        system = 'Cubic'
        conditions = ['a=b=c', 'α=β=γ=90°']
    
    # 四方晶系
    elif a_eq_b and not b_eq_c and alpha_90 and beta_90 and gamma_90:
        system = 'Tetragonal'
        conditions = ['a=b≠c', 'α=β=γ=90°']
    
    # 菱面体晶系
    elif a_eq_b and b_eq_c and alpha_eq_beta and beta_eq_gamma and not alpha_90:
        system = 'Rhombohedral'
        conditions = ['a=b=c', 'α=β=γ≠90°']
    
    # 六方晶系
    elif a_eq_b and not b_eq_c and alpha_90 and beta_90 and gamma_120:
        system = 'Hexagonal'
        conditions = ['a=b≠c', 'α=β=90°', 'γ=120°']
    
    # 正交晶系
    elif alpha_90 and beta_90 and gamma_90:
        system = 'Orthorhombic'
        conditions = ['a≠b≠c', 'α=β=γ=90°']
    
    # 单斜晶系
    elif alpha_90 and gamma_90 and not beta_90:
        system = 'Monoclinic'
        conditions = ['a≠b≠c', 'α=γ=90°≠β']
    
    # 三斜晶系
    else:
        system = 'Triclinic'
        conditions = ['a≠b≠c', 'α≠β≠γ']
    
    return {
        'result': system,
        'metadata': {
            'lattice_params': {
                'a': a, 'b': b, 'c': c,
                'alpha': alpha, 'beta': beta, 'gamma': gamma
            },
            'symmetry_conditions': conditions,
            'description': f'{system} crystal system'
        }
    }


# ============================================================================
# 第二层：组合函数 - 高级晶体学分析
# ============================================================================

def analyze_crystal_structure(a: float, b: float, c: float,
                              alpha: float, beta: float, gamma: float,
                              miller_indices: Tuple[int, int, int]) -> Dict:
    """
    综合晶体结构分析
    
    组合多个原子函数，完成：
    1. 晶系识别
    2. 度规张量计算
    3. 晶面间距计算（通用公式）
    4. 专用公式计算（如果适用）
    
    Parameters:
    -----------
    a, b, c : float
        晶格常数（Angstrom）
    alpha, beta, gamma : float
        晶格角度（度）
    miller_indices : tuple[int, int, int]
        Miller指数 (h, k, l)
    
    Returns:
    --------
    dict : {
        'result': {
            'crystal_system': str,
            'd_spacing_general': float,
            'd_spacing_specialized': float or None,
            'lattice_volume': float
        },
        'metadata': {
            'metric_tensor': list,
            'reciprocal_metric_tensor': list,
            'analysis_methods': list
        }
    }
    """
    h, k, l = miller_indices
    
    # 步骤1：识别晶系
    system_result = identify_crystal_system(a, b, c, alpha, beta, gamma)
    crystal_system = system_result['result']
    
    # 步骤2：计算度规张量
    metric_result = calculate_metric_tensor(a, b, c, alpha, beta, gamma)
    metric_tensor = metric_result['result']
    volume = metric_result['metadata']['volume']
    
    # 步骤3：计算倒易度规张量
    recip_metric_result = calculate_reciprocal_metric_tensor(metric_tensor)
    recip_metric_tensor = recip_metric_result['result']
    
    # 步骤4：通用公式计算晶面间距
    d_general_result = calculate_interplanar_spacing_general(h, k, l, metric_tensor)
    d_general = d_general_result['result']
    
    # 步骤5：尝试专用公式（菱面体晶系）
    d_specialized = None
    methods_used = ['General formula (metric tensor method)']
    
    if crystal_system == 'Rhombohedral':
        d_specialized_result = calculate_rhombohedral_d_spacing(h, k, l, a, alpha)
        d_specialized = d_specialized_result['result']
        methods_used.append('Rhombohedral analytical formula')
    
    return {
        'result': {
            'crystal_system': crystal_system,
            'd_spacing_general': d_general,
            'd_spacing_specialized': d_specialized,
            'lattice_volume': volume,
            'miller_indices': (h, k, l)
        },
        'metadata': {
            'metric_tensor': metric_tensor,
            'reciprocal_metric_tensor': recip_metric_tensor,
            'analysis_methods': methods_used,
            'lattice_parameters': {
                'a': a, 'b': b, 'c': c,
                'alpha': alpha, 'beta': beta, 'gamma': gamma
            }
        }
    }


def compare_calculation_methods(a: float, b: float, c: float,
                               alpha: float, beta: float, gamma: float,
                               miller_list: List[Tuple[int, int, int]]) -> Dict:
    """
    比较不同Miller指数的晶面间距
    
    批量计算多个晶面的间距，用于分析晶体的各向异性
    
    Parameters:
    -----------
    a, b, c : float
        晶格常数
    alpha, beta, gamma : float
        晶格角度
    miller_list : list[tuple[int, int, int]]
        Miller指数列表
    
    Returns:
    --------
    dict : {
        'result': list[dict],  # 每个晶面的计算结果
        'metadata': {
            'crystal_system': str,
            'num_planes': int,
            'min_d_spacing': float,
            'max_d_spacing': float
        }
    }
    """
    results = []
    
    # 识别晶系
    system_result = identify_crystal_system(a, b, c, alpha, beta, gamma)
    crystal_system = system_result['result']
    
    # 计算度规张量（只需计算一次）
    metric_result = calculate_metric_tensor(a, b, c, alpha, beta, gamma)
    metric_tensor = metric_result['result']
    
    for h, k, l in miller_list:
        d_result = calculate_interplanar_spacing_general(h, k, l, metric_tensor)
        results.append({
            'miller_indices': (h, k, l),
            'd_spacing': d_result['result'],
            'reciprocal_d_squared': d_result['metadata']['reciprocal_d_squared']
        })
    
    # 统计信息
    d_values = [r['d_spacing'] for r in results]
    
    return {
        'result': results,
        'metadata': {
            'crystal_system': crystal_system,
            'num_planes': len(miller_list),
            'min_d_spacing': float(min(d_values)),
            'max_d_spacing': float(max(d_values)),
            'lattice_parameters': {
                'a': a, 'b': b, 'c': c,
                'alpha': alpha, 'beta': beta, 'gamma': gamma
            }
        }
    }


def build_local_crystal_database(crystal_data: List[Dict]) -> Dict:
    """
    构建本地晶体数据库（SQLite）
    
    存储常见晶体的结构参数，用于快速查询
    
    Parameters:
    -----------
    crystal_data : list[dict]
        晶体数据列表，每个dict包含：
        {
            'name': str,
            'formula': str,
            'a': float, 'b': float, 'c': float,
            'alpha': float, 'beta': float, 'gamma': float,
            'space_group': str
        }
    
    Returns:
    --------
    dict : {
        'result': str,  # 数据库文件路径
        'metadata': {
            'num_entries': int,
            'database_type': str
        }
    }
    """
    import sqlite3
    
    db_path = './mid_result/crystallography/crystal_database.db'
    
    # 创建数据库连接
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS crystals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            formula TEXT,
            a REAL, b REAL, c REAL,
            alpha REAL, beta REAL, gamma REAL,
            space_group TEXT,
            crystal_system TEXT
        )
    ''')
    
    # 插入数据
    for crystal in crystal_data:
        # 识别晶系
        system_result = identify_crystal_system(
            crystal['a'], crystal['b'], crystal['c'],
            crystal['alpha'], crystal['beta'], crystal['gamma']
        )
        crystal_system = system_result['result']
        
        cursor.execute('''
            INSERT INTO crystals (name, formula, a, b, c, alpha, beta, gamma, space_group, crystal_system)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            crystal['name'],
            crystal.get('formula', ''),
            crystal['a'], crystal['b'], crystal['c'],
            crystal['alpha'], crystal['beta'], crystal['gamma'],
            crystal.get('space_group', ''),
            crystal_system
        ))
    
    conn.commit()
    conn.close()
    
    return {
        'result': db_path,
        'metadata': {
            'num_entries': len(crystal_data),
            'database_type': 'SQLite',
            'description': 'Local crystal structure database'
        }
    }


def query_crystal_database(db_path: str, query_type: str, 
                          query_value: str) -> Dict:
    """
    查询本地晶体数据库
    
    Parameters:
    -----------
    db_path : str
        数据库文件路径
    query_type : str
        查询类型：'name', 'formula', 'crystal_system'
    query_value : str
        查询值
    
    Returns:
    --------
    dict : {
        'result': list[dict],  # 查询结果
        'metadata': {
            'query_type': str,
            'num_results': int
        }
    }
    """
    import sqlite3
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 构建查询
    if query_type == 'name':
        cursor.execute('SELECT * FROM crystals WHERE name LIKE ?', (f'%{query_value}%',))
    elif query_type == 'formula':
        cursor.execute('SELECT * FROM crystals WHERE formula LIKE ?', (f'%{query_value}%',))
    elif query_type == 'crystal_system':
        cursor.execute('SELECT * FROM crystals WHERE crystal_system = ?', (query_value,))
    else:
        raise ValueError(f"不支持的查询类型: {query_type}")
    
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append(dict(zip(columns, row)))
    
    conn.close()
    
    return {
        'result': results,
        'metadata': {
            'query_type': query_type,
            'query_value': query_value,
            'num_results': len(results)
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def visualize_d_spacing_vs_miller_indices(comparison_result: Dict,
                                         save_path: str = None) -> Dict:
    """
    可视化不同Miller指数的晶面间距
    
    Parameters:
    -----------
    comparison_result : dict
        compare_calculation_methods()的返回结果
    save_path : str, optional
        图像保存路径
    
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': {
            'num_planes': int,
            'plot_type': str
        }
    }
    """
    if save_path is None:
        save_path = './tool_images/d_spacing_comparison.png'
    
    results = comparison_result['result']
    crystal_system = comparison_result['metadata']['crystal_system']
    
    # 提取数据
    miller_labels = [f"({r['miller_indices'][0]}{r['miller_indices'][1]}{r['miller_indices'][2]})" 
                     for r in results]
    d_spacings = [r['d_spacing'] for r in results]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(miller_labels)), d_spacings, 
                  color='steelblue', alpha=0.7, edgecolor='black')
    
    # 在柱状图上标注数值
    for i, (bar, d) in enumerate(zip(bars, d_spacings)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{d:.3f} Å',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Miller Indices (hkl)', fontsize=12, fontweight='bold')
    ax.set_ylabel('d-spacing (Angstrom)', fontsize=12, fontweight='bold')
    ax.set_title(f'Interplanar Spacing for {crystal_system} Crystal System', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(miller_labels)))
    ax.set_xticklabels(miller_labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'num_planes': len(results),
            'plot_type': 'bar_chart',
            'crystal_system': crystal_system,
            'file_type': 'png',
            'dpi': 300
        }
    }


def visualize_reciprocal_lattice_plane(h: int, k: int, l: int,
                                      metric_tensor: List[List[float]],
                                      save_path: str = None) -> Dict:
    """
    可视化倒易晶格平面
    
    绘制倒易空间中的晶面法向量和相关几何关系
    
    Parameters:
    -----------
    h, k, l : int
        Miller指数
    metric_tensor : list[list[float]]
        实空间度规张量
    save_path : str, optional
        图像保存路径
    
    Returns:
    --------
    dict : {
        'result': str,  # 图像文件路径
        'metadata': dict
    }
    """
    if save_path is None:
        save_path = f'./tool_images/reciprocal_lattice_{h}{k}{l}.png'
    
    # 计算倒易度规张量
    recip_result = calculate_reciprocal_metric_tensor(metric_tensor)
    g_star = np.array(recip_result['result'])
    
    # 计算倒易矢量
    reciprocal_vector = np.array([h, k, l])
    
    # 计算d-spacing
    d_result = calculate_interplanar_spacing_general(h, k, l, metric_tensor)
    d_spacing = d_result['result']
    
    # 创建3D图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制倒易矢量
    ax.quiver(0, 0, 0, h, k, l, color='red', arrow_length_ratio=0.1, 
              linewidth=3, label=f'Reciprocal vector ({h},{k},{l})')
    
    # 绘制坐标轴
    axis_length = max(abs(h), abs(k), abs(l)) * 1.5
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='gray', alpha=0.3, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='gray', alpha=0.3, arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='gray', alpha=0.3, arrow_length_ratio=0.05)
    
    # 绘制晶面（简化表示）
    xx, yy = np.meshgrid(np.linspace(-axis_length, axis_length, 10),
                         np.linspace(-axis_length, axis_length, 10))
    
    # 平面方程: hx + ky + lz = constant
    if l != 0:
        zz = (h*h + k*k + l*l - h*xx - k*yy) / l
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='blue')
    
    ax.set_xlabel('h*', fontsize=12)
    ax.set_ylabel('k*', fontsize=12)
    ax.set_zlabel('l*', fontsize=12)
    ax.set_title(f'Reciprocal Lattice Plane ({h}{k}{l})\nd-spacing = {d_spacing:.3f} Å', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'miller_indices': (h, k, l),
            'd_spacing': d_spacing,
            'plot_type': '3d_reciprocal_lattice',
            'file_type': 'png'
        }
    }


def create_crystal_structure_report(analysis_result: Dict,
                                   save_path: str = None) -> Dict:
    """
    生成晶体结构分析报告
    
    Parameters:
    -----------
    analysis_result : dict
        analyze_crystal_structure()的返回结果
    save_path : str, optional
        报告保存路径
    
    Returns:
    --------
    dict : {
        'result': str,  # 报告文件路径
        'metadata': dict
    }
    """
    if save_path is None:
        save_path = './mid_result/crystallography/structure_report.txt'
    
    result = analysis_result['result']
    metadata = analysis_result['metadata']
    
    report_lines = [
        "=" * 70,
        "CRYSTAL STRUCTURE ANALYSIS REPORT",
        "=" * 70,
        "",
        "1. LATTICE PARAMETERS",
        "-" * 70,
        f"   a = {metadata['lattice_parameters']['a']:.4f} Å",
        f"   b = {metadata['lattice_parameters']['b']:.4f} Å",
        f"   c = {metadata['lattice_parameters']['c']:.4f} Å",
        f"   α = {metadata['lattice_parameters']['alpha']:.2f}°",
        f"   β = {metadata['lattice_parameters']['beta']:.2f}°",
        f"   γ = {metadata['lattice_parameters']['gamma']:.2f}°",
        f"   Volume = {result['lattice_volume']:.4f} ų",
        "",
        "2. CRYSTAL SYSTEM",
        "-" * 70,
        f"   {result['crystal_system']}",
        "",
        "3. MILLER INDICES",
        "-" * 70,
        f"   (h k l) = {result['miller_indices']}",
        "",
        "4. INTERPLANAR SPACING",
        "-" * 70,
        f"   d-spacing (General formula) = {result['d_spacing_general']:.4f} Å",
    ]
    
    if result['d_spacing_specialized'] is not None:
        report_lines.append(f"   d-spacing (Specialized formula) = {result['d_spacing_specialized']:.4f} Å")
        diff = abs(result['d_spacing_general'] - result['d_spacing_specialized'])
        report_lines.append(f"   Difference = {diff:.6f} Å (验证一致性)")
    
    report_lines.extend([
        "",
        "5. CALCULATION METHODS",
        "-" * 70,
    ])
    
    for method in metadata['analysis_methods']:
        report_lines.append(f"   - {method}")
    
    report_lines.extend([
        "",
        "6. METRIC TENSOR (Direct Lattice)",
        "-" * 70,
    ])
    
    g = np.array(metadata['metric_tensor'])
    for i, row in enumerate(g):
        report_lines.append(f"   [{row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}]")
    
    report_lines.extend([
        "",
        "7. RECIPROCAL METRIC TENSOR",
        "-" * 70,
    ])
    
    g_star = np.array(metadata['reciprocal_metric_tensor'])
    for i, row in enumerate(g_star):
        report_lines.append(f"   [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}]")
    
    report_lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70
    ])
    
    # 写入文件
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"FILE_GENERATED: text | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'text',
            'num_lines': len(report_lines),
            'crystal_system': result['crystal_system'],
            'd_spacing': result['d_spacing_general']
        }
    }


# ============================================================================
# 主函数：演示3个场景
# ============================================================================

def main():
    """
    演示晶体学计算工具包的3个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 菱面体晶体(111)晶面间距计算")
    print("=" * 80)
    print("问题描述：计算菱面体晶体的(111)晶面间距")
    print("给定参数：a=b=c=10 Å, α=β=γ=30°")
    print("Miller指数：(1,1,1)")
    print("-" * 80)
    
    # 原始问题参数
    a = b = c = 10.0  # Angstrom
    alpha = beta = gamma = 30.0  # degrees
    h, k, l = 1, 1, 1
    
    # 步骤1：识别晶系
    # 调用函数：identify_crystal_system()
    system_result = identify_crystal_system(a, b, c, alpha, beta, gamma)
    print(f"FUNCTION_CALL: identify_crystal_system | PARAMS: {{a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}}} | RESULT: {system_result}")
    print(f"识别的晶系: {system_result['result']}")
    print(f"对称条件: {system_result['metadata']['symmetry_conditions']}")
    print()
    
    # 步骤2：计算度规张量
    # 调用函数：calculate_metric_tensor()
    metric_result = calculate_metric_tensor(a, b, c, alpha, beta, gamma)
    print(f"FUNCTION_CALL: calculate_metric_tensor | PARAMS: {{a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}}} | RESULT: {metric_result}")
    print(f"晶胞体积: {metric_result['metadata']['volume']:.4f} ų")
    print()
    
    # 步骤3：使用通用公式计算晶面间距
    # 调用函数：calculate_interplanar_spacing_general()
    d_general_result = calculate_interplanar_spacing_general(h, k, l, metric_result['result'])
    print(f"FUNCTION_CALL: calculate_interplanar_spacing_general | PARAMS: {{h={h}, k={k}, l={l}, metric_tensor=...}} | RESULT: {d_general_result}")
    print(f"晶面间距 (通用公式): {d_general_result['result']:.4f} Å")
    print()
    
    # 步骤4：使用菱面体专用公式验证
    # 调用函数：calculate_rhombohedral_d_spacing()
    d_rhombo_result = calculate_rhombohedral_d_spacing(h, k, l, a, alpha)
    print(f"FUNCTION_CALL: calculate_rhombohedral_d_spacing | PARAMS: {{h={h}, k={k}, l={l}, a={a}, alpha={alpha}}} | RESULT: {d_rhombo_result}")
    print(f"晶面间距 (菱面体公式): {d_rhombo_result['result']:.4f} Å")
    print()
    
    # 步骤5：综合分析
    # 调用函数：analyze_crystal_structure()
    analysis_result = analyze_crystal_structure(a, b, c, alpha, beta, gamma, (h, k, l))
    print(f"FUNCTION_CALL: analyze_crystal_structure | PARAMS: {{a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}, miller_indices=({h},{k},{l})}} | RESULT: {analysis_result}")
    print()
    
    # 步骤6：生成分析报告
    # 调用函数：create_crystal_structure_report()
    report_result = create_crystal_structure_report(analysis_result)
    print(f"FUNCTION_CALL: create_crystal_structure_report | PARAMS: {{analysis_result=...}} | RESULT: {report_result}")
    print()
    
    final_answer_1 = d_general_result['result']
    print(f"场景1最终答案: (111)晶面间距 = {final_answer_1:.2f} Å")
    print(f"与标准答案(9.54 Å)的误差: {abs(final_answer_1 - 9.54):.4f} Å")
    print()
    
    
    print("=" * 80)
    print("场景2：多晶面间距比较 - 分析菱面体晶体的各向异性")
    print("=" * 80)
    print("问题描述：计算同一菱面体晶体的多个晶面间距，分析结构各向异性")
    print("晶面列表：(100), (110), (111), (200), (210), (211)")
    print("-" * 80)
    
    # 定义多个Miller指数
    miller_list = [(1,0,0), (1,1,0), (1,1,1), (2,0,0), (2,1,0), (2,1,1)]
    
    # 步骤1：批量计算晶面间距
    # 调用函数：compare_calculation_methods()
    comparison_result = compare_calculation_methods(a, b, c, alpha, beta, gamma, miller_list)
    print(f"FUNCTION_CALL: compare_calculation_methods | PARAMS: {{a={a}, b={b}, c={c}, alpha={alpha}, beta={beta}, gamma={gamma}, miller_list={miller_list}}} | RESULT: {comparison_result}")
    print()
    
    print("各晶面间距结果:")
    for item in comparison_result['result']:
        print(f"  {item['miller_indices']}: d = {item['d_spacing']:.4f} Å")
    print()
    
    print(f"最小晶面间距: {comparison_result['metadata']['min_d_spacing']:.4f} Å")
    print(f"最大晶面间距: {comparison_result['metadata']['max_d_spacing']:.4f} Å")
    print()
    
    # 步骤2：可视化比较结果
    # 调用函数：visualize_d_spacing_vs_miller_indices()
    viz_result = visualize_d_spacing_vs_miller_indices(comparison_result)
    print(f"FUNCTION_CALL: visualize_d_spacing_vs_miller_indices | PARAMS: {{comparison_result=...}} | RESULT: {viz_result}")
    print()
    
    # 步骤3：可视化倒易晶格
    # 调用函数：visualize_reciprocal_lattice_plane()
    recip_viz_result = visualize_reciprocal_lattice_plane(1, 1, 1, metric_result['result'])
    print(f"FUNCTION_CALL: visualize_reciprocal_lattice_plane | PARAMS: {{h=1, k=1, l=1, metric_tensor=...}} | RESULT: {recip_viz_result}")
    print()
    
    final_answer_2 = f"成功分析了{len(miller_list)}个晶面，间距范围: {comparison_result['metadata']['min_d_spacing']:.4f} - {comparison_result['metadata']['max_d_spacing']:.4f} Å"
    print(f"FINAL_ANSWER: {final_answer_2}")
    print()
    
    
    print("=" * 80)
    print("场景3：晶体数据库构建与查询 - 常见晶体结构管理")
    print("=" * 80)
    print("问题描述：构建包含常见晶体结构的本地数据库，并进行查询分析")
    print("-" * 80)
    
    # 步骤1：准备晶体数据
    crystal_data = [
        {
            'name': 'Rhombohedral_Example',
            'formula': 'Example',
            'a': 10.0, 'b': 10.0, 'c': 10.0,
            'alpha': 30.0, 'beta': 30.0, 'gamma': 30.0,
            'space_group': 'R-3m'
        },
        {
            'name': 'Calcite',
            'formula': 'CaCO3',
            'a': 4.99, 'b': 4.99, 'c': 17.06,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0,
            'space_group': 'R-3c'
        },
        {
            'name': 'Corundum',
            'formula': 'Al2O3',
            'a': 4.76, 'b': 4.76, 'c': 12.99,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0,
            'space_group': 'R-3c'
        },
        {
            'name': 'Quartz',
            'formula': 'SiO2',
            'a': 4.916, 'b': 4.916, 'c': 5.405,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 120.0,
            'space_group': 'P3221'
        },
        {
            'name': 'NaCl',
            'formula': 'NaCl',
            'a': 5.64, 'b': 5.64, 'c': 5.64,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
            'space_group': 'Fm-3m'
        }
    ]
    
    # 步骤2：构建数据库
    # 调用函数：build_local_crystal_database()
    db_result = build_local_crystal_database(crystal_data)
    print(f"FUNCTION_CALL: build_local_crystal_database | PARAMS: {{crystal_data=[{len(crystal_data)} entries]}} | RESULT: {db_result}")
    print(f"数据库已创建: {db_result['result']}")
    print(f"包含 {db_result['metadata']['num_entries']} 条晶体结构数据")
    print()
    
    # 步骤3：查询菱面体晶系的晶体
    # 调用函数：query_crystal_database()
    query_result = query_crystal_database(db_result['result'], 'crystal_system', 'Rhombohedral')
    print(f"FUNCTION_CALL: query_crystal_database | PARAMS: {{db_path='{db_result['result']}', query_type='crystal_system', query_value='Rhombohedral'}} | RESULT: {query_result}")
    print(f"查询到 {query_result['metadata']['num_results']} 个菱面体晶体:")
    for crystal in query_result['result']:
        print(f"  - {crystal['name']} ({crystal['formula']})")
    print()
    
    # 步骤4：查询六方晶系的晶体
    # 调用函数：query_crystal_database()
    query_result_hex = query_crystal_database(db_result['result'], 'crystal_system', 'Hexagonal')
    print(f"FUNCTION_CALL: query_crystal_database | PARAMS: {{db_path='{db_result['result']}', query_type='crystal_system', query_value='Hexagonal'}} | RESULT: {query_result_hex}")
    print(f"查询到 {query_result_hex['metadata']['num_results']} 个六方晶体:")
    for crystal in query_result_hex['result']:
        print(f"  - {crystal['name']} ({crystal['formula']})")
        # 计算该晶体的(111)晶面间距
        cryst_a = crystal['a']
        cryst_b = crystal['b']
        cryst_c = crystal['c']
        cryst_alpha = crystal['alpha']
        cryst_beta = crystal['beta']
        cryst_gamma = crystal['gamma']
        
        metric_cryst = calculate_metric_tensor(cryst_a, cryst_b, cryst_c, 
                                               cryst_alpha, cryst_beta, cryst_gamma)
        d_cryst = calculate_interplanar_spacing_general(1, 1, 1, metric_cryst['result'])
        print(f"    (111)晶面间距: {d_cryst['result']:.4f} Å")
    print()
    
    final_answer_3 = f"成功构建晶体数据库，包含{db_result['metadata']['num_entries']}种晶体，查询到{query_result['metadata']['num_results']}个菱面体晶体和{query_result_hex['metadata']['num_results']}个六方晶体"
    print(f"FINAL_ANSWER: {final_answer_3}")
    print()
    
    # 最终总结
    print("=" * 80)
    print("工具包演示完成")
    print("=" * 80)
    print(f"场景1答案: (111)晶面间距 = {final_answer_1:.2f} Å (标准答案: 9.54 Å)")
    print(f"FINAL_ANSWER: {final_answer_1:.2f} Angstrom")


if __name__ == "__main__":
    main()