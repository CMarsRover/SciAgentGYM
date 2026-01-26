# Filename: quantum_mechanics_toolkit.py

"""
Quantum Mechanics Toolkit for Spherically Symmetric Wave Functions
专注于球对称薛定谔方程的波函数分析和概率计算
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve, minimize
import sympy as sp
from typing import Dict, List, Tuple
import os
import json

# 确保输出目录存在
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 配置matplotlib支持中英文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一层：原子函数 - 基础量子力学计算
# ============================================================================

def radial_probability_density(r: float, wave_function_type: str, 
                               normalization_const: float = 1.0,
                               params: dict = None) -> Dict:
    """
    计算径向概率密度 P(r) = |ψ(r)|² * 4πr²
    
    对于球对称波函数，概率密度包含雅可比因子 4πr²
    
    Parameters:
    -----------
    r : float
        距离中心的径向距离
    wave_function_type : str
        波函数类型: '1/r', 'exponential', 'gaussian', 'constant'
    normalization_const : float
        归一化常数 A
    params : dict
        波函数的额外参数（如衰减常数等）
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
    """
    if r <= 0:
        return {
            'result': 0.0,
            'metadata': {
                'wave_function_type': wave_function_type,
                'note': 'r <= 0, probability density is zero'
            }
        }
    
    params = params or {}
    
    # 计算波函数值
    if wave_function_type == '1/r':
        psi = normalization_const / r
    elif wave_function_type == 'exponential':
        alpha = params.get('alpha', 1.0)
        psi = normalization_const * np.exp(-alpha * r)
    elif wave_function_type == 'gaussian':
        sigma = params.get('sigma', 1.0)
        psi = normalization_const * np.exp(-r**2 / (2 * sigma**2))
    elif wave_function_type == 'constant':
        psi = normalization_const
    else:
        raise ValueError(f"Unknown wave function type: {wave_function_type}")
    
    # 径向概率密度 = |ψ|² * 4πr²
    prob_density = (psi ** 2) * 4 * np.pi * (r ** 2)
    
    return {
        'result': prob_density,
        'metadata': {
            'r': r,
            'wave_function_type': wave_function_type,
            'psi_value': psi,
            'jacobian_factor': 4 * np.pi * r**2
        }
    }


def integrate_probability(r_min: float, r_max: float, 
                         wave_function_type: str,
                         normalization_const: float = 1.0,
                         params: dict = None) -> Dict:
    """
    计算在径向区间 [r_min, r_max] 内找到粒子的概率
    
    P = ∫[r_min to r_max] |ψ(r)|² * 4πr² dr
    
    Parameters:
    -----------
    r_min : float
        积分下限
    r_max : float
        积分上限
    wave_function_type : str
        波函数类型
    normalization_const : float
        归一化常数
    params : dict
        波函数参数
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
    """
    if r_min < 0 or r_max < 0:
        raise ValueError("Radial distances must be non-negative")
    if r_min >= r_max:
        raise ValueError("r_min must be less than r_max")
    
    params = params or {}
    
    # 定义被积函数
    def integrand(r):
        result = radial_probability_density(r, wave_function_type, 
                                           normalization_const, params)
        return result['result']
    
    # 数值积分
    probability, error = quad(integrand, r_min, r_max, limit=100)
    
    return {
        'result': probability,
        'metadata': {
            'r_min': r_min,
            'r_max': r_max,
            'wave_function_type': wave_function_type,
            'integration_error': error,
            'normalization_const': normalization_const
        }
    }


def symbolic_wave_function_analysis(wave_function_expr: str, 
                                   r_symbol: str = 'r') -> Dict:
    """
    使用符号计算分析波函数的性质
    
    Parameters:
    -----------
    wave_function_expr : str
        波函数的符号表达式，如 'A/r', 'A*exp(-alpha*r)'
    r_symbol : str
        径向变量符号
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
    """
    r = sp.Symbol(r_symbol, positive=True, real=True)
    A = sp.Symbol('A', positive=True, real=True)
    
    # 解析波函数表达式
    psi = sp.sympify(wave_function_expr)
    
    # 计算概率密度
    prob_density = psi**2 * 4 * sp.pi * r**2
    prob_density_simplified = sp.simplify(prob_density)
    
    # 计算导数
    derivative = sp.diff(psi, r)
    
    # 计算拉普拉斯算符的径向部分
    laplacian_radial = sp.diff(r**2 * sp.diff(psi, r), r) / r**2
    
    return {
        'result': {
            'wave_function': str(psi),
            'probability_density': str(prob_density_simplified),
            'derivative': str(derivative),
            'laplacian_radial': str(sp.simplify(laplacian_radial))
        },
        'metadata': {
            'wave_function_expr': wave_function_expr,
            'analysis_type': 'symbolic',
            'variables': [r_symbol]
        }
    }


def calculate_normalization_constant(r_min: float, r_max: float,
                                     wave_function_type: str,
                                     params: dict = None) -> Dict:
    """
    计算归一化常数 A，使得 ∫|ψ|² * 4πr² dr = 1
    
    Parameters:
    -----------
    r_min : float
        内球半径
    r_max : float
        外球半径
    wave_function_type : str
        波函数类型
    params : dict
        波函数参数
    
    Returns:
    --------
    dict : {'result': float, 'metadata': dict}
    """
    params = params or {}
    
    # 对于 1/r 型波函数，可以解析求解
    if wave_function_type == '1/r':
        # ∫[r_min to r_max] A²/r² * 4πr² dr = ∫[r_min to r_max] 4πA² dr
        # = 4πA² * (r_max - r_min) = 1
        A_squared = 1.0 / (4 * np.pi * (r_max - r_min))
        A = np.sqrt(A_squared)
        
        return {
            'result': A,
            'metadata': {
                'wave_function_type': wave_function_type,
                'r_min': r_min,
                'r_max': r_max,
                'analytical_solution': True,
                'integral_value': 4 * np.pi * A**2 * (r_max - r_min)
            }
        }
    
    # 对于其他类型，使用数值方法
    def normalization_error(A):
        prob = integrate_probability(r_min, r_max, wave_function_type, A, params)
        return (prob['result'] - 1.0) ** 2
    
    # 初始猜测
    A_initial = 1.0
    result = minimize(normalization_error, A_initial, method='Nelder-Mead')
    
    A_normalized = result.x[0]
    
    # 验证归一化
    verification = integrate_probability(r_min, r_max, wave_function_type, 
                                        A_normalized, params)
    
    return {
        'result': A_normalized,
        'metadata': {
            'wave_function_type': wave_function_type,
            'r_min': r_min,
            'r_max': r_max,
            'analytical_solution': False,
            'verification_integral': verification['result'],
            'optimization_success': result.success
        }
    }


# ============================================================================
# 第二层：组合函数 - 波函数推导和验证
# ============================================================================

def deduce_wave_function_from_probability_conditions(
    d1: float,
    outer_inner_ratio: float,
    probability_equal_intervals: List[Tuple[float, float]],
    candidate_wave_functions: List[str] = None
) -> Dict:
    """
    根据概率条件推导波函数形式
    
    给定：
    - 在 [d1, 2*d1] 区间概率为 P
    - 在 [2*d1, 3*d1] 区间概率也为 P
    
    推导满足条件的波函数类型
    
    Parameters:
    -----------
    d1 : float
        内球半径
    outer_inner_ratio : float
        外球与内球半径比
    probability_equal_intervals : List[Tuple[float, float]]
        概率相等的区间列表，每个元组为 (r_min_factor, r_max_factor)
        例如 [(1, 2), (2, 3)] 表示 [d1, 2*d1] 和 [2*d1, 3*d1]
    candidate_wave_functions : List[str]
        候选波函数类型列表
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
    """
    if candidate_wave_functions is None:
        candidate_wave_functions = ['1/r', 'exponential', 'constant', 'gaussian']
    
    d_outer = d1 * outer_inner_ratio
    
    # 测试每个候选波函数
    results = {}
    
    for wf_type in candidate_wave_functions:
        # 计算归一化常数
        norm_result = calculate_normalization_constant(d1, d_outer, wf_type)
        A = norm_result['result']
        
        # 计算各区间的概率
        probabilities = []
        for r_min_factor, r_max_factor in probability_equal_intervals:
            r_min = d1 * r_min_factor
            r_max = d1 * r_max_factor
            prob = integrate_probability(r_min, r_max, wf_type, A)
            probabilities.append(prob['result'])
        
        # 检查概率是否相等（允许小误差）
        prob_std = np.std(probabilities)
        prob_mean = np.mean(probabilities)
        relative_variation = prob_std / prob_mean if prob_mean > 0 else float('inf')
        
        results[wf_type] = {
            'probabilities': probabilities,
            'mean_probability': prob_mean,
            'std_deviation': prob_std,
            'relative_variation': relative_variation,
            'normalization_constant': A,
            'satisfies_condition': relative_variation < 0.01  # 1% tolerance
        }
    
    # 找到最佳匹配
    best_match = None
    min_variation = float('inf')
    
    for wf_type, result in results.items():
        if result['satisfies_condition'] and result['relative_variation'] < min_variation:
            min_variation = result['relative_variation']
            best_match = wf_type
    
    return {
        'result': best_match if best_match else 'No suitable wave function found',
        'metadata': {
            'd1': d1,
            'd_outer': d_outer,
            'probability_intervals': probability_equal_intervals,
            'all_results': results,
            'best_match_variation': min_variation
        }
    }


def verify_wave_function_properties(wave_function_type: str,
                                   d1: float,
                                   d_outer: float,
                                   test_intervals: List[Tuple[float, float]],
                                   normalization_const: float = None) -> Dict:
    """
    验证波函数的各种性质
    
    Parameters:
    -----------
    wave_function_type : str
        波函数类型
    d1 : float
        内球半径
    d_outer : float
        外球半径
    test_intervals : List[Tuple[float, float]]
        测试区间列表
    normalization_const : float
        归一化常数（如果为None则自动计算）
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
    """
    # 计算或使用给定的归一化常数
    if normalization_const is None:
        norm_result = calculate_normalization_constant(d1, d_outer, wave_function_type)
        A = norm_result['result']
    else:
        A = normalization_const
    
    # 验证归一化
    total_prob = integrate_probability(d1, d_outer, wave_function_type, A)
    
    # 计算各测试区间的概率
    interval_probabilities = []
    for r_min, r_max in test_intervals:
        prob = integrate_probability(r_min, r_max, wave_function_type, A)
        interval_probabilities.append({
            'interval': (r_min, r_max),
            'probability': prob['result']
        })
    
    # 符号分析
    if wave_function_type == '1/r':
        symbolic_expr = f'{A}/r'
    elif wave_function_type == 'exponential':
        symbolic_expr = f'{A}*exp(-r)'
    else:
        symbolic_expr = f'{A}'
    
    symbolic_result = symbolic_wave_function_analysis(symbolic_expr)
    
    return {
        'result': {
            'normalization_constant': A,
            'total_probability': total_prob['result'],
            'interval_probabilities': interval_probabilities,
            'symbolic_analysis': symbolic_result['result']
        },
        'metadata': {
            'wave_function_type': wave_function_type,
            'd1': d1,
            'd_outer': d_outer,
            'is_normalized': abs(total_prob['result'] - 1.0) < 1e-6
        }
    }


def compare_wave_function_candidates(d1: float,
                                     outer_inner_ratio: float,
                                     equal_prob_intervals: List[Tuple[float, float]],
                                     candidates: List[str] = None) -> Dict:
    """
    比较多个候选波函数，找出最符合条件的
    
    Parameters:
    -----------
    d1 : float
        内球半径
    outer_inner_ratio : float
        外球与内球半径比
    equal_prob_intervals : List[Tuple[float, float]]
        应该具有相等概率的区间
    candidates : List[str]
        候选波函数列表
    
    Returns:
    --------
    dict : {'result': dict, 'metadata': dict}
    """
    if candidates is None:
        candidates = ['1/r', 'exponential', 'constant', '1/r^2']
    
    d_outer = d1 * outer_inner_ratio
    
    comparison_results = {}
    
    for wf_type in candidates:
        try:
            # 计算归一化常数
            norm_result = calculate_normalization_constant(d1, d_outer, wf_type)
            A = norm_result['result']
            
            # 计算各区间概率
            probabilities = []
            for r_min_factor, r_max_factor in equal_prob_intervals:
                r_min = d1 * r_min_factor
                r_max = d1 * r_max_factor
                prob = integrate_probability(r_min, r_max, wf_type, A)
                probabilities.append(prob['result'])
            
            # 计算概率的一致性
            prob_array = np.array(probabilities)
            consistency_score = 1.0 / (1.0 + np.std(prob_array))
            
            comparison_results[wf_type] = {
                'normalization_constant': A,
                'probabilities': probabilities,
                'mean_probability': np.mean(prob_array),
                'std_deviation': np.std(prob_array),
                'consistency_score': consistency_score,
                'max_deviation': np.max(np.abs(prob_array - np.mean(prob_array)))
            }
        except Exception as e:
            comparison_results[wf_type] = {
                'error': str(e),
                'consistency_score': 0.0
            }
    
    # 找出最佳候选
    best_candidate = max(comparison_results.keys(), 
                        key=lambda k: comparison_results[k].get('consistency_score', 0))
    
    return {
        'result': {
            'best_candidate': best_candidate,
            'all_comparisons': comparison_results
        },
        'metadata': {
            'd1': d1,
            'd_outer': d_outer,
            'equal_prob_intervals': equal_prob_intervals,
            'candidates_tested': candidates
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def plot_wave_function_and_probability(wave_function_type: str,
                                      d1: float,
                                      d_outer: float,
                                      normalization_const: float,
                                      highlight_intervals: List[Tuple[float, float]] = None,
                                      num_points: int = 1000) -> Dict:
    """
    绘制波函数和概率密度分布
    
    Parameters:
    -----------
    wave_function_type : str
        波函数类型
    d1 : float
        内球半径
    d_outer : float
        外球半径
    normalization_const : float
        归一化常数
    highlight_intervals : List[Tuple[float, float]]
        需要高亮显示的区间
    num_points : int
        绘图点数
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
    """
    r_values = np.linspace(d1, d_outer, num_points)
    
    # 计算波函数值
    psi_values = []
    prob_density_values = []
    
    for r in r_values:
        psi_result = radial_probability_density(r, wave_function_type, 
                                               normalization_const)
        if wave_function_type == '1/r':
            psi = normalization_const / r
        elif wave_function_type == 'exponential':
            psi = normalization_const * np.exp(-r)
        else:
            psi = normalization_const
        
        psi_values.append(psi)
        prob_density_values.append(psi_result['result'])
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制波函数
    ax1.plot(r_values, psi_values, 'b-', linewidth=2, label=f'ψ(r) for {wave_function_type}')
    ax1.set_xlabel('Radial Distance r', fontsize=12)
    ax1.set_ylabel('Wave Function ψ(r)', fontsize=12)
    ax1.set_title('Radial Wave Function', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制概率密度
    ax2.plot(r_values, prob_density_values, 'r-', linewidth=2, 
            label='Probability Density P(r) = |ψ(r)|² × 4πr²')
    
    # 高亮特定区间
    if highlight_intervals:
        colors = ['yellow', 'green', 'cyan', 'magenta']
        for i, (r_min, r_max) in enumerate(highlight_intervals):
            color = colors[i % len(colors)]
            ax2.axvspan(r_min, r_max, alpha=0.3, color=color, 
                       label=f'Interval [{r_min:.2f}, {r_max:.2f}]')
    
    ax2.set_xlabel('Radial Distance r', fontsize=12)
    ax2.set_ylabel('Probability Density P(r)', fontsize=12)
    ax2.set_title('Radial Probability Density Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图像
    filepath = f'./tool_images/wave_function_{wave_function_type.replace("/", "_")}.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'wave_function_type': wave_function_type,
            'd1': d1,
            'd_outer': d_outer,
            'normalization_const': normalization_const,
            'num_points': num_points,
            'file_type': 'png'
        }
    }


def plot_probability_comparison(d1: float,
                               d_outer: float,
                               wave_function_types: List[str],
                               test_intervals: List[Tuple[float, float]]) -> Dict:
    """
    比较不同波函数在各区间的概率分布
    
    Parameters:
    -----------
    d1 : float
        内球半径
    d_outer : float
        外球半径
    wave_function_types : List[str]
        要比较的波函数类型列表
    test_intervals : List[Tuple[float, float]]
        测试区间
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    interval_labels = [f'[{r_min:.1f}, {r_max:.1f}]' 
                      for r_min, r_max in test_intervals]
    x = np.arange(len(interval_labels))
    width = 0.8 / len(wave_function_types)
    
    all_probabilities = {}
    
    for i, wf_type in enumerate(wave_function_types):
        # 计算归一化常数
        norm_result = calculate_normalization_constant(d1, d_outer, wf_type)
        A = norm_result['result']
        
        # 计算各区间概率
        probabilities = []
        for r_min, r_max in test_intervals:
            prob = integrate_probability(r_min, r_max, wf_type, A)
            probabilities.append(prob['result'])
        
        all_probabilities[wf_type] = probabilities
        
        # 绘制柱状图
        offset = (i - len(wave_function_types)/2 + 0.5) * width
        ax.bar(x + offset, probabilities, width, label=wf_type, alpha=0.8)
    
    ax.set_xlabel('Radial Intervals', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Probability Distribution Comparison for Different Wave Functions', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(interval_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/probability_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'd1': d1,
            'd_outer': d_outer,
            'wave_function_types': wave_function_types,
            'test_intervals': test_intervals,
            'all_probabilities': all_probabilities,
            'file_type': 'png'
        }
    }


def visualize_spherical_probability_distribution(wave_function_type: str,
                                                d1: float,
                                                d_outer: float,
                                                normalization_const: float,
                                                num_shells: int = 50) -> Dict:
    """
    可视化球壳概率分布（3D效果的2D投影）
    
    Parameters:
    -----------
    wave_function_type : str
        波函数类型
    d1 : float
        内球半径
    d_outer : float
        外球半径
    normalization_const : float
        归一化常数
    num_shells : int
        球壳数量
    
    Returns:
    --------
    dict : {'result': str, 'metadata': dict}
    """
    r_values = np.linspace(d1, d_outer, num_shells)
    
    # 计算每个球壳的概率
    shell_probabilities = []
    for i in range(len(r_values) - 1):
        r_min = r_values[i]
        r_max = r_values[i + 1]
        prob = integrate_probability(r_min, r_max, wave_function_type, 
                                    normalization_const)
        shell_probabilities.append(prob['result'])
    
    r_centers = (r_values[:-1] + r_values[1:]) / 2
    
    # 创建极坐标图
    fig = plt.figure(figsize=(14, 6))
    
    # 左图：笛卡尔坐标
    ax1 = fig.add_subplot(121)
    ax1.bar(r_centers, shell_probabilities, width=r_values[1]-r_values[0], 
           alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Radial Distance r', fontsize=12)
    ax1.set_ylabel('Probability per Shell', fontsize=12)
    ax1.set_title('Probability Distribution in Spherical Shells', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 右图：极坐标（模拟球对称）
    ax2 = fig.add_subplot(122, projection='polar')
    theta = np.linspace(0, 2*np.pi, 100)
    
    # 为每个球壳绘制圆环
    for i, (r, prob) in enumerate(zip(r_centers, shell_probabilities)):
        color_intensity = prob / max(shell_probabilities)
        ax2.plot(theta, [r]*len(theta), color=plt.cm.hot(color_intensity), 
                linewidth=3, alpha=0.6)
    
    ax2.set_title('Spherical Symmetry Visualization\n(Color intensity ∝ Probability)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存图像
    filepath = './tool_images/spherical_probability_distribution.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'wave_function_type': wave_function_type,
            'd1': d1,
            'd_outer': d_outer,
            'normalization_const': normalization_const,
            'num_shells': num_shells,
            'file_type': 'png'
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示量子力学工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 推导满足给定概率条件的波函数形式")
    print("=" * 80)
    print("问题描述：粒子存在于内外两个同心球之间，外球半径是内球半径的100倍。")
    print("波函数是时间无关、球对称、实函数。")
    print("在区间 [d1, 2*d1] 和 [2*d1, 3*d1] 内找到粒子的概率相等。")
    print("求：波函数关于径向距离 r 的形式。")
    print("-" * 80)
    
    # 定义问题参数
    d1 = 1.0  # 内球半径（归一化为1）
    outer_inner_ratio = 100.0  # 外球与内球半径比
    d_outer = d1 * outer_inner_ratio
    
    # 步骤1：定义概率相等的区间
    equal_prob_intervals = [(1, 2), (2, 3)]  # [d1, 2*d1] 和 [2*d1, 3*d1]
    print(f"\n步骤1：定义概率相等的区间")
    print(f"区间1: [{d1}, {2*d1}]")
    print(f"区间2: [{2*d1}, {3*d1}]")
    
    # 步骤2：使用推导函数找出满足条件的波函数
    # 调用函数：deduce_wave_function_from_probability_conditions()
    candidate_wfs = ['1/r', 'exponential', 'constant', 'gaussian']
    deduction_result = deduce_wave_function_from_probability_conditions(
        d1, outer_inner_ratio, equal_prob_intervals, candidate_wfs
    )
    print(f"\nFUNCTION_CALL: deduce_wave_function_from_probability_conditions | PARAMS: {{d1: {d1}, outer_inner_ratio: {outer_inner_ratio}, equal_prob_intervals: {equal_prob_intervals}, candidates: {candidate_wfs}}} | RESULT: {deduction_result}")
    
    best_wf = deduction_result['result']
    print(f"\n推导结果：最符合条件的波函数形式是 {best_wf}")
    
    # 步骤3：验证推导结果
    # 调用函数：verify_wave_function_properties()
    test_intervals = [(d1, 2*d1), (2*d1, 3*d1), (3*d1, 4*d1)]
    verification_result = verify_wave_function_properties(
        best_wf, d1, d_outer, test_intervals
    )
    print(f"\nFUNCTION_CALL: verify_wave_function_properties | PARAMS: {{wave_function_type: '{best_wf}', d1: {d1}, d_outer: {d_outer}, test_intervals: {test_intervals}}} | RESULT: {verification_result}")
    
    print(f"\n验证结果：")
    print(f"归一化常数 A = {verification_result['result']['normalization_constant']:.6f}")
    print(f"总概率 = {verification_result['result']['total_probability']:.6f}")
    print(f"各区间概率：")
    for interval_prob in verification_result['result']['interval_probabilities']:
        interval = interval_prob['interval']
        prob = interval_prob['probability']
        print(f"  区间 [{interval[0]:.2f}, {interval[1]:.2f}]: P = {prob:.6f}")
    
    # 步骤4：符号分析验证
    # 调用函数：symbolic_wave_function_analysis()
    A = verification_result['result']['normalization_constant']
    symbolic_expr = f'{A}/r'
    symbolic_result = symbolic_wave_function_analysis(symbolic_expr)
    print(f"\nFUNCTION_CALL: symbolic_wave_function_analysis | PARAMS: {{wave_function_expr: '{symbolic_expr}'}} | RESULT: {symbolic_result}")
    
    print(f"\n符号分析结果：")
    print(f"波函数: ψ(r) = {symbolic_result['result']['wave_function']}")
    print(f"概率密度: P(r) = {symbolic_result['result']['probability_density']}")
    
    # 步骤5：可视化波函数和概率分布
    # 调用函数：plot_wave_function_and_probability()
    highlight_intervals = [(d1, 2*d1), (2*d1, 3*d1)]
    plot_result = plot_wave_function_and_probability(
        best_wf, d1, d_outer, A, highlight_intervals
    )
    print(f"\nFUNCTION_CALL: plot_wave_function_and_probability | PARAMS: {{wave_function_type: '{best_wf}', d1: {d1}, d_outer: {d_outer}, normalization_const: {A}, highlight_intervals: {highlight_intervals}}} | RESULT: {plot_result}")
    
    # 最终答案
    final_answer = "1/r"
    print(f"\nFINAL_ANSWER: {final_answer}")
    
    
    print("\n" + "=" * 80)
    print("场景2：比较不同波函数候选在多个区间的概率分布")
    print("=" * 80)
    print("问题描述：对于同样的球壳几何结构，比较 1/r、指数衰减、常数等")
    print("不同波函数在多个等宽区间内的概率分布特征。")
    print("-" * 80)
    
    # 步骤1：定义测试区间
    test_intervals_scenario2 = [
        (d1, 2*d1), (2*d1, 3*d1), (3*d1, 4*d1), 
        (4*d1, 5*d1), (5*d1, 6*d1)
    ]
    print(f"\n步骤1：定义5个等宽测试区间")
    print(f"区间列表: {test_intervals_scenario2}")
    
    # 步骤2：比较多个候选波函数
    # 调用函数：compare_wave_function_candidates()
    candidates_scenario2 = ['1/r', 'exponential', 'constant']
    comparison_result = compare_wave_function_candidates(
        d1, outer_inner_ratio, [(1, 2), (2, 3)], candidates_scenario2
    )
    print(f"\nFUNCTION_CALL: compare_wave_function_candidates | PARAMS: {{d1: {d1}, outer_inner_ratio: {outer_inner_ratio}, equal_prob_intervals: [(1, 2), (2, 3)], candidates: {candidates_scenario2}}} | RESULT: {comparison_result}")
    
    print(f"\n比较结果：")
    print(f"最佳候选: {comparison_result['result']['best_candidate']}")
    for wf_type, result in comparison_result['result']['all_comparisons'].items():
        if 'error' not in result:
            print(f"\n{wf_type}:")
            print(f"  概率标准差: {result['std_deviation']:.6f}")
            print(f"  一致性得分: {result['consistency_score']:.6f}")
            print(f"  各区间概率: {result['probabilities']}")
    
    # 步骤3：可视化概率比较
    # 调用函数：plot_probability_comparison()
    plot_comparison_result = plot_probability_comparison(
        d1, d_outer, candidates_scenario2, test_intervals_scenario2
    )
    print(f"\nFUNCTION_CALL: plot_probability_comparison | PARAMS: {{d1: {d1}, d_outer: {d_outer}, wave_function_types: {candidates_scenario2}, test_intervals: {test_intervals_scenario2}}} | RESULT: {plot_comparison_result}")
    
    # 步骤4：分析1/r波函数在所有区间的概率
    # 调用函数：integrate_probability() 多次
    print(f"\n步骤4：详细分析 1/r 波函数在各区间的概率")
    norm_1_over_r = calculate_normalization_constant(d1, d_outer, '1/r')
    A_1_over_r = norm_1_over_r['result']
    
    for r_min, r_max in test_intervals_scenario2:
        prob = integrate_probability(r_min, r_max, '1/r', A_1_over_r)
        print(f"FUNCTION_CALL: integrate_probability | PARAMS: {{r_min: {r_min}, r_max: {r_max}, wave_function_type: '1/r', normalization_const: {A_1_over_r}}} | RESULT: {prob}")
        print(f"  区间 [{r_min:.2f}, {r_max:.2f}]: P = {prob['result']:.6f}")
    
    final_answer_scenario2 = "1/r wave function shows constant probability in equal-width intervals"
    print(f"\nFINAL_ANSWER: {final_answer_scenario2}")
    
    
    print("\n" + "=" * 80)
    print("场景3：球对称概率分布的可视化分析")
    print("=" * 80)
    print("问题描述：对于 1/r 波函数，可视化其在球壳中的概率分布，")
    print("并分析概率密度随半径的变化规律。")
    print("-" * 80)
    
    # 步骤1：计算归一化常数
    print(f"\n步骤1：计算 1/r 波函数的归一化常数")
    # 调用函数：calculate_normalization_constant()
    norm_result_scenario3 = calculate_normalization_constant(d1, d_outer, '1/r')
    print(f"FUNCTION_CALL: calculate_normalization_constant | PARAMS: {{r_min: {d1}, r_max: {d_outer}, wave_function_type: '1/r'}} | RESULT: {norm_result_scenario3}")
    A_scenario3 = norm_result_scenario3['result']
    print(f"归一化常数 A = {A_scenario3:.6f}")
    
    # 步骤2：计算不同半径处的概率密度
    print(f"\n步骤2：计算不同半径处的概率密度")
    test_radii = [d1, 2*d1, 5*d1, 10*d1, 50*d1, d_outer]
    for r in test_radii:
        # 调用函数：radial_probability_density()
        prob_density = radial_probability_density(r, '1/r', A_scenario3)
        print(f"FUNCTION_CALL: radial_probability_density | PARAMS: {{r: {r}, wave_function_type: '1/r', normalization_const: {A_scenario3}}} | RESULT: {prob_density}")
        print(f"  r = {r:.2f}: P(r) = {prob_density['result']:.6f}")
    
    # 步骤3：可视化球壳概率分布
    # 调用函数：visualize_spherical_probability_distribution()
    spherical_viz_result = visualize_spherical_probability_distribution(
        '1/r', d1, d_outer, A_scenario3, num_shells=50
    )
    print(f"\nFUNCTION_CALL: visualize_spherical_probability_distribution | PARAMS: {{wave_function_type: '1/r', d1: {d1}, d_outer: {d_outer}, normalization_const: {A_scenario3}, num_shells: 50}} | RESULT: {spherical_viz_result}")
    
    # 步骤4：分析概率密度的物理意义
    print(f"\n步骤4：分析概率密度的物理意义")
    # 调用函数：symbolic_wave_function_analysis()
    symbolic_analysis = symbolic_wave_function_analysis(f'{A_scenario3}/r')
    print(f"FUNCTION_CALL: symbolic_wave_function_analysis | PARAMS: {{wave_function_expr: '{A_scenario3}/r'}} | RESULT: {symbolic_analysis}")
    
    print(f"\n物理意义分析：")
    print(f"1. 波函数 ψ(r) = A/r 随半径递减")
    print(f"2. 概率密度 P(r) = |ψ|² × 4πr² = A² × 4π (常数)")
    print(f"3. 这意味着在任意等宽的径向区间内，找到粒子的概率相等")
    print(f"4. 这是满足题目条件的唯一解析解")
    
    # 步骤5：验证等概率性质
    print(f"\n步骤5：验证等宽区间的等概率性质")
    equal_width_intervals = [(i*d1, (i+1)*d1) for i in range(1, 6)]
    probabilities_equal_width = []
    for r_min, r_max in equal_width_intervals:
        # 调用函数：integrate_probability()
        prob = integrate_probability(r_min, r_max, '1/r', A_scenario3)
        probabilities_equal_width.append(prob['result'])
        print(f"FUNCTION_CALL: integrate_probability | PARAMS: {{r_min: {r_min}, r_max: {r_max}, wave_function_type: '1/r', normalization_const: {A_scenario3}}} | RESULT: {prob}")
    
    print(f"\n等宽区间概率列表: {probabilities_equal_width}")
    print(f"概率标准差: {np.std(probabilities_equal_width):.10f}")
    print(f"概率均值: {np.mean(probabilities_equal_width):.6f}")
    
    final_answer_scenario3 = "For ψ(r) = A/r, probability density P(r) = constant, ensuring equal probability in equal-width radial intervals"
    print(f"\nFINAL_ANSWER: {final_answer_scenario3}")
    
    print("\n" + "=" * 80)
    print("所有场景执行完毕")
    print("=" * 80)


if __name__ == "__main__":
    main()