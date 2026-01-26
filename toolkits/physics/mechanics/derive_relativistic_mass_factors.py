import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, simplify, sqrt, lambdify

# ========================================
# 核心物理概念说明：
# 在相对论力学中，质量不再是常数。
# 当力平行于速度方向时：m_long = γ³ * m0  → “纵向质量”
# 当力垂直于速度方向时：m_trans = γ * m0 → “横向质量”
# 其中 γ = 1 / sqrt(1 - v²/c²)
# 这些概念源于早期相对论表述（现已被四维力取代），但仍具教学意义。
# ========================================

# ========================================
# 1. 数学核心函数（Math Functions）
# 负责符号推导与公式构建
# ========================================

def derive_relativistic_mass_factors():

    """
    计算相对论效应相关参数
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    float
        计算结果
    """
    
    m0, v, c = symbols('m0 v c')
    gamma = 1 / sqrt(1 - v**2/c**2)

    # 纵向质量：F_parallel / a_parallel = γ³ * m0
    m_long_expr = gamma**3 * m0

    # 横向质量：F_perp / a_perp = γ * m0
    m_trans_expr = gamma * m0

    print("【符号推导结果】")
    print(f"纵向质量表达式: {simplify(m_long_expr)}")
    print(f"横向质量表达式: {simplify(m_trans_expr)}")

    return m_long_expr, m_trans_expr

def get_mass_functions(c_val=3e8):

    """
    获取数据或属性
    
    Parameters:
    -----------
    c_val : float
        c_val参数，默认值为300000000.0
    
    Returns:
    --------
    any
        获取的数据
    """
    
    m0, v, c = symbols('m0 v c')
    gamma = 1 / sqrt(1 - v**2/c**2)
    
    m_long_sym = gamma**3 * m0
    m_trans_sym = gamma * m0

    # 转为数值函数
    m_long_func = lambdify((v, m0), m_long_sym.subs(c, c_val), 'numpy')
    m_trans_func = lambdify((v, m0), m_trans_sym.subs(c, c_val), 'numpy')

    return m_long_func, m_trans_func

# ========================================
# 2. 核心计算函数（Coding Functions）
# 提供通用接口进行计算
# ========================================

def calculate_relativistic_masses(velocities, rest_mass, c=3e8):

    """
    计算相对论效应相关参数
    
    Parameters:
    -----------
    velocities : float
        velocities参数
    rest_mass : float
        rest_mass，单位为kg
    c : float
        c参数
    
    Returns:
    --------
    float
        计算结果
    """
    
    m_long_func, m_trans_func = get_mass_functions(c)
    
    # 计算
    m_long = m_long_func(velocities, rest_mass)
    m_trans = m_trans_func(velocities, rest_mass)
    
    return {
        'velocity': velocities,
        'longitudinal_mass': m_long,
        'transverse_mass': m_trans,
        'rest_mass': rest_mass,
        'c': c
    }

# ========================================
# 3. 可视化函数（Visual Functions）
# 绘制质量随速度变化曲线
# ========================================

def plot_relativistic_masses(result_dict, show_gamma=False):

    """
    计算相对论效应相关参数
    
    Parameters:
    -----------
    result_dict : dict
        result_dict，结果字典
    show_gamma : float
        show_gamma参数
    
    Returns:
    --------
    object
        图表对象
    """
    
    v = result_dict['velocity']
    m_long = result_dict['longitudinal_mass']
    m_trans = result_dict['transverse_mass']
    m0 = result_dict['rest_mass']
    c = result_dict['c']

    # 转换为 c 的单位（更直观）
    v_norm = v / c

    plt.figure(figsize=(10, 6))
    plt.plot(v_norm, m_long, label=r'Longitudinal Mass $m_{\parallel} = \gamma^3 m_0$', lw=2)
    plt.plot(v_norm, m_trans, label=r'Transverse Mass $m_{\perp} = \gamma m_0$', lw=2)
    plt.axhline(y=m0, color='k', linestyle='--', label=r'Rest Mass $m_0$')

    if show_gamma:
        gamma = 1 / np.sqrt(1 - v_norm**2)
        plt.plot(v_norm, gamma * m0, '--', color='gray', alpha=0.7, label=r'$\gamma m_0$')
        plt.plot(v_norm, gamma**3 * m0, ':', color='gray', alpha=0.7, label=r'$\gamma^3 m_0$')

    plt.title(f'Relativistic Longitudinal and Transverse Mass (Rest Mass = {m0:.2e} kg)', fontsize=14)
    plt.xlabel(r'Velocity $v/c$', fontsize=12)
    plt.ylabel(r'Effective Mass (kg)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.xlim(0, 0.99)
    plt.ylim(m0 * 0.9, max(m_long.max(), m_trans.max()) * 1.1)
    plt.tight_layout()
    plt.show()

# ========================================
# 示例使用（可取消注释运行）
# ========================================

if __name__ == "__main__":
    # 步骤1：符号推导（数学核心）
    print("正在进行符号推导...")
    derive_relativistic_mass_factors()

    # 步骤2：数值计算
    print("开始数值计算...")
    v_array = np.linspace(0, 0.99 * 3e8, 500)  # 0 到 0.99c
    result = calculate_relativistic_masses(v_array, rest_mass=9.11e-31)  # 电子质量

    # 步骤3：可视化
    plot_relativistic_masses(result, show_gamma=True)

    # 输出部分数据验证
    print("示例数据（v = 0.9c）:")
    idx = np.argmin(np.abs(result['velocity'] - 0.9 * 3e8))
    print(f"v = {result['velocity'][idx]:.2e} m/s ({0.9:.2f}c)")
    print(f"纵向质量 = {result['longitudinal_mass'][idx]:.2e} kg")
    print(f"横向质量 = {result['transverse_mass'][idx]:.2e} kg")
