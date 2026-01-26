import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve, Eq
import pandas as pd

# =============================================
# 底层核心数学计算模块说明：
# =============================================
# 本问题的核心是 **溶度积平衡（Ksp）与水解平衡（Kw, Ka/Kb）的耦合计算**
# 关键步骤：
# 1. 判断金属离子是否沉淀：基于 [M^{n+}] 和 [OH^-] 计算离子积 Q，与 Ksp 比较
# 2. 控制 pH 实现选择性沉淀：找到 Fe³⁺ 沉淀而 Mg²⁺ 不沉淀的 pH 区间
# 3. 计算开始沉淀和完全沉淀的 pH（通常定义“完全”为 [Mⁿ⁺] ≤ 1e-5 mol/kg）
#
# 核心公式：
#   [OH⁻] = sqrt(Ksp / [M²⁺])        → 对于 Mg(OH)₂
#   [OH⁻] = cbrt(Ksp / [M³⁺])        → 对于 Fe(OH)₃
#   pH = 14 + log10([OH⁻])
#
# 所需数据（25°C）：
#   Ksp[Fe(OH)₃] = 2.79 × 10⁻³⁹
#   Ksp[Mg(OH)₂] = 5.61 × 10⁻¹²
#   Kw = 1e-14

# =============================================
# 1. 数学计算函数（math func）
# =============================================

def calculate_precipitation_pH(concentration, ksp, hydroxide_stoich, complete_threshold=1e-5):
    """
    计算金属离子开始沉淀和完全沉淀的pH值
    
    基于溶度积平衡原理，计算指定金属离子在不同pH条件下的沉淀行为。
    通过比较离子积与溶度积常数，确定开始沉淀和完全沉淀的pH值。
    
    Parameters:
    -----------
    concentration : float
        金属离子浓度，单位为mol/L
    ksp : float
        金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)
    hydroxide_stoich : int
        氢氧根离子的化学计量数（如Mg(OH)₂为2，Fe(OH)₃为3）
    complete_threshold : float, optional
        完全沉淀的浓度阈值，默认为1e-5 mol/L
    
    Returns:
    --------
    tuple
        包含两个元素的元组：
        - pH_start : float
            开始沉淀的pH值
        - pH_complete : float
            完全沉淀的pH值
    """
    
    # 完全沉淀标准
    COMPLETE_THRESHOLD = complete_threshold

    # 计算 [OH⁻] 开始沉淀
    oh_start = (ksp / concentration) ** (1 / hydroxide_stoich)
    # 计算 [OH⁻] 完全沉淀
    oh_complete = (ksp / COMPLETE_THRESHOLD) ** (1 / hydroxide_stoich)

    # 转换为 pH
    pH_start = 14 + np.log10(oh_start)
    pH_complete = 14 + np.log10(oh_complete)

    return pH_start, pH_complete

def can_separate_ions(conc1, conc2, ksp1, ksp2, stoich1, stoich2, complete_threshold=1e-5):
    """
    判断两种金属离子是否可以通过pH控制实现选择性沉淀分离
    
    分析两种金属离子的沉淀行为，确定是否存在pH区间可以实现选择性沉淀。
    通过比较两种离子的开始沉淀pH值，找到分离窗口。
    
    Parameters:
    -----------
    conc1 : float
        第一种金属离子浓度，单位为mol/L
    conc2 : float
        第二种金属离子浓度，单位为mol/L
    ksp1 : float
        第一种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)
    ksp2 : float
        第二种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)
    stoich1 : int
        第一种金属氢氧化物中氢氧根离子的化学计量数
    stoich2 : int
        第二种金属氢氧化物中氢氧根离子的化学计量数
    complete_threshold : float, optional
        完全沉淀的浓度阈值，默认为1e-5 mol/L
    
    Returns:
    --------
    dict
        包含分离分析结果的字典：
        - separable : bool
            是否可以实现选择性分离
        - separation_pH_range : tuple or None
            分离pH区间，如果可分离则返回(lower_bound, upper_bound)
        - ion1 : dict
            第一种离子的沉淀pH信息
        - ion2 : dict
            第二种离子的沉淀pH信息
        - target_to_precipitate_first : str
            优先沉淀的离子标识
    """
    
    pH_start_1, pH_complete_1 = calculate_precipitation_pH(conc1, ksp1, stoich1, complete_threshold)
    pH_start_2, pH_complete_2 = calculate_precipitation_pH(conc2, ksp2, stoich2, complete_threshold)

    # 假设 ion1 是更容易沉淀的（如 Fe³⁺）
    # 我们希望：pH_start_1 < pH_start_2，这样可以在 pH ∈ [pH_start_1, pH_start_2) 沉淀 ion1 而不沉淀 ion2
    if pH_start_1 < pH_start_2:
        lower_bound = pH_start_1
        upper_bound = pH_start_2
        separable = True if lower_bound < upper_bound else False
        target = "ion1"
    else:
        lower_bound = pH_start_2
        upper_bound = pH_start_1
        separable = True if lower_bound < upper_bound else False
        target = "ion2"

    return {
        "separable": separable,
        "separation_pH_range": (lower_bound, upper_bound) if separable else None,
        "ion1": {"start": pH_start_1, "complete": pH_complete_1},
        "ion2": {"start": pH_start_2, "complete": pH_complete_2},
        "target_to_precipitate_first": target
    }

# =============================================
# 2. 可视化函数（visual func）
# =============================================

def plot_separation_window(conc1, conc2, ksp1, ksp2, stoich1, stoich2, 
                          label1="Fe³⁺", label2="Mg²⁺", title="Selective Precipitation by pH",
                          complete_threshold=1e-5, figsize=(10, 6), n_points=500):
    """
    绘制选择性沉淀分离的pH窗口可视化图表
    
    创建离子积与pH的关系图，显示两种金属离子的沉淀行为，
    高亮显示选择性分离的pH区间，帮助理解分离原理。
    
    Parameters:
    -----------
    conc1 : float
        第一种金属离子浓度，单位为mol/L
    conc2 : float
        第二种金属离子浓度，单位为mol/L
    ksp1 : float
        第一种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)
    ksp2 : float
        第二种金属氢氧化物的溶度积常数，单位为(mol/L)^(n+1)
    stoich1 : int
        第一种金属氢氧化物中氢氧根离子的化学计量数
    stoich2 : int
        第二种金属氢氧化物中氢氧根离子的化学计量数
    label1 : str, optional
        第一种离子的标签，默认为"Fe³⁺"
    label2 : str, optional
        第二种离子的标签，默认为"Mg²⁺"
    title : str, optional
        图表标题，默认为"Selective Precipitation by pH"
    complete_threshold : float, optional
        完全沉淀的浓度阈值，默认为1e-5 mol/L
    figsize : tuple, optional
        图表尺寸，默认为(10, 6)
    n_points : int, optional
        绘图点数，默认为500
    
    Returns:
    --------
    dict
        分离分析结果字典，包含separable、separation_pH_range等信息
    """
    
    result = can_separate_ions(conc1, conc2, ksp1, ksp2, stoich1, stoich2, complete_threshold)
    
    # 创建 pH 轴
    pH = np.linspace(0, 14, n_points)
    pOH = 14 - pH
    oh = 10 ** (-pOH)

    # 计算离子积 Q1 和 Q2
    Q1 = conc1 * (oh ** stoich1)
    Q2 = conc2 * (oh ** stoich2)

    # Ksp 线
    Ksp1_line = np.full_like(pH, ksp1)
    Ksp2_line = np.full_like(pH, ksp2)

    plt.figure(figsize=figsize)
    plt.plot(pH, Q1, label=f"Ion Product: {label1}", color='red', linewidth=2)
    plt.plot(pH, Q2, label=f"Ion Product: {label2}", color='blue', linewidth=2)
    plt.axhline(y=ksp1, color='red', linestyle='--', label=f'Ksp({label1}) = {ksp1:.2e}')
    plt.axhline(y=ksp2, color='blue', linestyle='--', label=f'Ksp({label2}) = {ksp2:.2e}')

    # 填充沉淀区
    plt.fill_between(pH, Q1, where=(Q1 >= ksp1), color='red', alpha=0.2, label=f"{label1} precipitates")
    plt.fill_between(pH, Q2, where=(Q2 >= ksp2), color='blue', alpha=0.2, label=f"{label2} precipitates")

    # 分离窗口高亮
    if result["separable"]:
        low, high = result["separation_pH_range"]
        plt.axvspan(low, high, color='green', alpha=0.3, label=f"Separation window: [{low:.2f}, {high:.2f}]")

    plt.yscale('log')
    plt.xlim(0, 14)
    plt.ylim(1e-40, 1e-2)
    plt.xlabel("pH")
    plt.ylabel("Ion Product [M][OH⁻]ⁿ")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return result

# =============================================
# 3. 主计算函数（coding func）——用户接口
# =============================================

def ChemicalEquilibriumCalculator_separate_metals(
    conc_fe=0.01, 
    conc_mg=0.01,
    ksp_fe=2.79e-39,    # Fe(OH)3
    ksp_mg=5.61e-12,    # Mg(OH)2
    stoich_fe=3,
    stoich_mg=2,
    visualize=True,
    complete_threshold=1e-5,
    verbose=True
):
    """
    化学平衡计算器：金属离子选择性沉淀分离分析
    
    分析Fe³⁺和Mg²⁺两种金属离子在氢氧化物沉淀中的选择性分离行为。
    通过计算不同pH条件下的沉淀行为，确定选择性分离的pH窗口。
    
    Parameters:
    -----------
    conc_fe : float, optional
        Fe³⁺离子浓度，单位为mol/L，默认为0.01
    conc_mg : float, optional
        Mg²⁺离子浓度，单位为mol/L，默认为0.01
    ksp_fe : float, optional
        Fe(OH)₃的溶度积常数，单位为(mol/L)⁴，默认为2.79e-39
    ksp_mg : float, optional
        Mg(OH)₂的溶度积常数，单位为(mol/L)³，默认为5.61e-12
    stoich_fe : int, optional
        Fe(OH)₃中氢氧根离子的化学计量数，默认为3
    stoich_mg : int, optional
        Mg(OH)₂中氢氧根离子的化学计量数，默认为2
    visualize : bool, optional
        是否生成可视化图表，默认为True
    complete_threshold : float, optional
        完全沉淀的浓度阈值，默认为1e-5 mol/L
    verbose : bool, optional
        是否显示详细输出，默认为True
    
    Returns:
    --------
    dict
        包含分离分析结果的字典：
        - separable : bool
            是否可以实现选择性分离
        - separation_pH_range : tuple or None
            分离pH区间
        - ion1 : dict
            Fe³⁺的沉淀pH信息
        - ion2 : dict
            Mg²⁺的沉淀pH信息
        - target_to_precipitate_first : str
            优先沉淀的离子标识
    """
    
    result = can_separate_ions(
        conc1=conc_fe, conc2=conc_mg,
        ksp1=ksp_fe, ksp2=ksp_mg,
        stoich1=stoich_fe, stoich2=stoich_mg,
        complete_threshold=complete_threshold
    )

    # 输出解释
    if verbose:
        print("=== 选择性沉淀分析结果 ===")
        print(f"{result['ion1']['start']:.2f} ≤ pH < {result['ion2']['start']:.2f} 时，可沉淀 Fe³⁺ 而不沉淀 Mg²⁺")
        print(f"Fe³⁺ 开始沉淀 pH: {result['ion1']['start']:.2f}")
        print(f"Fe³⁺ 完全沉淀 pH: {result['ion1']['complete']:.2f}")
        print(f"Mg²⁺ 开始沉淀 pH: {result['ion2']['start']:.2f}")
        print(f"Mg²⁺ 完全沉淀 pH: {result['ion2']['complete']:.2f}")

        if result["separable"]:
            low, high = result["separation_pH_range"]
            print(f"✅ 可以通过控制 pH 在 [{low:.2f}, {high:.2f}) 区间实现分离。")
            # print(f"在此 pH 范围内，Fe³⁺ 会沉淀，而 Mg²⁺ 仍留在溶液中。")
        else:
            print(f"❌ 无法通过 pH 控制实现完全分离。")

    # 可视化
    if visualize:
        plot_separation_window(
            conc_fe, conc_mg, ksp_fe, ksp_mg, stoich_fe, stoich_mg,
            label1="Fe³⁺", label2="Mg²⁺",
            title="Fe³⁺ and Mg²⁺ Selective Hydroxide Precipitation",
            complete_threshold=complete_threshold
        )

    return result

# =============================================
# 使用示例（可取消注释运行）
# =============================================

# result = ChemicalEquilibriumCalculator_separate_metals(
#     conc_fe=0.01,
#     conc_mg=0.01,
#     visualize=True
# )
