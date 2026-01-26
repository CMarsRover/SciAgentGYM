import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import R
import warnings
import csv
import pandas as pd 
# 定义法拉第常数
F = 96485.33212  # C/mol (法拉第常数)

def export_results_to_csv(result: dict, csv_path: str) -> None:
    """
    将阴极反应分析结果导出为CSV文件
    
    将analyze_cathode_reaction函数的分析结果导出为两个CSV文件：
    1. 详细反应数据文件：包含所有反应的详细信息
    2. 摘要信息文件：包含分析结果的总结信息
    
    Parameters:
    -----------
    result : dict
        阴极反应分析结果字典，包含以下键值对：
        - all_reactions : list
            所有反应的详细信息列表
        - dominant_reaction : dict
            主导反应信息
        - reaction_type : str
            反应类型
        - temperature : float
            温度(K)
        - pH : float
            溶液pH值
        - E_cathode : float
            阴极电位(V)
        - electrode : str
            电极类型
        - remaining_concentration : float
            剩余离子浓度
    csv_path : str
        输出CSV文件路径，会自动生成对应的摘要文件
    
    Returns:
    --------
    None
        直接生成CSV文件到指定路径
    """
    fieldnames = ['reaction', 'type', 'ion', 'c0', 'E_thermo', 'E_effective', 'driving_force']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in result.get('all_reactions', []):
            writer.writerow({
                'reaction': r.get('reaction'),
                'type': r.get('type'),
                'ion': r.get('ion'),
                'c0': r.get('c0'),
                'E_thermo': float(r.get('E_thermo')) if r.get('E_thermo') is not None else None,
                'E_effective': float(r.get('E_effective')) if r.get('E_effective') is not None else None,
                'driving_force': float(r.get('driving_force')) if r.get('driving_force') is not None else None,
            })
    summary_path = csv_path.replace('.csv', '_summary.csv')
    summary_fields = ['dominant_reaction', 'reaction_type', 'temperature', 'pH', 'E_cathode', 'electrode', 'remaining_concentration']
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        dom = result.get('dominant_reaction')
        writer.writerow({
            'dominant_reaction': dom.get('reaction') if dom else None,
            'reaction_type': result.get('reaction_type'),
            'temperature': result.get('temperature'),
            'pH': result.get('pH'),
            'E_cathode': result.get('E_cathode'),
            'electrode': result.get('electrode'),
            'remaining_concentration': float(result.get('remaining_concentration')) if result.get('remaining_concentration') is not None else None
        })

# ========================================
# 电化学反应分析核心计算模块
# 功能：判断阴极反应优先级、计算离子残留浓度、可视化电位-浓度关系
# 通用性设计：支持多离子体系、pH影响、过电位修正、温度变量
# ========================================

# ========================================
# 1. 数学核心函数（Math Functions）
# ========================================

def nernst_potential(E0, n, a_oxidized, a_reduced=1, T=298.15):
    """
    计算Nernst电位
    
    根据Nernst方程计算电化学反应的平衡电位，考虑反应物和产物的活度以及温度的影响。
    用于预测电化学反应在不同条件下的电位变化。
    
    Parameters:
    -----------
    E0 : float
        标准电极电位，单位为V（相对于标准氢电极SHE）
    n : int
        参与反应的电子数
    a_oxidized : float
        氧化态物质的活度（通常用浓度近似）
    a_reduced : float, optional
        还原态物质的活度，默认为1（纯固体或标准状态）
    T : float, optional
        温度，单位为K，默认为298.15K（25°C）
    
    Returns:
    --------
    float
        计算得到的Nernst电位，单位为V
    """
    return E0 - (R * T / (n * F)) * np.log(a_oxidized / a_reduced)

def hydrogen_evolution_potential(pH, T=298.15, overpotential=0.0):
    """
    计算析氢反应电位
    
    计算在不同pH值和温度下析氢反应(HER)的电位，考虑过电位的影响。
    析氢反应是电化学中重要的竞争反应，影响金属离子的电沉积过程。
    
    Parameters:
    -----------
    pH : float
        溶液的pH值，影响H+离子浓度
    T : float, optional
        温度，单位为K，默认为298.15K（25°C）
    overpotential : float, optional
        析氢反应的过电位，单位为V，默认为0.0V
        不同电极材料具有不同的析氢过电位
    
    Returns:
    --------
    float
        析氢反应的有效电位，单位为V（相对于SHE）
        考虑过电位后的实际析氢电位
    """
    E0_H = 0.0  # SHE reference
    a_H = 10**(-pH)
    E_H_thermo = E0_H - (R * T / F) * np.log(a_H)  # 简化：H+ + e- -> 1/2 H2
    return E_H_thermo - overpotential  # 实际需克服过电位

def solve_remaining_concentration(E_applied, E0, n, T=298.15, overpotential=0.0):
    """
    计算电沉积后的剩余离子浓度
    
    根据施加的电位和电化学参数，计算电沉积反应达到平衡时的剩余离子浓度。
    用于预测电沉积过程的完成程度和离子去除效率。
    
    Parameters:
    -----------
    E_applied : float
        施加的阴极电位，单位为V（相对于SHE）
    E0 : float
        标准电极电位，单位为V
    n : int
        参与反应的电子数
    T : float, optional
        温度，单位为K，默认为298.15K（25°C）
    overpotential : float, optional
        过电位，单位为V，默认为0.0V
        考虑电极材料对反应动力学的影响
    
    Returns:
    --------
    float
        剩余离子浓度，单位为mol/kg
        最小值限制为1e-20以防止数值下溢
    """
    E_effective = E_applied + overpotential  # 阴极电位更负，需加上过电位才能驱动反应
    exponent = (E0 - E_effective) * (n * F) / (R * T)
    c_remaining = np.exp(exponent)
    return max(c_remaining, 1e-20)  # 防止数值下溢

# ========================================
# 2. 主要逻辑函数（Coding Function）
# ========================================

def analyze_cathode_reaction(ions, pH, T, E_cathode, overpotentials, electrode_type):
    """
    分析阴极反应优先级和离子沉积行为
    
    综合分析多离子体系中的阴极反应，确定主导反应类型，计算各离子的沉积电位，
    预测剩余离子浓度，并考虑析氢反应的竞争影响。
    
    Parameters:
    -----------
    ions : list
        离子信息列表，每个元素为字典，包含以下键值对：
        - name : str
            离子名称，如'Cu²⁺', 'Zn²⁺'
        - E0 : float
            标准电极电位，单位为V
        - n : int
            参与反应的电子数
        - c0 : float
            初始离子浓度，单位为mol/kg
        - product : str
            还原产物，如'Cu', 'Zn'
        - overpotential : float
            该离子的过电位，单位为V
    pH : float
        溶液的pH值
    T : float
        温度，单位为°C或K（自动判断）
    E_cathode : float
        施加的阴极电位，单位为V（相对于SHE）
    overpotentials : dict
        不同电极材料的析氢过电位字典，键为电极类型，值为过电位(V)
    electrode_type : str
        电极材料类型，如'Pt', 'Cu', 'Ni'等
    
    Returns:
    --------
    dict
        分析结果字典，包含以下键值对：
        - all_reactions : list
            所有反应的详细信息列表
        - dominant_reaction : dict
            主导反应信息
        - reaction_type : str
            反应类型（'metal'或'HER'）
        - remaining_concentration : float
            剩余离子浓度（仅金属沉积时）
        - temperature : float
            温度(K)
        - pH : float
            pH值
        - E_cathode : float
            阴极电位
        - electrode : str
            电极类型
    """
    T_K = T + 273.15 if T < 100 else T  # 自动判断是否为摄氏度
    results = []

    # Step 1: 计算各金属离子的Nernst电位
    for ion in ions:
        E_ion = nernst_potential(
            E0=ion['E0'],
            n=ion['n'],
            a_oxidized=ion['c0'],
            T=T_K
        )
        # 实际沉积电位需克服过电位（阴极更负）
        E_ion_effective = E_ion - ion.get('overpotential', 0.0)
        driving_force = E_cathode - E_ion_effective
        results.append({
            'reaction': f"{ion['name']} -> {ion['product']}",
            'E_thermo': E_ion,
            'E_effective': E_ion_effective,
            'driving_force': driving_force,
            'type': 'metal',
            'ion': ion['name'],
            'c0': ion['c0']
        })

    # Step 2: 计算析氢反应（HER）电位
    eta_H = overpotentials.get(electrode_type, 0.0)
    E_H = hydrogen_evolution_potential(pH=pH, T=T_K, overpotential=eta_H)
    driving_force_H = E_cathode - E_H
    results.append({
        'reaction': "2H+ + 2e- -> H2",
        'E_thermo': 0.0 - (R*T_K/F)*np.log(10)*pH,
        'E_effective': E_H,
        'driving_force': driving_force_H,
        'type': 'HER',
        'ion': 'H+',
        'c0': None
    })
   
    

    # Step 3: 判断哪个反应优先（驱动力最大且E_cathode ≤ E_effective）
    feasible = [r for r in results if r['driving_force'] <= 0]  # 阴极电位更负才能驱动
    if not feasible:
        dominant = None
        reaction_type = "No reaction"
    else:
        # 最可能反应是E_effective最大的（最正），最容易被还原
        dominant = min(feasible, key=lambda x: x['E_effective'])  # 最正电位优先
        reaction_type = dominant['type']

    # Step 4: 若为金属沉积，计算剩余浓度
    remaining_concentration = None
    if dominant and dominant['type'] == 'metal':
        remaining_concentration = solve_remaining_concentration(
            E_applied=E_cathode,
            E0=ions[[i['name'] for i in ions].index(dominant['ion'])]['E0'],
            n=ions[[i['name'] for i in ions].index(dominant['ion'])]['n'],
            T=T_K,
            overpotential=ions[[i['name'] for i in ions].index(dominant['ion'])].get('overpotential', 0.0)
        )
    # 导出CSV
    result = {
        'all_reactions': results,
        'dominant_reaction': dominant,
        'reaction_type': reaction_type,
        'remaining_concentration': remaining_concentration,
        'temperature': T_K,
        'pH': pH,
        'E_cathode': E_cathode,
        'electrode': electrode_type
    }
    export_results_to_csv(result, 'result_multi_ions.csv')
    return result

# ========================================
# 3. 可视化函数（Visualization Function）
# ========================================

def plot_cathode_analysis(result, title="Cathode Reaction Analysis"):
    """
    可视化阴极反应分析结果
    
    生成阴极反应分析的图表，展示各反应的热力学电位、有效电位和过电位，
    标注施加电位和主导反应，并打印分析结果摘要。
    
    Parameters:
    -----------
    result : dict
        阴极反应分析结果字典，包含以下键值对：
        - all_reactions : list
            所有反应的详细信息列表
        - dominant_reaction : dict
            主导反应信息
        - temperature : float
            温度(K)
        - pH : float
            pH值
        - E_cathode : float
            阴极电位
        - electrode : str
            电极类型
        - remaining_concentration : float
            剩余离子浓度
    title : str, optional
        图表标题，默认为"Cathode Reaction Analysis"
    
    Returns:
    --------
    None
        显示图表并打印分析结果到控制台
    """
    reactions = result['all_reactions']
    names = [r['reaction'] for r in reactions]
    E_effective = [r['E_effective'] for r in reactions]
    E_thermo = [r['E_thermo'] for r in reactions]
    colors = ['tab:blue' if r['type'] == 'metal' else 'tab:red' for r in reactions]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(reactions))

    # 绘制热力学电位与有效电位
    ax.barh(y_pos, E_thermo, color=colors, alpha=0.6, label='Thermodynamic Potential')
    ax.barh(y_pos, [e - t for e, t in zip(E_effective, E_thermo)], left=E_thermo,
            color='k', height=0.3, alpha=0.8, label='Overpotential Shift')

    # 标注施加电位
    E_cathode = result['E_cathode']
    ax.axvline(E_cathode, color='purple', linestyle='--', linewidth=2, label=f'Applied E = {E_cathode:.2f} V')

    # 标注优先反应
    if result['dominant_reaction']:
        dom_idx = reactions.index(result['dominant_reaction'])
        ax.axhline(dom_idx, color='green', linewidth=2, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Electrode Potential (V vs. SHE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印结果
    print(f"Temperature: {result['temperature']:.2f} K, pH: {result['pH']}")
    print(f"Applied Cathode Potential: {result['E_cathode']:.3f} V vs. SHE")
    print(f"Electrode: {result['electrode']}")
    if result['dominant_reaction']:
        print(f"Dominant Reaction: {result['dominant_reaction']['reaction']}")
        if result['remaining_concentration'] is not None:
            print(f"Remaining [Cu²⁺]: {result['remaining_concentration']:.3e} mol/kg")
    else:
        print("No spontaneous reaction at this potential.")

# ========================================
# 4. 示例：解决原始问题
# ========================================

if __name__ == "__main__":
    # 已知条件
    T = 2  # °C
    pH = 4.0
    E_cathode = -0.30  # V vs. SHE
    overpotentials = {'Pt': 0.1, 'Cu': 0.2, 'Ni': 0.08, 'Ag': 0.03, 'Co': 0.09, 'Pb': 0.07}  # H2 overpotential on Pt and Cu
    electrode_type = 'Pt'

    # 离子信息（标准电极电位 vs. SHE）
    ions = [
        {
            'name': 'Cu²⁺',
            'E0': 0.34,      # Cu²⁺/Cu
            'n': 2,
            'c0': 1.0,       # mol/kg
            'product': 'Cu',
            'overpotential': 0.0  # 假设Cu沉积在Pt上过电位小，忽略
        },
        {
            'name': 'Zn²⁺',
            'E0': -0.76,     # Zn²⁺/Zn
            'n': 2,
            'c0': 1.0,
            'product': 'Zn',
            'overpotential': 0.0
        },
          {'name': 'Ni²⁺', 'E0': -0.25, 'n': 2, 'c0': 0.2, 'product': 'Ni', 'overpotential': 0.08},{'name': 'Fe³⁺', 'E0': 0.77,  'n': 3, 'c0': 0.05, 'product': 'Fe', 'overpotential': 0.12},
    {'name': 'Ag⁺',  'E0': 0.80,  'n': 1, 'c0': 0.02, 'product': 'Ag', 'overpotential': 0.03},
    {'name': 'Co²⁺', 'E0': -0.28, 'n': 2, 'c0': 0.1,  'product': 'Co', 'overpotential': 0.09},
    {'name': 'Pb²⁺', 'E0': -0.13, 'n': 2, 'c0': 0.3,  'product': 'Pb', 'overpotential': 0.07}
    ]

    # 执行分析
    result = analyze_cathode_reaction(
        ions=ions,
        pH=pH,
        T=T,
        E_cathode=E_cathode,
        overpotentials=overpotentials,
        electrode_type=electrode_type
    )
    print(result)

    
    
    # 可视化与输出
    plot_cathode_analysis(result, title="Cathode Reaction at -0.30 V (Pt electrode)")
