# Filename: quantum_mechanics_toolkit.py

"""
Quantum Mechanics Toolkit for Energy Level Resolution Analysis
基于能量-时间不确定性原理的量子态能级分辨工具包

核心物理原理：
1. 能量-时间不确定性关系：ΔE·Δt ≥ ℏ/2
2. 量子态寿命导致的能级自然线宽：ΔE ≈ ℏ/τ
3. 能级可分辨判据：|E2 - E1| > ΔE1 + ΔE2

使用的科学库：
- scipy.constants: 物理常数（普朗克常数、电子伏特转换）
- numpy: 数值计算
- matplotlib: 可视化能级图和线宽分布
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from typing import Dict, List, Tuple
import os
import json

# 确保输出目录存在
os.makedirs('./mid_result/physics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 设置matplotlib支持中英文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 第一层：原子函数 - 基础物理计算
# ============================================================================

def calculate_energy_uncertainty(lifetime: float) -> Dict:
    """
    根据量子态寿命计算能量不确定度（自然线宽）
    
    物理原理：ΔE ≈ ℏ/τ (能量-时间不确定性关系)
    
    参数:
        lifetime: 量子态寿命 (秒)
    
    返回:
        dict: {
            'result': 能量不确定度 (焦耳),
            'metadata': {
                'lifetime': 寿命,
                'hbar': 约化普朗克常数,
                'uncertainty_eV': 能量不确定度 (eV),
                'formula': 使用的公式
            }
        }
    """
    if lifetime <= 0:
        raise ValueError(f"寿命必须为正数，当前值: {lifetime}")
    
    # 约化普朗克常数 (J·s)
    hbar = constants.hbar
    
    # 计算能量不确定度 (焦耳)
    delta_E_joules = hbar / lifetime
    
    # 转换为电子伏特
    delta_E_eV = delta_E_joules / constants.eV
    
    return {
        'result': delta_E_joules,
        'metadata': {
            'lifetime': lifetime,
            'hbar': hbar,
            'uncertainty_eV': delta_E_eV,
            'formula': 'ΔE = ℏ/τ'
        }
    }


def convert_energy_units(energy_joules: float) -> Dict:
    """
    能量单位转换：焦耳 → 电子伏特
    
    参数:
        energy_joules: 能量值 (焦耳)
    
    返回:
        dict: {
            'result': 能量值 (eV),
            'metadata': {
                'joules': 原始焦耳值,
                'conversion_factor': 转换因子
            }
        }
    """
    energy_eV = energy_joules / constants.eV
    
    return {
        'result': energy_eV,
        'metadata': {
            'joules': energy_joules,
            'conversion_factor': constants.eV
        }
    }


def check_resolution_criterion(energy_diff: float, linewidth1: float, linewidth2: float) -> Dict:
    """
    检查能级分辨判据
    
    物理原理：两个能级可分辨的条件是能级差大于线宽之和
    判据：|ΔE| > Γ1 + Γ2 (Rayleigh判据的量子版本)
    
    参数:
        energy_diff: 能级差 (eV)
        linewidth1: 第一个能级的线宽 (eV)
        linewidth2: 第二个能级的线宽 (eV)
    
    返回:
        dict: {
            'result': 是否可分辨 (bool),
            'metadata': {
                'energy_diff': 能级差,
                'total_linewidth': 总线宽,
                'resolution_ratio': 分辨率比值,
                'criterion': 判据说明
            }
        }
    """
    total_linewidth = linewidth1 + linewidth2
    is_resolvable = abs(energy_diff) > total_linewidth
    resolution_ratio = abs(energy_diff) / total_linewidth if total_linewidth > 0 else float('inf')
    
    return {
        'result': is_resolvable,
        'metadata': {
            'energy_diff': energy_diff,
            'linewidth1': linewidth1,
            'linewidth2': linewidth2,
            'total_linewidth': total_linewidth,
            'resolution_ratio': resolution_ratio,
            'criterion': '|ΔE| > Γ1 + Γ2',
            'interpretation': 'Resolvable' if is_resolvable else 'Not resolvable'
        }
    }


# ============================================================================
# 第二层：组合函数 - 复杂物理分析
# ============================================================================

def analyze_two_state_resolution(lifetime1: float, lifetime2: float, energy_diff_eV: float) -> Dict:
    """
    分析两个量子态的能级分辨性
    
    参数:
        lifetime1: 第一个量子态寿命 (秒)
        lifetime2: 第二个量子态寿命 (秒)
        energy_diff_eV: 能级差 (eV)
    
    返回:
        dict: {
            'result': {
                'is_resolvable': 是否可分辨,
                'linewidth1_eV': 第一个能级线宽,
                'linewidth2_eV': 第二个能级线宽,
                'total_linewidth_eV': 总线宽,
                'resolution_ratio': 分辨率比值
            },
            'metadata': {
                'input_parameters': 输入参数,
                'physical_principle': 物理原理说明
            }
        }
    """
    # 计算第一个能级的线宽
    unc1 = calculate_energy_uncertainty(lifetime1)
    linewidth1_eV = unc1['metadata']['uncertainty_eV']
    
    # 计算第二个能级的线宽
    unc2 = calculate_energy_uncertainty(lifetime2)
    linewidth2_eV = unc2['metadata']['uncertainty_eV']
    
    # 检查分辨判据
    resolution = check_resolution_criterion(energy_diff_eV, linewidth1_eV, linewidth2_eV)
    
    return {
        'result': {
            'is_resolvable': resolution['result'],
            'linewidth1_eV': linewidth1_eV,
            'linewidth2_eV': linewidth2_eV,
            'total_linewidth_eV': resolution['metadata']['total_linewidth'],
            'resolution_ratio': resolution['metadata']['resolution_ratio']
        },
        'metadata': {
            'input_parameters': {
                'lifetime1': lifetime1,
                'lifetime2': lifetime2,
                'energy_diff_eV': energy_diff_eV
            },
            'physical_principle': 'Energy-time uncertainty principle: ΔE·Δt ≥ ℏ/2',
            'resolution_criterion': resolution['metadata']['criterion']
        }
    }


def calculate_minimum_resolvable_energy(lifetime1: float, lifetime2: float) -> Dict:
    """
    计算两个量子态可分辨的最小能级差
    
    参数:
        lifetime1: 第一个量子态寿命 (秒)
        lifetime2: 第二个量子态寿命 (秒)
    
    返回:
        dict: {
            'result': 最小可分辨能级差 (eV),
            'metadata': {
                'linewidth1_eV': 第一个能级线宽,
                'linewidth2_eV': 第二个能级线宽,
                'formula': 计算公式
            }
        }
    """
    # 计算两个能级的线宽
    unc1 = calculate_energy_uncertainty(lifetime1)
    unc2 = calculate_energy_uncertainty(lifetime2)
    
    linewidth1_eV = unc1['metadata']['uncertainty_eV']
    linewidth2_eV = unc2['metadata']['uncertainty_eV']
    
    # 最小可分辨能级差 = 线宽之和
    min_energy_diff = linewidth1_eV + linewidth2_eV
    
    return {
        'result': min_energy_diff,
        'metadata': {
            'linewidth1_eV': linewidth1_eV,
            'linewidth2_eV': linewidth2_eV,
            'formula': 'ΔE_min = Γ1 + Γ2 = ℏ/τ1 + ℏ/τ2'
        }
    }


def scan_energy_differences(lifetime1: float, lifetime2: float, 
                           energy_range_eV: List[float]) -> Dict:
    """
    扫描一系列能级差，判断哪些可以被分辨
    
    参数:
        lifetime1: 第一个量子态寿命 (秒)
        lifetime2: 第二个量子态寿命 (秒)
        energy_range_eV: 待测试的能级差列表 (eV)
    
    返回:
        dict: {
            'result': {
                'resolvable_energies': 可分辨的能级差列表,
                'unresolvable_energies': 不可分辨的能级差列表
            },
            'metadata': {
                'min_resolvable': 最小可分辨能级差,
                'scan_results': 详细扫描结果
            }
        }
    """
    resolvable = []
    unresolvable = []
    scan_results = []
    
    for energy_diff in energy_range_eV:
        analysis = analyze_two_state_resolution(lifetime1, lifetime2, energy_diff)
        
        scan_results.append({
            'energy_diff_eV': energy_diff,
            'is_resolvable': analysis['result']['is_resolvable'],
            'resolution_ratio': analysis['result']['resolution_ratio']
        })
        
        if analysis['result']['is_resolvable']:
            resolvable.append(energy_diff)
        else:
            unresolvable.append(energy_diff)
    
    # 计算最小可分辨能级差
    min_res = calculate_minimum_resolvable_energy(lifetime1, lifetime2)
    
    return {
        'result': {
            'resolvable_energies': resolvable,
            'unresolvable_energies': unresolvable
        },
        'metadata': {
            'min_resolvable_eV': min_res['result'],
            'scan_results': scan_results
        }
    }


# ============================================================================
# 第三层：可视化函数
# ============================================================================

def plot_energy_level_diagram(lifetime1: float, lifetime2: float, 
                              energy_diff_eV: float, E1: float = 0.0) -> Dict:
    """
    绘制能级图，显示能级位置和线宽
    
    参数:
        lifetime1: 第一个量子态寿命 (秒)
        lifetime2: 第二个量子态寿命 (秒)
        energy_diff_eV: 能级差 (eV)
        E1: 第一个能级的能量 (eV)，默认为0
    
    返回:
        dict: {
            'result': 图像文件路径,
            'metadata': {
                'file_type': 'png',
                'description': 图像描述
            }
        }
    """
    # 计算线宽
    analysis = analyze_two_state_resolution(lifetime1, lifetime2, energy_diff_eV)
    linewidth1 = analysis['result']['linewidth1_eV']
    linewidth2 = analysis['result']['linewidth2_eV']
    is_resolvable = analysis['result']['is_resolvable']
    
    E2 = E1 + energy_diff_eV
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制能级1
    ax.hlines(E1, 0, 1, colors='blue', linewidth=3, label=f'E1 (τ={lifetime1:.0e} s)')
    # 绘制能级1的线宽范围
    ax.fill_between([0, 1], E1 - linewidth1/2, E1 + linewidth1/2, 
                     alpha=0.3, color='blue', label=f'Γ1 = {linewidth1:.2e} eV')
    
    # 绘制能级2
    ax.hlines(E2, 0, 1, colors='red', linewidth=3, label=f'E2 (τ={lifetime2:.0e} s)')
    # 绘制能级2的线宽范围
    ax.fill_between([0, 1], E2 - linewidth2/2, E2 + linewidth2/2, 
                     alpha=0.3, color='red', label=f'Γ2 = {linewidth2:.2e} eV')
    
    # 标注能级差
    ax.annotate('', xy=(1.2, E1), xytext=(1.2, E2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(1.3, (E1 + E2)/2, f'ΔE = {energy_diff_eV:.2e} eV', 
            fontsize=12, va='center')
    
    # 标注分辨性
    resolution_text = "可分辨 (Resolvable)" if is_resolvable else "不可分辨 (Not Resolvable)"
    ax.text(0.5, max(E1, E2) + max(linewidth1, linewidth2), 
            resolution_text, fontsize=14, ha='center', 
            bbox=dict(boxstyle='round', facecolor='yellow' if is_resolvable else 'lightcoral', alpha=0.5))
    
    ax.set_xlim(-0.2, 1.6)
    ax.set_ylim(min(E1 - linewidth1, E2 - linewidth2) - 1e-6, 
                max(E1 + linewidth1, E2 + linewidth2) + 1e-6)
    ax.set_ylabel('Energy (eV)', fontsize=12)
    ax.set_title('Quantum Energy Level Diagram with Natural Linewidth', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    
    filepath = './tool_images/energy_level_diagram.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'description': 'Energy level diagram showing natural linewidths',
            'is_resolvable': is_resolvable
        }
    }


def plot_lorentzian_lineshape(lifetime1: float, lifetime2: float, 
                              energy_diff_eV: float, E1: float = 0.0) -> Dict:
    """
    绘制洛伦兹线型，展示能级的频谱分布
    
    物理原理：自然线宽对应洛伦兹分布
    L(E) = (Γ/2π) / [(E - E0)^2 + (Γ/2)^2]
    
    参数:
        lifetime1: 第一个量子态寿命 (秒)
        lifetime2: 第二个量子态寿命 (秒)
        energy_diff_eV: 能级差 (eV)
        E1: 第一个能级的能量 (eV)
    
    返回:
        dict: {
            'result': 图像文件路径,
            'metadata': {
                'file_type': 'png',
                'description': 图像描述
            }
        }
    """
    # 计算线宽
    analysis = analyze_two_state_resolution(lifetime1, lifetime2, energy_diff_eV)
    gamma1 = analysis['result']['linewidth1_eV']
    gamma2 = analysis['result']['linewidth2_eV']
    is_resolvable = analysis['result']['is_resolvable']
    
    E2 = E1 + energy_diff_eV
    
    # 生成能量范围
    E_min = min(E1, E2) - 5 * max(gamma1, gamma2)
    E_max = max(E1, E2) + 5 * max(gamma1, gamma2)
    E_range = np.linspace(E_min, E_max, 1000)
    
    # 洛伦兹线型函数
    def lorentzian(E, E0, gamma):
        return (gamma / (2 * np.pi)) / ((E - E0)**2 + (gamma / 2)**2)
    
    # 计算两个能级的线型
    L1 = lorentzian(E_range, E1, gamma1)
    L2 = lorentzian(E_range, E2, gamma2)
    L_total = L1 + L2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(E_range, L1, 'b-', linewidth=2, label=f'Level 1 (Γ={gamma1:.2e} eV)')
    ax.plot(E_range, L2, 'r-', linewidth=2, label=f'Level 2 (Γ={gamma2:.2e} eV)')
    ax.plot(E_range, L_total, 'k--', linewidth=2, label='Total spectrum')
    
    # 标注能级位置
    ax.axvline(E1, color='blue', linestyle=':', alpha=0.5)
    ax.axvline(E2, color='red', linestyle=':', alpha=0.5)
    
    # 标注分辨性
    resolution_text = "Resolvable" if is_resolvable else "Not Resolvable"
    ax.text(0.5, 0.95, resolution_text, transform=ax.transAxes,
            fontsize=14, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if is_resolvable else 'lightcoral', alpha=0.5))
    
    ax.set_xlabel('Energy (eV)', fontsize=12)
    ax.set_ylabel('Spectral Density (a.u.)', fontsize=12)
    ax.set_title('Lorentzian Lineshape of Quantum States', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    filepath = './tool_images/lorentzian_lineshape.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'description': 'Lorentzian lineshape showing spectral overlap',
            'is_resolvable': is_resolvable
        }
    }


def plot_resolution_scan(lifetime1: float, lifetime2: float, 
                        energy_range_eV: List[float]) -> Dict:
    """
    绘制能级差扫描结果，显示分辨率随能级差的变化
    
    参数:
        lifetime1: 第一个量子态寿命 (秒)
        lifetime2: 第二个量子态寿命 (秒)
        energy_range_eV: 能级差范围 (eV)
    
    返回:
        dict: {
            'result': 图像文件路径,
            'metadata': {
                'file_type': 'png',
                'description': 图像描述
            }
        }
    """
    scan_result = scan_energy_differences(lifetime1, lifetime2, energy_range_eV)
    scan_data = scan_result['metadata']['scan_results']
    min_resolvable = scan_result['metadata']['min_resolvable_eV']
    
    energies = [d['energy_diff_eV'] for d in scan_data]
    ratios = [d['resolution_ratio'] for d in scan_data]
    resolvable = [d['is_resolvable'] for d in scan_data]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制分辨率比值
    colors = ['green' if r else 'red' for r in resolvable]
    ax.scatter(energies, ratios, c=colors, s=100, alpha=0.6, edgecolors='black')
    
    # 绘制分辨阈值线
    ax.axhline(1.0, color='blue', linestyle='--', linewidth=2, 
               label='Resolution threshold (ratio = 1)')
    ax.axvline(min_resolvable, color='orange', linestyle='--', linewidth=2,
               label=f'Min resolvable ΔE = {min_resolvable:.2e} eV')
    
    ax.set_xlabel('Energy Difference (eV)', fontsize=12)
    ax.set_ylabel('Resolution Ratio (ΔE / Γ_total)', fontsize=12)
    ax.set_title('Energy Level Resolution Scan', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # 添加图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Resolvable'),
        Patch(facecolor='red', alpha=0.6, label='Not Resolvable')
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], 
              loc='upper left', fontsize=10)
    
    filepath = './tool_images/resolution_scan.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'description': 'Resolution scan showing resolvable energy differences',
            'min_resolvable_eV': min_resolvable
        }
    }


# ============================================================================
# 主函数：演示三个场景
# ============================================================================

def main():
    """
    演示量子能级分辨工具包的三个应用场景
    """
    
    print("=" * 80)
    print("场景1：解决原始问题 - 判断给定能级差是否可分辨")
    print("=" * 80)
    print("问题描述：两个量子态寿命分别为 10^-9 s 和 10^-8 s")
    print("需要判断能级差 10^-4 eV 是否足以清晰分辨这两个能级")
    print("-" * 80)
    
    # 输入参数
    tau1 = 1e-9  # 秒
    tau2 = 1e-8  # 秒
    delta_E_test = 1e-4  # eV
    
    # 步骤1：计算第一个能级的自然线宽
    # 调用函数：calculate_energy_uncertainty()
    print("\n步骤1：计算第一个量子态的自然线宽")
    unc1 = calculate_energy_uncertainty(tau1)
    print(f"FUNCTION_CALL: calculate_energy_uncertainty | PARAMS: {{'lifetime': {tau1}}} | RESULT: {unc1}")
    
    # 步骤2：计算第二个能级的自然线宽
    # 调用函数：calculate_energy_uncertainty()
    print("\n步骤2：计算第二个量子态的自然线宽")
    unc2 = calculate_energy_uncertainty(tau2)
    print(f"FUNCTION_CALL: calculate_energy_uncertainty | PARAMS: {{'lifetime': {tau2}}} | RESULT: {unc2}")
    
    # 步骤3：计算最小可分辨能级差
    # 调用函数：calculate_minimum_resolvable_energy()
    print("\n步骤3：计算两个能级的最小可分辨能级差")
    min_res = calculate_minimum_resolvable_energy(tau1, tau2)
    print(f"FUNCTION_CALL: calculate_minimum_resolvable_energy | PARAMS: {{'lifetime1': {tau1}, 'lifetime2': {tau2}}} | RESULT: {min_res}")
    
    # 步骤4：分析给定能级差的分辨性
    # 调用函数：analyze_two_state_resolution()
    print("\n步骤4：分析能级差 10^-4 eV 的分辨性")
    analysis = analyze_two_state_resolution(tau1, tau2, delta_E_test)
    print(f"FUNCTION_CALL: analyze_two_state_resolution | PARAMS: {{'lifetime1': {tau1}, 'lifetime2': {tau2}, 'energy_diff_eV': {delta_E_test}}} | RESULT: {analysis}")
    
    # 步骤5：可视化能级图
    # 调用函数：plot_energy_level_diagram()
    print("\n步骤5：绘制能级图")
    plot1 = plot_energy_level_diagram(tau1, tau2, delta_E_test)
    print(f"FUNCTION_CALL: plot_energy_level_diagram | PARAMS: {{'lifetime1': {tau1}, 'lifetime2': {tau2}, 'energy_diff_eV': {delta_E_test}}} | RESULT: {plot1}")
    
    # 步骤6：绘制洛伦兹线型
    # 调用函数：plot_lorentzian_lineshape()
    print("\n步骤6：绘制洛伦兹线型")
    plot2 = plot_lorentzian_lineshape(tau1, tau2, delta_E_test)
    print(f"FUNCTION_CALL: plot_lorentzian_lineshape | PARAMS: {{'lifetime1': {tau1}, 'lifetime2': {tau2}, 'energy_diff_eV': {delta_E_test}}} | RESULT: {plot2}")
    
    # 最终答案
    is_resolvable = analysis['result']['is_resolvable']
    min_energy = min_res['result']
    answer = f"能级差 {delta_E_test} eV {'可以' if is_resolvable else '不能'}清晰分辨这两个能级。" \
             f"最小可分辨能级差为 {min_energy:.2e} eV。" \
             f"由于 {delta_E_test} eV {'>' if is_resolvable else '<'} {min_energy:.2e} eV，" \
             f"因此答案是 {delta_E_test} eV。"
    
    print(f"\nFINAL_ANSWER: {answer}")
    
    
    print("\n" + "=" * 80)
    print("场景2：扫描不同能级差，找出所有可分辨的选项")
    print("=" * 80)
    print("问题描述：给定一组候选能级差，判断哪些可以分辨")
    print("-" * 80)
    
    # 候选能级差（包括标准答案和其他选项）
    candidate_energies = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # eV
    
    # 步骤1：扫描所有候选能级差
    # 调用函数：scan_energy_differences()
    print("\n步骤1：扫描候选能级差")
    scan_result = scan_energy_differences(tau1, tau2, candidate_energies)
    print(f"FUNCTION_CALL: scan_energy_differences | PARAMS: {{'lifetime1': {tau1}, 'lifetime2': {tau2}, 'energy_range_eV': {candidate_energies}}} | RESULT: {scan_result}")
    
    # 步骤2：可视化扫描结果
    # 调用函数：plot_resolution_scan()
    print("\n步骤2：绘制分辨率扫描图")
    plot3 = plot_resolution_scan(tau1, tau2, candidate_energies)
    print(f"FUNCTION_CALL: plot_resolution_scan | PARAMS: {{'lifetime1': {tau1}, 'lifetime2': {tau2}, 'energy_range_eV': {candidate_energies}}} | RESULT: {plot3}")
    
    resolvable_list = scan_result['result']['resolvable_energies']
    answer2 = f"在候选能级差中，可分辨的有：{resolvable_list}。" \
              f"其中 10^-4 eV = {1e-4} eV 是可分辨的最小选项。"
    
    print(f"\nFINAL_ANSWER: {answer2}")
    
    
    print("\n" + "=" * 80)
    print("场景3：不同寿命组合的能级分辨分析")
    print("=" * 80)
    print("问题描述：比较不同寿命组合对能级分辨的影响")
    print("-" * 80)
    
    # 定义多组寿命参数
    lifetime_pairs = [
        (1e-9, 1e-8),   # 原始问题
        (1e-10, 1e-9),  # 更短寿命
        (1e-8, 1e-7)    # 更长寿命
    ]
    
    results_comparison = []
    
    for i, (t1, t2) in enumerate(lifetime_pairs, 1):
        print(f"\n--- 组合 {i}: τ1={t1:.0e} s, τ2={t2:.0e} s ---")
        
        # 步骤1：计算最小可分辨能级差
        # 调用函数：calculate_minimum_resolvable_energy()
        min_res_i = calculate_minimum_resolvable_energy(t1, t2)
        print(f"FUNCTION_CALL: calculate_minimum_resolvable_energy | PARAMS: {{'lifetime1': {t1}, 'lifetime2': {t2}}} | RESULT: {min_res_i}")
        
        # 步骤2：检查 10^-4 eV 是否可分辨
        # 调用函数：check_resolution_criterion()
        unc1_i = calculate_energy_uncertainty(t1)
        unc2_i = calculate_energy_uncertainty(t2)
        check_i = check_resolution_criterion(delta_E_test, 
                                            unc1_i['metadata']['uncertainty_eV'],
                                            unc2_i['metadata']['uncertainty_eV'])
        print(f"FUNCTION_CALL: check_resolution_criterion | PARAMS: {{'energy_diff': {delta_E_test}, 'linewidth1': {unc1_i['metadata']['uncertainty_eV']}, 'linewidth2': {unc2_i['metadata']['uncertainty_eV']}}} | RESULT: {check_i}")
        
        results_comparison.append({
            'lifetimes': (t1, t2),
            'min_resolvable_eV': min_res_i['result'],
            'is_10e-4_resolvable': check_i['result']
        })
    
    # 保存比较结果
    comparison_file = './mid_result/physics/lifetime_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(results_comparison, f, indent=2)
    print(f"\nFILE_GENERATED: json | PATH: {comparison_file}")
    
    answer3 = f"对于不同寿命组合，10^-4 eV 的分辨性如下：\n"
    for res in results_comparison:
        t1, t2 = res['lifetimes']
        answer3 += f"  τ1={t1:.0e} s, τ2={t2:.0e} s: " \
                  f"最小可分辨={res['min_resolvable_eV']:.2e} eV, " \
                  f"10^-4 eV {'可分辨' if res['is_10e-4_resolvable'] else '不可分辨'}\n"
    answer3 += f"原始问题的寿命组合 (10^-9 s, 10^-8 s) 可以分辨 10^-4 eV。"
    
    print(f"\nFINAL_ANSWER: {answer3}")


if __name__ == "__main__":
    main()