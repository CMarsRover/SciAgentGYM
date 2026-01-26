
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Physical constants in eV
RYDBERG_E = 13.605693122994  # Rydberg constant in eV (≈ 13.6 eV)

def math_ionization_energy_relations(I1: float, Z: int = 2) -> Dict[str, float]:
    """
    计算电离能相关参数
    
    基于第一电离能计算氢原子类系统的各种量子化学参数，包括第二电离能、
    总能量、电子间排斥能、有效核电荷和屏蔽常数。适用于氢原子类离子体系分析。
    
    Parameters:
    -----------
    I1 : float
        第一电离能，单位为eV。从基态原子中移去一个电子所需的能量
    Z : int, optional
        原子序数，默认为2。核电荷数，决定氢原子类系统的性质
    
    Returns:
    --------
    Dict[str, float]
        包含以下键值对的计算结果字典：
        - I1 : float
            第一电离能，单位为eV
        - I2 : float
            第二电离能，单位为eV
        - E_total : float
            中性原子的总基态能量，单位为eV
        - V_ee : float
            电子间排斥能，单位为eV
        - Z_eff : float
            有效核电荷
        - sigma : float
            屏蔽常数
        - Z : int
            原子序数
        - Rydberg : float
            里德伯常数，单位为eV
    """
    
    # (1) Second ionization energy: He⁺ → He²⁺ + e⁻
    # This is hydrogen-like system with Z=2, n=1
    I2 = RYDBERG_E * (Z ** 2)  # Exact value for hydrogen-like ion
    
    # (2) Total energy of neutral He atom
    # I1 = E(He⁺) - E(He)  => E(He) = E(He⁺) - I1
    # E(He⁺) = -I2 (since He⁺ is hydrogen-like, ground state energy = -I2)
    E_He_plus = -I2
    E_total = E_He_plus - I1  # Ground state energy of neutral He
    
    # (3) Electron-electron repulsion energy
    # In hydrogen-like approximation (no e⁻-e⁻ repulsion), energy would be:
    E_hydrogen_like = 2 * (-RYDBERG_E * Z**2)  # Both electrons feel full Z
    V_ee = E_total - E_hydrogen_like  # Repulsion raises energy (less negative)
    
    # (4) Effective nuclear charge Z_eff
    # Using energy of one electron: E ≈ -RYDBERG_E * Z_eff^2
    # Total energy ≈ 2 * (-RYDBERG_E * Z_eff^2) + V_ee (approx)
    # But simpler: from first ionization: I1 ≈ RYDBERG_E * (Z_eff^2 - Z^2/4) ? 
    # Instead, use variational principle insight:
    # From perturbation theory: E ≈ -2*RYDBERG_E*(Z - 5/8)^2 for He
    # But here we solve for Z_eff from average energy per electron
    # I1 = energy to remove one electron = energy of He⁺ - energy of He
    # But I1 also ≈ RYDBERG_E * Z_eff^2  (as if the electron is in Z_eff potential)
    Z_eff = np.sqrt(I1 / RYDBERG_E)
    
    # (5) Shielding constant σ
    σ = Z - Z_eff
    
    return {
        'I1': I1,
        'I2': I2,
        'E_total': E_total,
        'V_ee': V_ee,
        'Z_eff': Z_eff,
        'sigma': σ,
        'Z': Z,
        'Rydberg': RYDBERG_E
    }

def context_ionization_energy_relations(results: Dict[str, float], Z: int = 2, verbose: bool = True, precision: int = 4) -> Dict[str, float]:
    """
    显示电离能分析结果
    
    格式化输出电离能分析的计算结果，包括第一电离能、第二电离能、总能量、
    电子间排斥能、有效核电荷和屏蔽常数等量子化学参数。
    
    Parameters:
    -----------
    results : Dict[str, float]
        电离能分析结果字典，包含I1、I2、E_total、V_ee、Z_eff、sigma等参数
    Z : int, optional
        原子序数，默认为2。用于显示分析的元素信息
    verbose : bool, optional
        是否显示详细输出，默认为True。控制是否打印分析结果到控制台
    precision : int, optional
        数值显示精度，默认为4位小数。控制输出数值的小数位数
    
    Returns:
    --------
    Dict[str, float]
        返回输入的结果字典，用于链式调用或进一步处理
    """
    
    if verbose:
        print(f"Quantum Chemistry Analysis for Element (Z={Z})")
        print(f"First Ionization Energy (I₁)     = {results['I1']:.{precision}f} eV")
        print(f"Second Ionization Energy (I₂)    = {results['I2']:.{precision}f} eV")
        print(f"Total Ground State Energy (E)    = {results['E_total']:.{precision}f} eV")
        print(f"e⁻–e⁻ Repulsion Energy (V_ee)    = {results['V_ee']:.{precision}f} eV")
        print(f"Effective Nuclear Charge (Z_eff) = {results['Z_eff']:.{precision}f}")
        print(f"Shielding Constant (σ)           = {results['sigma']:.{precision}f}")
    

def visual_func(results: Dict[str, float], save_path: str = None, figsize: Tuple[int, int] = (14, 6), dpi: int = 150):
    """
    生成电离能分析的可视化图表
    
    创建包含两个子图的综合分析图表：1. 能级图显示原子的能量状态和电离能；
    2. 屏蔽效应示意图展示核电荷屏蔽和有效核电荷的概念。
    
    Parameters:
    -----------
    results : Dict[str, float]
        电离能分析结果字典，包含I1、I2、E_total、Z、Z_eff、sigma等参数
    save_path : str, optional
        图表保存路径，默认为None。如果提供路径，图表将保存为文件
    figsize : Tuple[int, int], optional
        图表尺寸，默认为(14, 6)。控制图表的宽度和高度
    dpi : int, optional
        图像分辨率，默认为150。控制保存图像的质量
    
    Returns:
    --------
    None
        显示图表，如果提供save_path则同时保存文件
    """
    
    I1 = results['I1']
    I2 = results['I2']
    E_total = results['E_total']
    Z = results['Z']
    Z_eff = results['Z_eff']
    sigma = results['sigma']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Energy Level Diagram
    energies = [E_total, -I2, 0]
    labels = [f'He (E = {E_total:.2f} eV)', f'He⁺ (E = {-I2:.2f} eV)', 'He²⁺ + 2e⁻']
    y_pos = np.arange(len(energies))

    bars = ax1.barh(y_pos, energies, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Energy (eV)')
    ax1.set_title(f'Energy Levels of Helium Atom (Z={Z})', fontsize=14)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Annotate ionization energies
    ax1.annotate(f'I₁ = {I1:.2f} eV', xy=(E_total/2, 0), xytext=(E_total/2, 0.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                 fontsize=12, color='red', ha='center')
    ax1.annotate(f'I₂ = {I2:.2f} eV', xy=(-I2/2, 1), xytext=(-I2/2, 1.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
                 fontsize=12, color='red', ha='center')

    # Plot 2: Nuclear Shielding Visualization
    circle_outer = plt.Circle((0, 0), 1.5, color='lightblue', alpha=0.5, label='Electron Cloud')
    circle_inner = plt.Circle((0, 0), 0.5, color='red', alpha=0.7)

    ax2.add_patch(circle_outer)
    ax2.add_patch(circle_inner)
    ax2.text(0, 0, f"Z={Z}", color="white", fontsize=16, ha='center', va='center')
    ax2.text(0, -2, f"Z_eff = {Z_eff:.2f} σ = {sigma:.2f}", fontsize=12, ha='center')

    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.set_title('Shielding Effect in Helium Atom', fontsize=14)
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

# Example Usage and Test
if __name__ == "__main__":
    # Given: First ionization energy of helium
    instruction = "Given the first ionization energy of a helium atom I1=24.59 eV, calculate the second ionization energy, the repulsion energy between two electrons in the 1s orbital, the effective nuclear charge, and the shielding constant."
    I1_he = 24.59  # eV
    # Solve all parts
    results = math_ionization_energy_relations(I1_he, Z=2)

    # Visualize
    visual_func(results)

    # Example: Extend to other helium-like ions (e.g., Li⁺, Be²⁺)
    # print("\n" + "="*50)
    # print("Example: Lithium ion (Li⁺, Z=3)")
    # coding_func(I1=75.64, Z=3)