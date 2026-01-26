import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf
import argparse

def calculate_h2_energy(bond_length, basis='sto-3g'):
    """
    计算给定键长的氢分子(H₂)能量。
    
    Parameters:
    -----------
    bond_length : float
        氢分子的键长，单位为埃(Å)
    basis : str, optional
        计算使用的基组，默认为'sto-3g'
        
    Returns:
    --------
    float
        分子能量，单位为哈特里(Hartree)
    """
    # 创建分子对象
    mol = gto.Mole()
    mol.atom = f'H 0 0 0; H 0 0 {bond_length}'  # 原子坐标
    mol.basis = basis
    mol.build()
    
    # 进行Hartree-Fock计算
    mf = scf.RHF(mol)
    energy = mf.kernel()
    
    return energy

def find_equilibrium_bond_length(bond_lengths, basis='sto-3g'):
    """
    通过计算一系列键长下的能量，找出平衡键长。
    
    Parameters:
    -----------
    bond_lengths : numpy.ndarray
        要计算的键长数组，单位为埃(Å)
    basis : str, optional
        计算使用的基组，默认为'sto-3g'
        
    Returns:
    --------
    tuple
        (平衡键长, 最小能量, 所有键长的能量)
    """
    energies = []
    
    # 计算每个键长对应的能量
    for length in bond_lengths:
        energy = calculate_h2_energy(length, basis)
        energies.append(energy)
    
    # 找出能量最低点
    energies = np.array(energies)
    min_idx = np.argmin(energies)
    equilibrium_length = bond_lengths[min_idx]
    min_energy = energies[min_idx]
    
    return equilibrium_length, min_energy, energies

def plot_potential_energy_curve(bond_lengths, energies, equilibrium_length=None, min_energy=None):
    """
    绘制势能曲线。
    
    Parameters:
    -----------
    bond_lengths : numpy.ndarray
        键长数组，单位为埃(Å)
    energies : numpy.ndarray
        对应的能量数组，单位为哈特里(Hartree)
    equilibrium_length : float, optional
        平衡键长，如果提供则会在图上标记
    min_energy : float, optional
        最小能量，如果提供则会在图上标记
        
    Returns:
    --------
    matplotlib.figure.Figure
        生成的图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制势能曲线
    ax.plot(bond_lengths, energies, 'o-', label='Potential Energy Curve')
    
    # 标记平衡点
    if equilibrium_length is not None and min_energy is not None:
        ax.plot(equilibrium_length, min_energy, 'r*', markersize=10, 
                label=f'Equilibrium Bond Length: {equilibrium_length:.3f} Å')
        
    ax.set_xlabel('Bond Length (Å)')
    ax.set_ylabel('Energy (Hartree)')
    ax.set_title('H₂ Potential Energy Curve')
    ax.grid(True)
    ax.legend()
    
    return fig

def parse_args():
    """
    解析命令行参数。
    
    Returns:
    --------
    argparse.Namespace
        解析后的参数
    """
    parser = argparse.ArgumentParser(description='H₂ Potential Energy Curve Calculator')
    parser.add_argument('--min_length', type=float, default=0.5,
                        help='Minimum bond length to calculate (Å)')
    parser.add_argument('--max_length', type=float, default=2.0,
                        help='Maximum bond length to calculate (Å)')
    parser.add_argument('--num_points', type=int, default=20,
                        help='Number of points to calculate')
    parser.add_argument('--basis', type=str, default='sto-3g',
                        help='Basis set for calculation')
    parser.add_argument('--output', type=str, default='h2_potential_curve.png',
                        help='Output file for the potential energy curve')
    
    return parser.parse_args()

def main():
    """
    主函数，执行H₂分子势能曲线计算和可视化。
    
    Returns:
    --------
    dict
        计算结果，包括平衡键长和最小能量
    """
    args = parse_args()
    
    # 生成要计算的键长数组
    bond_lengths = np.linspace(args.min_length, args.max_length, args.num_points)
    
    print(f"Calculating H₂ energies for {args.num_points} bond lengths between "
          f"{args.min_length} and {args.max_length} Å using {args.basis} basis...")
    
    # 计算平衡键长
    equilibrium_length, min_energy, energies = find_equilibrium_bond_length(bond_lengths, args.basis)
    
    print(f"Equilibrium bond length: {equilibrium_length:.4f} Å")
    print(f"Minimum energy: {min_energy:.6f} Hartree")
    
    # 绘制势能曲线
    fig = plot_potential_energy_curve(bond_lengths, energies, equilibrium_length, min_energy)
    
    # 保存图像
    fig.savefig(args.output)
    print(f"Potential energy curve saved to {args.output}")
    
    return {
        "isTrue": True,
        "answer": {
            "equilibrium_bond_length": float(equilibrium_length),
            "minimum_energy": float(min_energy)
        }
    }

if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")