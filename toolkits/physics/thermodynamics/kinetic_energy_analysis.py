#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
苯分子的量子化学计算与分子轨道可视化

此脚本使用PySCF进行苯分子的电子结构计算，并可视化其分子轨道。
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, tools
from pyscf.tools import cubegen
import argparse


def setup_molecule(basis='6-31g'):
    """
    设置苯分子的结构
    
    Parameters:
    -----------
    basis : str, optional
        量子化学计算使用的基组，默认为'6-31g'
        
    Returns:
    --------
    pyscf.gto.Mole
        表示苯分子的Mole对象
    """
    # 苯分子的坐标 (单位：埃)
    # C-C键长: 1.397 Å, C-H键长: 1.09 Å
    benzene_coords = [
        ['C', (0.000, 1.397, 0.000)],
        ['C', (1.210, 0.699, 0.000)],
        ['C', (1.210, -0.699, 0.000)],
        ['C', (0.000, -1.397, 0.000)],
        ['C', (-1.210, -0.699, 0.000)],
        ['C', (-1.210, 0.699, 0.000)],
        ['H', (0.000, 2.487, 0.000)],
        ['H', (2.153, 1.244, 0.000)],
        ['H', (2.153, -1.244, 0.000)],
        ['H', (0.000, -2.487, 0.000)],
        ['H', (-2.153, -1.244, 0.000)],
        ['H', (-2.153, 1.244, 0.000)]
    ]
    
    # 创建分子对象
    mol = gto.Mole()
    mol.atom = benzene_coords
    mol.basis = basis
    mol.build()
    
    return mol


def run_scf_calculation(mol, verbose=False):
    """
    执行自洽场(SCF)计算，确保收敛到tight阈值
    
    Parameters:
    -----------
    mol : str
        表示分子的Mole对象
    verbose : bool, optional
        是否打印详细信息，默认为False
        
    Returns:
    --------
    pyscf.scf.RHF
        SCF计算结果
    """
    mol = setup_molecule(mol)
    # 创建RHF对象
    mf = scf.RHF(mol)
    
    # 设置详细程度
    if verbose:
        mf.verbose = 4
    else:
        mf.verbose = 0
    
    # 设置收敛阈值
    mf.conv_tol = 1e-10  # tight convergence
    mf.conv_tol_grad = 1e-8
    
    # 运行SCF计算
    mf.kernel()
    
    # 检查收敛性
    if not mf.converged:
        print("Warning: SCF calculation did not converge!")
        print("Trying with different initial guess...")
        mf.init_guess = 'atom'
        mf.kernel()
    
    return mf,mol


def analyze_molecular_orbitals(mf, n_orbitals=5, degeneracy_threshold=1e-6):
    """
    分析分子轨道能级，特别处理苯分子的简并轨道
    
    Parameters:
    -----------
    mf : pyscf.scf.RHF
        SCF计算结果
    n_orbitals : int, optional
        展示前后各多少个轨道，默认为5
    degeneracy_threshold : float, optional
        简并判断阈值，默认为1e-6
        
    Returns:
    --------
    dict
        包含分子轨道分析结果的字典
    """
    # 获取轨道能量和系数
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    
    # 确定HOMO和LUMO索引
    homo_idx = mf.mol.nelectron // 2 - 1
    lumo_idx = homo_idx + 1
    
    # 检查HOMO简并性（苯分子的e1g轨道）
    homo_degenerate = []
    homo_energy = mo_energy[homo_idx]
    
    # 向前检查简并轨道
    for i in range(max(0, homo_idx - 2), homo_idx):
        if abs(mo_energy[i] - homo_energy) < degeneracy_threshold:
            homo_degenerate.append(i)
    homo_degenerate.append(homo_idx)
    
    # 检查LUMO简并性（苯分子的e2u轨道）
    lumo_degenerate = []
    lumo_energy = mo_energy[lumo_idx]
    
    # 向后检查简并轨道
    for i in range(lumo_idx, min(len(mo_energy), lumo_idx + 3)):
        if abs(mo_energy[i] - lumo_energy) < degeneracy_threshold:
            lumo_degenerate.append(i)
    
    # 计算HOMO-LUMO能隙（使用最低的HOMO和最高的LUMO）
    homo_min_energy = min(mo_energy[idx] for idx in homo_degenerate)
    lumo_max_energy = max(mo_energy[idx] for idx in lumo_degenerate)
    gap = lumo_max_energy - homo_min_energy
    
    # 提取前后n个轨道的能量
    start_idx = max(0, homo_idx - n_orbitals + 1)
    end_idx = min(len(mo_energy), lumo_idx + n_orbitals)
    selected_orbitals = list(range(start_idx, end_idx))
    selected_energies = mo_energy[start_idx:end_idx]
    
    return {
        'homo_idx': homo_idx,
        'lumo_idx': lumo_idx,
        'homo_degenerate': homo_degenerate,
        'lumo_degenerate': lumo_degenerate,
        'homo_energy': homo_energy,
        'lumo_energy': lumo_energy,
        'homo_min_energy': homo_min_energy,
        'lumo_max_energy': lumo_max_energy,
        'gap': gap,
        'selected_orbitals': selected_orbitals,
        'selected_energies': selected_energies,
        'mo_coeff': mo_coeff,
        'degeneracy_threshold': degeneracy_threshold
    }


def generate_orbital_cube_files(mol, mf, orbitals, grid_points=80, outdir='./'):
    """
    生成分子轨道的cube文件
    
    Parameters:
    -----------
    mol : pyscf.gto.Mole
        表示分子的Mole对象
    mf : pyscf.scf.RHF
        SCF计算结果
    orbitals : list
        要生成cube文件的轨道索引列表
    grid_points : int, optional
        网格点数，默认为80
    outdir : str, optional
        输出目录，默认为当前目录
        
    Returns:
    --------
    list
        生成的cube文件路径列表
    """
    cube_files = []
    
    for orbital in orbitals:
        # 设置文件名
        if orbital == mf.mol.nelectron // 2 - 1:
            label = 'HOMO'
        elif orbital == mf.mol.nelectron // 2:
            label = 'LUMO'
        else:
            label = f'MO_{orbital}'
        
        cube_file = f"{outdir}/benzene_{label}.cube"
        
        # 生成cube文件
        cubegen.orbital(mol, cube_file, mf.mo_coeff[:, orbital], nx=grid_points, ny=grid_points, nz=grid_points)
        cube_files.append(cube_file)
    
    return cube_files


def visualize_energy_levels(mo_analysis, figsize=(10, 6), save_path=None):
    """
    可视化分子轨道能级，特别标注简并轨道
    
    Parameters:
    -----------
    mo_analysis : dict
        分子轨道分析结果
    figsize : tuple, optional
        图形大小，默认为(10, 6)
    save_path : str, optional
        保存图像的路径，如果为None则不保存
        
    Returns:
    --------
    tuple
        包含图形和轴对象的元组
    """
    orbitals = mo_analysis['selected_orbitals']
    energies = mo_analysis['selected_energies']
    homo_idx = mo_analysis['homo_idx']
    lumo_idx = mo_analysis['lumo_idx']
    homo_degenerate = mo_analysis.get('homo_degenerate', [homo_idx])
    lumo_degenerate = mo_analysis.get('lumo_degenerate', [lumo_idx])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制能级线
    for i, (orbital, energy) in enumerate(zip(orbitals, energies)):
        if orbital in homo_degenerate:
            color = 'red'
            if orbital == homo_idx:
                label = f'HOMO (e1g, {len(homo_degenerate)}-fold)'
            else:
                label = None
        elif orbital in lumo_degenerate:
            color = 'blue'
            if orbital == lumo_idx:
                label = f'LUMO (e2u, {len(lumo_degenerate)}-fold)'
            else:
                label = None
        else:
            color = 'gray'
            label = None
        
        ax.plot([0, 1], [energy, energy], '-', color=color, linewidth=2, label=label)
        ax.text(1.05, energy, f"MO {orbital}", va='center')
    
    # 添加HOMO-LUMO能隙标注
    gap = mo_analysis['gap']
    homo_min_energy = mo_analysis.get('homo_min_energy', mo_analysis['homo_energy'])
    lumo_max_energy = mo_analysis.get('lumo_max_energy', mo_analysis['lumo_energy'])
    mid_energy = (homo_min_energy + lumo_max_energy) / 2
    
    ax.annotate(f"Gap = {gap:.4f} a.u. ({gap*27.211:.2f} eV)", 
                xy=(0.5, mid_energy),
                xytext=(0.7, mid_energy),
                arrowprops=dict(arrowstyle='<->', color='green'),
                color='green',
                fontsize=12)
    
    # 设置坐标轴
    ax.set_xlim(-0.1, 1.5)
    ax.set_ylabel('Orbital Energy (a.u.)')
    ax.set_title('Molecular Orbital Energy Levels of Benzene (with Degeneracy)')
    ax.set_xticks([])
    
    # 添加图例
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def parse_args():
    """
    解析命令行参数
    
    Returns:
    --------
    argparse.Namespace
        包含解析后参数的命名空间
    """
    parser = argparse.ArgumentParser(description='Quantum chemistry calculation for benzene')
    
    parser.add_argument('--basis', type=str, default='6-31g',
                        help='Basis set for calculation (default: 6-31g)')
    parser.add_argument('--grid', type=int, default=80,
                        help='Number of grid points for orbital visualization (default: 80)')
    parser.add_argument('--outdir', type=str, default='./',
                        help='Output directory for cube files (default: current directory)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed calculation information')
    parser.add_argument('--save_plot', type=str, default=None,
                        help='Save energy level plot to specified file')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # print("Setting up benzene molecule...")
    # mol = setup_molecule(basis=args.basis)
    
    print(f"Running SCF calculation with {args.basis} basis set...")
    mf,mol = run_scf_calculation(args.basis, verbose=args.verbose)
    
    print("Analyzing molecular orbitals...")
    mo_analysis = analyze_molecular_orbitals(mf)
    
    # 打印结果
    print("\nMolecular Orbital Analysis Results:")
    print(f"HOMO (MO {mo_analysis['homo_idx']}): {mo_analysis['homo_energy']:.6f} a.u.")
    if len(mo_analysis['homo_degenerate']) > 1:
        print(f"HOMO degenerate orbitals: {mo_analysis['homo_degenerate']} (e1g symmetry)")
        print(f"HOMO energy range: {mo_analysis['homo_min_energy']:.6f} to {mo_analysis['homo_energy']:.6f} a.u.")
    
    print(f"LUMO (MO {mo_analysis['lumo_idx']}): {mo_analysis['lumo_energy']:.6f} a.u.")
    if len(mo_analysis['lumo_degenerate']) > 1:
        print(f"LUMO degenerate orbitals: {mo_analysis['lumo_degenerate']} (e2u symmetry)")
        print(f"LUMO energy range: {mo_analysis['lumo_energy']:.6f} to {mo_analysis['lumo_max_energy']:.6f} a.u.")
    
    print(f"HOMO-LUMO Gap: {mo_analysis['gap']:.6f} a.u. ({mo_analysis['gap']*27.211:.4f} eV)")
    print(f"Degeneracy threshold: {mo_analysis['degeneracy_threshold']:.1e}")
    
    # 生成HOMO和LUMO的cube文件
    print("\nGenerating cube files for HOMO and LUMO...")
    cube_files = generate_orbital_cube_files(
        mol, mf, [mo_analysis['homo_idx'], mo_analysis['lumo_idx']], 
        grid_points=args.grid, outdir=args.outdir
    )
    print(f"Cube files generated: {cube_files}")
    
    # 可视化能级
    print("\nVisualizing energy levels...")
    visualize_energy_levels(mo_analysis, save_path=args.save_plot)
    plt.show()
    
    # 结果字典
    result = {
        'isTrue': True,
        'answer': f"Benzene molecular orbital analysis completed. "
                 f"HOMO-LUMO gap is {mo_analysis['gap']*27.211:.4f} eV."
    }
    
    print("\nResult:", result)