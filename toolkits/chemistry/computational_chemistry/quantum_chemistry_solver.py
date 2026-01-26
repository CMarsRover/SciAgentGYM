# quantum_chemistry_solver.py

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft
from pyscf.tools import molden
from pyscf.tools import cubegen
import argparse
import os

def setup_molecule(molecule_spec, basis='cc-pvdz', charge=0, spin=0):
    """
    设置分子计算模型
    
    Parameters:
    -----------
    molecule_spec : str
        分子规格，可以是原子坐标列表或XYZ文件路径
    basis : str
        基组名称，默认为'cc-pvdz'
    charge : int
        分子电荷，默认为0
    spin : int
        自旋多重度减1，默认为0（单重态）
        
    Returns:
    --------
    pyscf.gto.Mole
        设置好的分子对象
    """
    mol = gto.Mole()
    mol.verbose = 4  # 输出详细信息
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    
    # 如果是文件路径，从文件读取分子规格
    if isinstance(molecule_spec, str) and os.path.exists(molecule_spec):
        try:
            mol.build(parse_xyz(molecule_spec))
        except:
            raise ValueError(f"无法从文件 {molecule_spec} 解析分子结构")
    else:
        mol.build(atom=molecule_spec)
    
    return mol

def parse_xyz(xyz_file: str):
    """
    解析XYZ文件
    
    Parameters:
    -----------
    xyz_file : str
        XYZ文件路径
        
    Returns:
    --------
    list
        原子坐标列表
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    # 跳过前两行（原子数和注释行）
    atom_list = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            atom = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atom_list.append([atom, (x, y, z)])
    
    return atom_list

def run_scf_calculation(molecule_spec, basis='cc-pvdz', charge=0, spin=0, method='RHF', xc=None, grid_level=3):
    """
    执行自洽场计算
    
    Parameters:
    -----------
    molecule_spec : str
        分子规格，可以是原子坐标列表或XYZ文件路径
    basis : str
        基组名称，默认为'cc-pvdz'
    charge : int
        分子电荷，默认为0
    spin : int
        自旋多重度减1，默认为0（单重态）
    method : str
        计算方法，可选'RHF'（限制性Hartree-Fock）或'DFT'（密度泛函理论），默认为'RHF'
    xc : str
        DFT计算中使用的交换关联泛函，默认为None
    grid_level : int
        DFT计算中的网格精度，默认为3
        
    Returns:
    --------
    pyscf.scf.hf.RHF or pyscf.dft.rks.RKS
        SCF计算结果对象
    """
    mol = setup_molecule(molecule_spec=molecule_spec, basis=basis, charge=charge, spin=spin)
    if method.upper() == 'RHF':
        mf = scf.RHF(mol)
    elif method.upper() == 'DFT':
        if xc is None:
            xc = 'B3LYP'  # 默认使用B3LYP泛函
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = grid_level
    else:
        raise ValueError(f"不支持的计算方法: {method}")
    
    # 执行SCF计算
    mf.kernel()
    
    return mf

def optimize_geometry(molecule_spec, basis='cc-pvdz', charge=0, spin=0, method='RHF', xc=None, grid_level=3):
    """
    优化分子几何结构
    
    Parameters:
    -----------
    molecule_spec : str
        分子规格，可以是原子坐标列表或XYZ文件路径
    basis : str
        基组名称，默认为'cc-pvdz'
    charge : int
        分子电荷，默认为0
    spin : int
        自旋多重度减1，默认为0（单重态）
    method : str
        计算方法，默认为'RHF'
    xc : str
        DFT计算中使用的交换关联泛函，默认为None
    grid_level : int
        DFT计算中的网格精度，默认为3
        
    Returns:
    --------
    tuple
        (优化后的分子对象, 优化后的能量, 优化后的原子坐标)
    """
    mol = setup_molecule(molecule_spec=molecule_spec, basis=basis, charge=charge, spin=spin)
    from pyscf.geomopt.berny_solver import optimize
    
    if method.upper() == 'RHF':
        mf = scf.RHF(mol)
    elif method.upper() == 'DFT':
        if xc is None:
            xc = 'B3LYP'
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = grid_level
    else:
        raise ValueError(f"不支持的计算方法: {method}")
    
    # 执行几何优化
    mol_opt = optimize(mf)
    
    # 用优化后的几何结构重新计算能量
    mol_new = mol_opt.copy()
    mf_new = run_scf_calculation(mol_new, method, xc, grid_level)
    
    return mol_new, mf_new.e_tot, mol_new.atom_coords()

def analyze_molecular_properties(molecule_spec, basis='cc-pvdz', charge=0, spin=0, method='RHF', xc=None, grid_level=3):
    """
    分析分子性质
    
    Parameters:
    -----------
    molecule_spec : str
        分子规格，可以是原子坐标列表或XYZ文件路径
    basis : str
        基组名称，默认为'cc-pvdz'
    charge : int
        分子电荷，默认为0
    spin : int
        自旋多重度减1，默认为0（单重态）
    method : str
        SCF计算方法，可选'RHF'（限制性Hartree-Fock）或'DFT'（密度泛函理论），默认为'RHF'
    xc : str
        DFT计算中使用的交换关联泛函，默认为None
    grid_level : int
        DFT计算中的网格精度，默认为3
        
    Returns:
    --------
    dict
        分子性质字典
    """
    mol = setup_molecule(molecule_spec=molecule_spec, basis=basis, charge=charge, spin=spin)
    mf = run_scf_calculation(molecule_spec=molecule_spec, basis=basis, charge=charge, spin=spin, method=method, xc=xc, grid_level=grid_level)
    # 计算Mulliken电荷
    mulliken_charges = mf.mulliken_pop()[1]
    
    # 计算偶极矩
    dipole = mf.dip_moment()
    
    # 计算HOMO-LUMO能隙
    mo_energy = mf.mo_energy
    homo_idx = mol.nelectron // 2 - 1
    homo_energy = mo_energy[homo_idx]
    lumo_energy = mo_energy[homo_idx + 1]
    gap = lumo_energy - homo_energy
    
    # 收集结果
    properties = {
        'total_energy': mf.e_tot,
        'nuclear_repulsion': mol.energy_nuc(),
        'electronic_energy': mf.e_tot - mol.energy_nuc(),
        'mulliken_charges': mulliken_charges,
        'dipole_moment': dipole,
        'homo_energy': homo_energy,
        'lumo_energy': lumo_energy,
        'homo_lumo_gap': gap
    }
    
    return properties

def parse_args():
    """
    解析命令行参数
    
    Returns:
    --------
    argparse.Namespace
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='量子化学计算工具')
    
    parser.add_argument('--molecule', type=str, default='H 0 0 0; O 0 0 1.1; H 0 1.0 1.1',
                        help='分子规格或XYZ文件路径')
    parser.add_argument('--basis', type=str, default='cc-pvdz',
                        help='基组名称')
    parser.add_argument('--method', type=str, default='RHF',
                        help='计算方法 (RHF或DFT)')
    parser.add_argument('--xc', type=str, default='B3LYP',
                        help='DFT交换关联泛函')
    parser.add_argument('--optimize', action='store_true',
                        help='执行几何优化')
    parser.add_argument('--orbitals', type=int, nargs='+',
                        help='要计算的轨道索引列表')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置分子
    print("设置分子...")
    mol = setup_molecule(args.molecule, basis=args.basis)
    
    # 是否执行几何优化
    if args.optimize:
        print("执行几何优化...")
        mol, opt_energy, opt_coords = optimize_geometry(args.molecule, basis=args.basis, method=args.method, xc=args.xc)
        print(f"优化后的能量: {opt_energy:.8f} Hartree")
        print("优化后的坐标:")
        for i, atom in enumerate(mol.atom_symbol()):
            x, y, z = opt_coords[i]
            print(f"{atom} {x:.6f} {y:.6f} {z:.6f}")
    
    # 执行SCF计算
    print(f"执行{args.method}计算...")
    mf = run_scf_calculation(args.molecule, basis=args.basis, method=args.method, xc=args.xc)
    
    # 分析分子性质
    print("分析分子性质...")
    properties = analyze_molecular_properties(args.molecule, basis=args.basis, method=args.method, xc=args.xc)
    
    # 打印结果
    print("\n分子性质:")
    print(f"总能量: {properties['total_energy']:.8f} Hartree")
    print(f"HOMO能量: {properties['homo_energy']:.6f} Hartree")
    print(f"LUMO能量: {properties['lumo_energy']:.6f} Hartree")
    print(f"HOMO-LUMO能隙: {properties['homo_lumo_gap']:.6f} Hartree ({properties['homo_lumo_gap']*27.211:.4f} eV)")
    print(f"偶极矩: {np.linalg.norm(properties['dipole_moment']):.4f} Debye")
    
    return {
        'isTrue': True,
        'answer': f"已成功计算{mol.basis}分子的量子化学性质。总能量为{properties['total_energy']:.8f} Hartree，HOMO-LUMO能隙为{properties['homo_lumo_gap']*27.211:.4f} eV。"
    }

if __name__ == "__main__":
    result = main()
    print(f"\n结果: {result['answer']}")