#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
    主要依据SMILES格式构建，基于RDKit工具做开发，生成3D坐标、计算分子的3D性质、用mol、pdb、xyz三种格式保存分子的3D结构;
"""
import pubchempy as pcp  
import json 
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, rdMolDescriptors, SanitizeFlags
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, rdchem
from rdkit import DataStructs
from rdkit.Chem.rdChemReactions import ReactionFromSmarts 
from rdkit.Chem import rdMolDescriptors, rdDistGeom
import numpy as np 
from typing import Union
import sys
import os

# 添加当前文件所在目录到sys.path，确保可以导入py_smiles
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from py_smiles import chemical_name_to_molecular_formula
class SMILES3DGenerator:
    """SMILES转3D分子生成器"""
    
    def __init__(self, smiles):
        """
        初始化
        
        Args:
            smiles: SMILES字符串
        """
        self.smiles = smiles
        self.mol_2d = None
        self.mol_3d = None
        self.energy_before = None
        self.energy_after = None
        
    def _try_parse_smiles(self, smiles, errors):
        """
        尝试解析SMILES字符串，必要时跳过kekulization再补做芳香性设定。
        """
        if not smiles:
            errors.append("空SMILES字符串")
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return mol
        except Exception as exc:  # 捕获所有异常，记录日志信息
            # 提取更友好的错误信息
            error_msg = str(exc)
            if "SMILES Parse Error" in error_msg or "syntax error" in error_msg.lower():
                errors.append(f"SMILES语法错误: 无法解析 '{smiles}'（可能是化学名称而非SMILES格式）")
            else:
                errors.append(f"MolFromSmiles(sanitize=True)失败: {exc}")
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        except Exception as exc:
            errors.append(f"MolFromSmiles(sanitize=False)失败: {exc}")
            return None
        if mol is None:
            errors.append("MolFromSmiles(sanitize=False)返回None")
            return None
        try:
            Chem.SanitizeMol(mol)
            return mol
        except rdchem.KekulizeException as exc:
            errors.append(f"Kekulize失败: {exc}")
            try:
                partial_ops = SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_KEKULIZE
                Chem.SanitizeMol(mol, sanitizeOps=partial_ops)
                Chem.SetAromaticity(mol)
                return mol
            except Exception as inner_exc:
                errors.append(f"部分Sanitize失败: {inner_exc}")
                return None
        except Exception as exc:
            errors.append(f"Sanitize失败: {exc}")
            return None

    def parse_smiles(self):
        """解析SMILES"""
        if self.mol_2d is not None:
            return self.mol_2d

        errors = []
        mol = self._try_parse_smiles(self.smiles, errors)
        if mol is None:
            # 尝试将输入作为化学名称进行转换
            resolved_smiles = None
            try:
                resolved_smiles = chemical_name_to_molecular_formula(self.smiles)
            except Exception as exc:
                errors.append(f"名称解析失败: {exc}")
            if resolved_smiles and resolved_smiles != self.smiles:
                # 如果成功转换为SMILES，更新self.smiles并尝试解析
                print(f"  检测到化学名称 '{self.smiles}'，已转换为SMILES: {resolved_smiles}")
                self.smiles = resolved_smiles  # 更新为转换后的SMILES
                mol = self._try_parse_smiles(resolved_smiles, errors)
        if mol is None:
            detail = "; ".join(errors) if errors else "未提供额外错误信息"
            raise ValueError(
                f"无法解析输入 '{self.smiles}'\n"
                f"错误详情: {detail}\n"
                f"提示: 请确保输入是有效的SMILES字符串或化学名称（如 'benzene', 'aspirin' 等）"
            )

        self.mol_2d = mol 

        
        print(f"✓ SMILES解析成功: {self.smiles}")
        print(f"  分子式: {rdMolDescriptors.CalcMolFormula(self.mol_2d)}")
        print(f"  原子数: {self.mol_2d.GetNumAtoms()}")
        print(f"  键数: {self.mol_2d.GetNumBonds()}")
        
        return self.mol_2d
    
    def generate_3d(self, method='ETKDG', num_confs=1, random_seed=42):
        """
        生成3D坐标
        
        Args:
            method: 'ETKDG' (推荐), 'ETKDGv3', 'basic'
            num_confs: 生成构象数量
            random_seed: 随机种子
        """
        if self.mol_2d is None:
            self.parse_smiles()
        if self.mol_2d is None:
            raise RuntimeError("SMILES解析失败，无法生成3D坐标")
        
        # 添加氢原子
        self.mol_3d = Chem.AddHs(self.mol_2d)
        
        method_key = (method or 'ETKDGv3')
        print(f"\n生成3D坐标（方法: {method_key}）...")
        method_lower = method_key.lower()
        
        if method_lower in ('etkdg', 'etkdgv3'):
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = random_seed
            success = AllChem.EmbedMultipleConfs(self.mol_3d, num_confs, params)
        else:  # basic
            success = AllChem.EmbedMolecule(self.mol_3d, randomSeed=random_seed)
        
        if success == -1 or (isinstance(success, tuple) and -1 in success):
            raise RuntimeError("3D坐标生成失败")
        
        print(f"✓ 3D坐标生成成功")
        if num_confs > 1:
            print(f"  生成了 {self.mol_3d.GetNumConformers()} 个构象")
        
        return self.mol_3d
    
    def optimize_geometry(self, force_field='MMFF', max_iters=200):
        """
        优化分子几何（能量最小化）
        
        Args:
            force_field: 'MMFF' 或 'UFF'
            max_iters: 最大迭代次数
        """
        # 处理 max_iters 为 None 的情况
        if max_iters is None:
            max_iters = 200
            
        if self.mol_3d is None:
            raise RuntimeError("请先生成3D坐标")
        
        print(f"\n能量最小化（力场: {force_field}）...")
        
        num_confs = self.mol_3d.GetNumConformers()
        energies_before = []
        energies_after = []
        
        for conf_id in range(num_confs):
            if force_field == 'MMFF':
                # MMFF94力场
                props = AllChem.MMFFGetMoleculeProperties(self.mol_3d)
                if props is None:
                    print("  警告: MMFF力场不适用，切换到UFF")
                    force_field = 'UFF'
                else:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        self.mol_3d, props, confId=conf_id
                    )
                    energy_before = ff.CalcEnergy()
                    energies_before.append(energy_before)
                    
                    # 优化
                    converged = ff.Minimize(max_iters)
                    energy_after = ff.CalcEnergy()
                    energies_after.append(energy_after)
            
            if force_field == 'UFF':
                # UFF力场
                ff = AllChem.UFFGetMoleculeForceField(self.mol_3d, confId=conf_id)
                energy_before = ff.CalcEnergy()
                energies_before.append(energy_before)
                
                converged = ff.Minimize(max_iters)
                energy_after = ff.CalcEnergy()
                energies_after.append(energy_after)
        
        self.energy_before = energies_before
        self.energy_after = energies_after
        
        print(f"✓ 能量最小化完成")
        for i, (e_b, e_a) in enumerate(zip(energies_before, energies_after)):
            delta = e_a - e_b
            print(f"  构象 {i}: {e_b:.2f} → {e_a:.2f} kcal/mol (Δ={delta:.2f})")
        
        return self.mol_3d
    
    def get_3d_properties(self, conf_id=0):
        """计算3D性质"""
        if self.mol_3d is None:
            raise RuntimeError("请先生成3D坐标")
        
        print(f"\n3D分子性质（构象 {conf_id}）:")
        
        # 主惯性矩
        pmi1 = rdMolDescriptors.CalcPMI1(self.mol_3d, confId=conf_id)
        pmi2 = rdMolDescriptors.CalcPMI2(self.mol_3d, confId=conf_id)
        pmi3 = rdMolDescriptors.CalcPMI3(self.mol_3d, confId=conf_id)
        print(f"  主惯性矩: {pmi1:.2f}, {pmi2:.2f}, {pmi3:.2f}")
        
        # NPR (Normalized Principal Moments Ratio)
        npr1 = rdMolDescriptors.CalcNPR1(self.mol_3d, confId=conf_id)
        npr2 = rdMolDescriptors.CalcNPR2(self.mol_3d, confId=conf_id)
        print(f"  NPR1: {npr1:.3f}, NPR2: {npr2:.3f}")
        
        # 形状判断
        if npr1 < 0.8:
                if npr1 < 0.3:
                    shape = "极端棒状/纤维状"
                else:
                    shape = "棒状"
        elif npr1 > 1.7:
            shape = "盘状"
        elif 0.8 <= npr1 <= 1.2 and npr2 > 0.95:
            shape = "近球形"
        else:
            shape = "不对称过渡形状"

        # 补充NPR2的辅助判断
        if "球形" not in shape and npr2 < 0.85:
            shape += "（高度不对称）"
        print(f"  分子形状: {shape}")
        
        # 回转半径
        radius = rdMolDescriptors.CalcRadiusOfGyration(self.mol_3d, confId=conf_id)
        print(f"  回转半径: {radius:.3f} Å")
        
        # 非球形度
        asphericity = rdMolDescriptors.CalcAsphericity(self.mol_3d, confId=conf_id)
        print(f"  非球形度: {asphericity:.3f}")
        
        # 偏心率
        eccentricity = rdMolDescriptors.CalcEccentricity(self.mol_3d, confId=conf_id)
        print(f"  偏心率: {eccentricity:.3f}")
        
        return {
            'pmi': (pmi1, pmi2, pmi3),
            'npr': (npr1, npr2),
            'shape': shape,
            'radius': radius,
            'asphericity': asphericity,
            'eccentricity': eccentricity
        }
    
    def save_to_file(self, filename, format='mol', conf_id=0):
        """
        保存3D分子到文件
        
        Args:
            filename: 文件名
            format: 'mol', 'sdf', 'pdb', 'xyz'
            conf_id: 构象ID
        """
        if self.mol_3d is None:
            raise RuntimeError("请先生成3D坐标")
        
        if format.lower() == 'mol' or format.lower() == 'sdf':
            writer = Chem.SDWriter(filename)
            writer.write(self.mol_3d, confId=conf_id)
            writer.close()
        elif format.lower() == 'pdb':
            Chem.MolToPDBFile(self.mol_3d, filename, confId=conf_id)
        elif format.lower() == 'xyz':
            # XYZ格式
            conf = self.mol_3d.GetConformer(conf_id)
            with open(filename, 'w') as f:
                f.write(f"{self.mol_3d.GetNumAtoms()}\n")
                f.write(f"{self.smiles}\n")
                for atom in self.mol_3d.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    symbol = atom.GetSymbol()
                    f.write(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
        
        print(f"\n✓ 已保存到 {filename} ({format.upper()}格式)") 

## func_calling
def mol_analyzer_generate_3d(smiles:str, method:str):
    """
    从SMILES字符串生成分子的3D构象
    
    基于RDKit的嵌入算法（ETKDG/ETKDGv3）生成分子的三维坐标。
    ETKDG (Experimental Torsion-angle Knowledge Distance Geometry) 
    算法结合了实验扭转角知识和距离几何方法，生成更符合真实的分子构象。
    
    Parameters:
    -----------
    smiles : str
        分子的SMILES表示字符串，或化学名称（会自动转换）
        例如: 'CCO', 'c1ccccc1', 'aspirin'
    method : str
        3D坐标生成方法：
        - 'ETKDG': 标准ETKDG方法
        - 'ETKDGv3': 改进的ETKDGv3方法（推荐）
        - 'basic': 基本嵌入方法
        
    Returns:
    --------
    rdkit.Chem.rdchem.Mol
        包含3D坐标的RDKit分子对象，已添加氢原子
        
    Examples:
    ---------
    >>> # 生成乙醇的3D构象
    >>> mol_3d = mol_analyzer_generate_3d('CCO', method='ETKDGv3')
    >>> print(f"原子数: {mol_3d.GetNumAtoms()}")
    原子数: 9
    
    >>> # 使用化学名称
    >>> mol_3d = mol_analyzer_generate_3d('benzene', method='ETKDG')
    
    Note:
    -----
    生成的分子对象未经能量优化，建议后续使用optimize_geometry()进行优化
    """  
    generator = SMILES3DGenerator(smiles) 
    mol_3d = generator.generate_3d(method=method or 'ETKDGv3') 
    return mol_3d  

def optimize_geometry(smiles:str, method:str, force_field='MMFF', max_iters=200): 
        """
        分子几何优化工具（能量最小化）
        
        基于分子力场（MMFF94或UFF）进行能量最小化，优化分子的三维几何构型，
        使其达到局部能量最小值。通过迭代调整原子坐标，减小分子内部应力。
        该函数内部会先调用mol_analyzer_generate_3d生成3D构象，然后进行优化。
        
        Parameters:
        -----------
        smiles : str
           
        method : str
            3D坐标生成方法：
            - 'ETKDG': 标准ETKDG方法
            - 'ETKDGv3': 改进的ETKDGv3方法（推荐）
            - 'basic': 基本嵌入方法
        force_field : str, optional
            力场类型，默认为'MMFF'
            - 'MMFF': MMFF94力场，适用于大多数有机分子，精度较高
            - 'UFF': 通用力场，适用范围更广但精度稍低
        max_iters : int, optional
            能量最小化的最大迭代次数，默认200次
            
        Returns:
        --------
        float
            优化后的最低能量值（单位: kcal/mol）
            
        
        """
        # 处理 max_iters 为 None 的情况
        if max_iters is None:
            max_iters = 200
        
        mol_3d = mol_analyzer_generate_3d(smiles,method)
       
        
        print(f"\n能量最小化（力场: {force_field}）...")
        
        num_confs = mol_3d.GetNumConformers()
        energies_before = []
        energies_after = []
        
        for conf_id in range(num_confs):
            if force_field == 'MMFF':
                # MMFF94力场
                props = AllChem.MMFFGetMoleculeProperties(mol_3d)
                if props is None:
                    print("  警告: MMFF力场不适用，切换到UFF")
                    force_field = 'UFF'
                else:
                    ff = AllChem.MMFFGetMoleculeForceField(
                        mol_3d, props, confId=conf_id
                    )
                    energy_before = ff.CalcEnergy()
                    energies_before.append(energy_before)
                    
                    # 优化
                    converged = ff.Minimize(max_iters)
                    energy_after = ff.CalcEnergy()
                    energies_after.append(energy_after)
            
            if force_field == 'UFF':
                # UFF力场
                ff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=conf_id)
                energy_before = ff.CalcEnergy()
                energies_before.append(energy_before)
                
                converged = ff.Minimize(max_iters)
                energy_after = ff.CalcEnergy()
                energies_after.append(energy_after)
        
        gen_energy_before = energies_before
        gen_energy_after = energies_after
        
        print(f"✓ 能量最小化完成")
        for i, (e_b, e_a) in enumerate(zip(gen_energy_before, gen_energy_after)):
            delta = e_a - e_b
            print(f"  构象 {i}: {e_b:.2f} → {e_a:.2f} kcal/mol (Δ={delta:.2f})") 

        #  # 找到能量最低的构象
        # min_energy_id = gen_energy_after.index(min(gen_energy_after))
        # print(f"\n最低能量构象: {min_energy_id}")
        # print(f"能量: {gen_energy_after[min_energy_id]:.2f} kcal/mol")
        
        return gen_energy_after[gen_energy_after.index(min(gen_energy_after))]

def get_3d_properties(smiles:str, method:str, conf_id=0):
        """
        计算分子的3D几何性质和形状描述符
        
        基于主惯性矩（PMI）分析分子的三维形状特征，包括：
        - 主惯性矩 (PMI1, PMI2, PMI3)
        - 归一化主矩比 (NPR1, NPR2)
        - 分子形状分类（球形、棒状、盘状等）
        - 回转半径、非球形度、偏心率等几何参数
        该函数内部会先调用mol_analyzer_generate_3d生成3D构象，然后计算性质。
        
        Parameters:
        -----------
        smiles : str
            分子的SMILES表示字符串，或化学名称（会自动转换）
            例如: 'CCO', 'c1ccccc1', 'aspirin'
        method : str
            3D坐标生成方法：
            - 'ETKDG': 标准ETKDG方法
            - 'ETKDGv3': 改进的ETKDGv3方法（推荐）
            - 'basic': 基本嵌入方法
        conf_id : int, optional
            构象ID，默认为0（第一个构象）
            
        Returns:
        --------
        dict
            包含以下3D性质的字典：
            - 'pmi': (pmi1, pmi2, pmi3) - 三个主惯性矩值
            - 'npr': (npr1, npr2) - 归一化主矩比
            - 'shape': str - 分子形状描述
            - 'radius': float - 回转半径 (Å)
            - 'asphericity': float - 非球形度 [0,1]
            - 'eccentricity': float - 偏心率 [0,1]
            
       
        """
        
        mol_3d = mol_analyzer_generate_3d(smiles,method)
        print(f"\n3D分子性质（构象 {conf_id}）:")
        
        # 主惯性矩
        pmi1 = rdMolDescriptors.CalcPMI1(mol_3d, confId=conf_id)
        pmi2 = rdMolDescriptors.CalcPMI2(mol_3d, confId=conf_id)
        pmi3 = rdMolDescriptors.CalcPMI3(mol_3d, confId=conf_id)
        print(f"  主惯性矩: {pmi1:.2f}, {pmi2:.2f}, {pmi3:.2f}")
        
        # NPR (Normalized Principal Moments Ratio)
        npr1 = rdMolDescriptors.CalcNPR1(mol_3d, confId=conf_id)
        npr2 = rdMolDescriptors.CalcNPR2(mol_3d, confId=conf_id)
        print(f"  NPR1: {npr1:.3f}, NPR2: {npr2:.3f}")
        
        # 形状判断
        if npr1 < 0.8:
                if npr1 < 0.3:
                    shape = "极端棒状/纤维状"
                else:
                    shape = "棒状"
        elif npr1 > 1.7:
            shape = "盘状"
        elif 0.8 <= npr1 <= 1.2 and npr2 > 0.95:
            shape = "近球形"
        else:
            shape = "不对称过渡形状"

        # 补充NPR2的辅助判断
        if "球形" not in shape and npr2 < 0.85:
            shape += "（高度不对称）"
        print(f"  分子形状: {shape}")
        
        # 回转半径
        radius = rdMolDescriptors.CalcRadiusOfGyration(mol_3d, confId=conf_id)
        print(f"  回转半径: {radius:.3f} Å")
        
        # 非球形度
        asphericity = rdMolDescriptors.CalcAsphericity(mol_3d, confId=conf_id)
        print(f"  非球形度: {asphericity:.3f}")
        
        # 偏心率
        eccentricity = rdMolDescriptors.CalcEccentricity(mol_3d, confId=conf_id)
        print(f"  偏心率: {eccentricity:.3f}")
        
        return {
            'pmi': (pmi1, pmi2, pmi3),
            'npr': (npr1, npr2),
            'shape': shape,
            'radius': radius,
            'asphericity': asphericity,
            'eccentricity': eccentricity
        }
    
def save_to_file(smiles, mol_3d, method, filename, format='mol', conf_id=0):
        """
        保存3D分子结构到文件
        
        支持多种常用的分子文件格式，可用于分子建模软件（如PyMOL、VMD、Chimera等）
        进行可视化和进一步分析。该函数内部会先调用mol_analyzer_generate_3d生成3D构象。
        
        Parameters:
        -----------
        smiles : str
            分子的SMILES表示字符串，或化学名称（会自动转换）
            例如: 'CCO', 'c1ccccc1', 'aspirin'
        mol_3d : rdkit.Chem.rdchem.Mol
            包含3D坐标的RDKit分子对象（此参数会被内部生成的对象覆盖）
        method : str
            3D坐标生成方法：
            - 'ETKDG': 标准ETKDG方法
            - 'ETKDGv3': 改进的ETKDGv3方法（推荐）
            - 'basic': 基本嵌入方法
        filename : str
            输出文件名（含路径）
        format : str, optional
            文件格式，默认为'mol'
            - 'mol' 或 'sdf': MDL Molfile格式（标准化学文件格式）
            - 'pdb': Protein Data Bank格式（蛋白质和小分子）
            - 'xyz': XYZ坐标格式（简单的笛卡尔坐标）
        conf_id : int, optional
            要保存的构象ID，默认为0（第一个构象）
            
        Returns:
        --------
        None
            直接保存文件到指定路径
            
       
        """
        
        mol_3d = mol_analyzer_generate_3d(smiles,method)
        
        if format.lower() == 'mol' or format.lower() == 'sdf':
            writer = Chem.SDWriter(filename)
            writer.write(mol_3d, confId=conf_id)
            writer.close()
        elif format.lower() == 'pdb':
            Chem.MolToPDBFile(mol_3d, filename, confId=conf_id)
        elif format.lower() == 'xyz':
            # XYZ格式
            conf = mol_3d.GetConformer(conf_id)
            with open(filename, 'w') as f:
                f.write(f"{mol_3d.GetNumAtoms()}\n")
                f.write(f"{smiles}\n")
                for atom in mol_3d.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    symbol = atom.GetSymbol()
                    f.write(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n")
        
        print(f"\n✓ 已保存到 {filename} ({format.upper()}格式)") 


class MoleculeAnalyzer:
    """分子分析工具"""
    
    def __init__(self, smiles: str):
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        if self.mol is None: 
            smiles = chemical_name_to_molecular_formula(smiles)  
            if smiles is None:
                raise ValueError(f"无法解析SMILES: {self.smiles}") 
            else:
                self.mol = Chem.MolFromSmiles(smiles)

    def get_basic_info(self):
        """获取基本信息"""
        return {
            'SMILES': self.smiles,
            'Molecular Formula': rdMolDescriptors.CalcMolFormula(self.mol),
            'Molecular Weight': f"{Descriptors.ExactMolWt(self.mol):.2f}",
            'Heavy Atom Count': self.mol.GetNumHeavyAtoms(),
            'Total Atom Count': self.mol.GetNumAtoms()
        }
    
    def get_properties(self):
        """获取物化性质"""
        return {
            'LogP': f"{Descriptors.MolLogP(self.mol):.2f}",
            'TPSA': f"{Descriptors.TPSA(self.mol):.2f}",
            'H-Bond Donors': Descriptors.NumHDonors(self.mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(self.mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(self.mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(self.mol),
            'Total Rings': Descriptors.RingCount(self.mol)
        }
    
    def get_lipinski_rule(self):
        """Lipinski五规则"""
        mw = Descriptors.ExactMolWt(self.mol)
        logp = Descriptors.MolLogP(self.mol)
        hbd = Descriptors.NumHDonors(self.mol)
        hba = Descriptors.NumHAcceptors(self.mol)
        
        violations = 0
        rules = {}
        
        rules['MW ≤ 500'] = mw <= 500
        if not rules['MW ≤ 500']:
            violations += 1
            
        rules['LogP ≤ 5'] = logp <= 5
        if not rules['LogP ≤ 5']:
            violations += 1
            
        rules['HBD ≤ 5'] = hbd <= 5
        if not rules['HBD ≤ 5']:
            violations += 1
            
        rules['HBA ≤ 10'] = hba <= 10
        if not rules['HBA ≤ 10']:
            violations += 1
        
        return {
            'violations': violations,
            'drug_like': violations <= 1,
            'rules': rules
        }

def mol_basic_physicochemical_info(smiles):
    """
    计算分子的基本理化性质和类药性评估
    
    基于RDKit的描述符计算功能，提供分子的基本信息、物化性质和Lipinski五规则评估。
    Lipinski五规则是评估口服药物类药性的经验法则。
    
    Parameters:
    -----------
    smiles : str
        分子的SMILES表示字符串，或化学名称（会自动转换）
        例如: 'CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'aspirin'
        
    Returns:
    --------
    str
        格式化的分子信息报告，包含三个部分：
        
        【基本信息】
        - SMILES: 标准化的SMILES字符串
        - Molecular Formula: 分子式
        - Molecular Weight: 分子量
        - Heavy Atom Count: 重原子数（非氢原子）
        - Total Atom Count: 总原子数（含氢）
        
        【物化性质】
        - LogP: 辛醇-水分配系数（亲脂性）
        - TPSA: 拓扑极性表面积（Å²）
        - H-Bond Donors: 氢键供体数
        - H-Bond Acceptors: 氢键受体数
        - Rotatable Bonds: 可旋转键数（分子柔性）
        - Aromatic Rings: 芳香环数
        - Total Rings: 总环数
        
        【Lipinski规则】（类药性评估）
        - 违规数: 违反规则的数量
        - 类药性: ≤1个违规视为类药
        - MW ≤ 500: 分子量不超过500 Da
        - LogP ≤ 5: 亲脂性不超过5
        - HBD ≤ 5: 氢键供体不超过5个
        - HBA ≤ 10: 氢键受体不超过10个
        
    
    """ 
    mol = MoleculeAnalyzer(smiles)
    info = '' 
    info += "\n【基本信息】"
    for key, value in mol.get_basic_info().items():
            info += f"  {key:<25} {value}"
        
    info += "\n【物化性质】"
    for key, value in mol.get_properties().items():
            info += f"  {key:<25} {value}"
        
    info += "\n【Lipinski规则】"
    lipinski = mol.get_lipinski_rule()
    info += f"  违规数: {lipinski['violations']}"
    info += f"  类药性: {'✓ 是' if lipinski['drug_like'] else '✗ 否'}"
    for rule, passed in lipinski['rules'].items():
            status = '✓' if passed else '✗'
            info += f"    {status} {rule}" 
    return mol, info





def generate_multiple_conformers_with_optimization(smiles: str, num_confs: int = 10, 
                                                    method: str = 'ETKDGv3', 
                                                    force_field: str = 'MMFF', 
                                                    max_iters: int = 200):
    """
    为分子生成多个构象并进行能量优化
    
    这是一个集成工具函数，用于处理需要多构象分析的场景，如构象搜索、
    柔性分子研究等。函数会生成指定数量的初始构象，然后对每个构象
    进行能量最小化优化。
    
    Parameters:
    -----------
    smiles : str
        分子的SMILES表示字符串
    num_confs : int, optional
        要生成的构象数量，默认为10
    method : str, optional
        3D坐标生成方法，默认'ETKDGv3'
        - 'ETKDG': 标准ETKDG方法
        - 'ETKDGv3': 改进版本（推荐）
    force_field : str, optional
        力场类型，默认'MMFF'
        - 'MMFF': MMFF94力场
        - 'UFF': 通用力场
    max_iters : int, optional
        能量优化的最大迭代次数，默认200
        
    Returns:
    --------
    dict
        包含以下键的字典：
        - 'energies': list[float] - 所有构象优化后的能量列表(kcal/mol)
        - 'E_min': float - 最低能量值
        - 'E_max': float - 最高能量值
        - 'min_conf_id': int - 最低能量构象的索引
        - 'delta_E_span': float - 能量跨度(E_max - E_min)
        
    Examples:
    ---------
    >>> # 生成环己烷的10个构象并优化
    >>> result = generate_multiple_conformers_with_optimization('C1CCCCC1', num_confs=10)
    >>> print(f"能量跨度: {result['delta_E_span']:.2f} kcal/mol")
    >>> print(f"最低能量: {result['E_min']:.2f} kcal/mol")
    
    >>> # 分析柔性分子的构象分布
    >>> result = generate_multiple_conformers_with_optimization('CCCCCCCC', num_confs=20)
    >>> energies = result['energies']
    >>> print(f"找到{len(energies)}个构象")
    
    Note:
    -----
    该函数会打印生成和优化过程的详细信息
    """
    # 创建生成器
    generator = SMILES3DGenerator(smiles)
    generator.parse_smiles()
    
    # 生成多个构象
    print(f"\n生成{num_confs}个3D构象...")
    mol_3d = generator.generate_3d(method=method, num_confs=num_confs)
    
    # 优化所有构象
    print(f"\n能量最小化（力场: {force_field}）...")
    num_conformers = mol_3d.GetNumConformers()
    energies = []
    use_uff = False
    
    for conf_id in range(num_conformers):
        if force_field == 'MMFF' and not use_uff:
            props = AllChem.MMFFGetMoleculeProperties(mol_3d)
            if props is None:
                print(f"  警告: MMFF力场不适用，切换到UFF")
                use_uff = True
            else:
                ff = AllChem.MMFFGetMoleculeForceField(mol_3d, props, confId=conf_id)
                ff.Minimize(max_iters)
                energy = ff.CalcEnergy()
                energies.append(energy)
        
        if force_field == 'UFF' or use_uff:
            ff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=conf_id)
            ff.Minimize(max_iters)
            energy = ff.CalcEnergy()
            energies.append(energy)
    
    print(f"✓ 能量最小化完成，优化了 {len(energies)} 个构象")
    
    if energies:
        E_min = min(energies)
        E_max = max(energies)
        min_conf_id = energies.index(E_min)
        delta_E_span = E_max - E_min
        
        print(f"  能量范围: {E_min:.2f} - {E_max:.2f} kcal/mol")
        print(f"  能量跨度: {delta_E_span:.2f} kcal/mol")
        
        return {
            'energies': energies,
            'E_min': E_min,
            'E_max': E_max,
            'min_conf_id': min_conf_id,
            'delta_E_span': delta_E_span
        }
    else:
        raise RuntimeError("未能成功优化任何构象")

def calculate_synthetic_accessibility(smiles: str) -> float:
    """
    计算分子的合成可及性分数 (Synthetic Accessibility Score, SAS)
    
    合成可及性评分是评估有机分子合成难度的重要指标，基于分子的结构复杂度、
    环系统、立体化学以及已知的化学反应和合成片段。该评分由Ertl和Schuffenhauer
    开发，广泛用于药物发现和虚拟筛选。
    
    Parameters:
    -----------
    smiles : str
        分子的SMILES表示字符串，或化学名称（会自动转换）
        例如: 'CCO', 'CC(=O)Oc1ccccc1C(=O)O', 'aspirin'
        
    Returns:
    --------
    float
        合成可及性分数，范围1.0-10.0，保留1位小数
        - 1.0: 非常容易合成（如简单的醇、醚、烷烃）
        - 2.0-4.0: 相对容易合成（常见的药物骨架）
        - 4.0-6.0: 中等难度（复杂的杂环、多手性中心）
        - 6.0-8.0: 较难合成（复杂的天然产物骨架）
        - 8.0-10.0: 非常难合成（复杂的天然产物、多个手性中心）
        
    Raises:
    -------
    ValueError
        如果SMILES字符串无效或无法解析
    ImportError
        如果RDKit的SA_Score模块未安装
        
    Examples:
    ---------
    >>> # 简单分子（乙醇）
    >>> score = calculate_synthetic_accessibility('CCO')
    >>> print(f"乙醇的SAS: {score}")  # 约1.0
    
    >>> # 中等复杂度（阿司匹林）
    >>> score = calculate_synthetic_accessibility('CC(=O)Oc1ccccc1C(=O)O')
    >>> print(f"阿司匹林的SAS: {score}")  # 约2.5
    
    >>> # 复杂分子（紫杉醇）
    >>> score = calculate_synthetic_accessibility('taxol')
    >>> print(f"紫杉醇的SAS: {score}")  # 约8.0+
    
    >>> # 比较不同分子
    >>> molecules = {
    ...     '甲醇': 'CO',
    ...     '布洛芬': 'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    ...     '青蒿素': 'CC1CCC2C(=C)C(=O)OC2C3CCC4C1(C)CCC(O3)C4(C)OO'
    ... }
    >>> for name, smi in molecules.items():
    ...     score = calculate_synthetic_accessibility(smi)
    ...     print(f"{name}: SAS = {score}")
    
    Note:
    -----
    - SAS基于大量已知化合物的统计分析
    - 分数越低表示越容易合成
    - 该评分是估计值，实际合成难度还取决于实验室条件、试剂可得性等
    - 对于化学名称输入，会先尝试转换为SMILES
    
    References:
    -----------
    Ertl, P., & Schuffenhauer, A. (2009). 
    "Estimation of synthetic accessibility score of drug-like molecules 
    based on molecular complexity and fragment contributions."
    Journal of Cheminformatics, 1(1), 8.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        raise ImportError(
            "RDKit的SA_Score模块未找到。请确保RDKit安装完整，"
            "或手动下载sascorer.py到RDKit/Contrib/SA_Score/目录"
        )
    
    # 尝试解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    
    # 如果解析失败，尝试作为化学名称处理
    if mol is None:
        try:
            smiles_converted = chemical_name_to_molecular_formula(smiles)
            if smiles_converted:
                mol = Chem.MolFromSmiles(smiles_converted)
        except:
            pass
    
    # 如果仍然无法解析，抛出错误
    if mol is None:
        raise ValueError(
            f"无法解析输入: {smiles}\n"
            f"请提供有效的SMILES字符串或化学名称"
        )
    
    # 计算合成可及性分数
    try:
        sa_score = sascorer.calculateScore(mol)
        # 四舍五入到1位小数
        sa_score_rounded = round(sa_score, 1)
        
        # 确保分数在合理范围内
        if sa_score_rounded < 1.0:
            sa_score_rounded = 1.0
        elif sa_score_rounded > 10.0:
            sa_score_rounded = 10.0
            
        return sa_score_rounded
        
    except Exception as e:
        raise RuntimeError(
            f"计算合成可及性分数时出错: {str(e)}\n"
            f"分子SMILES: {Chem.MolToSmiles(mol)}"
        )



if __name__ == "__main__":
    def example_basic():
        """示例1：基本使用"""
        print("="*70)
        print("示例1：基本3D生成和优化")
        print("="*70)
        
        # 创建生成器
        # smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # 阿司匹林 CC(C)Cc1ccc(cc1)C(C)C(O)=O
        smiles = 'aspirin'
        generator = SMILES3DGenerator(smiles)
        
        # 步骤1: 解析SMILES
        generator.parse_smiles()
        
        # 步骤2: 生成3D坐标
        generator.generate_3d(method='ETKDGv3')
        
        # 步骤3: 能量最小化
        generator.optimize_geometry(force_field='MMFF')
        
        # 步骤4: 计算3D性质
        props = generator.get_3d_properties()
        
        # 步骤5: 保存文件
        generator.save_to_file('aspirin_3d.mol', 'mol')
        generator.save_to_file('aspirin_3d.pdb', 'pdb')
        generator.save_to_file('aspirin_3d.xyz', 'xyz')
        
        return generator


    def example_multiple_conformers():
        """示例2：多构象生成"""
        print("\n" + "="*70)
        print("示例2：多构象生成和能量比较")
        print("="*70)
        
        smiles = 'aspirin'  # 辛烷（柔性分子）
        generator = SMILES3DGenerator(smiles)
        
        # 生成多个构象
        generator.parse_smiles()
        generator.generate_3d(method='ETKDG', num_confs=10)
        generator.optimize_geometry() 

        # 找到能量最低的构象
        min_energy_id = generator.energy_after.index(min(generator.energy_after))
        print(f"\n最低能量构象: {min_energy_id}")
        print(f"能量: {generator.energy_after[min_energy_id]:.2f} kcal/mol")
        
        # 保存最低能量构象
        generator.save_to_file('octane_lowest_energy.mol', 'mol', conf_id=min_energy_id)
        
        return generator


    def example_comparison():
        """示例3：比较不同方法"""
        print("\n" + "="*70)
        print("示例3：比较不同3D生成方法")
        print("="*70)
        
        smiles = 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'  # 布洛芬
        
        methods = ['ETKDG', 'basic']
        force_fields = ['MMFF', 'UFF']
        
        for method in methods:
            for ff in force_fields:
                print(f"\n【{method} + {ff}】")
                try:
                    gen = SMILES3DGenerator(smiles)
                    gen.parse_smiles()
                    gen.generate_3d(method=method)
                    gen.optimize_geometry(force_field=ff)
                    print(f"  最终能量: {gen.energy_after[0]:.2f} kcal/mol")
                except Exception as e:
                    print(f"  失败: {e}")

    # example_multiple_conformers() 
    # ============ 使用示例 ============

    # 分析金刚烷酯
    print("\n图片中的分子:")
    info1 = mol_basic_physicochemical_info('C=CC(=O)OC12CC3CC(CC(C3)C1)C2')
    # 分析布洛芬
    print("\n\n布洛芬:")
    info2 = mol_basic_physicochemical_info('CC(C)Cc1ccc(cc1)C(C)C(=O)O')  
    print("对比分析： ")
    print(info1)
    print(info2)
