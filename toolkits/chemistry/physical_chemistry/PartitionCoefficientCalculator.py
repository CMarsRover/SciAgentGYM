# partition_coefficient.py

import numpy as np
import argparse
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToImage

class PartitionCoefficientCalculator:
    
    def __init__(self, temperature=298.15):
        
        self.temperature = temperature
        self.R = 8.314  # 气体常数，J/(mol·K)
        
    def calculate_ionization_fraction(self, pH, pKa):
        
        if isinstance(pKa, (list, tuple)):
            return [1.0 / (1.0 + 10**(pKa_i - pH)) for pKa_i in pKa]
        else:
            # 对于酸性基团，电离度 = [A-]/([HA]+[A-])
            return 1.0 / (1.0 + 10**(pKa - pH))
    
    def calculate_apparent_logP(self, logP_neutral, pH, pKa, acidic=True):
        
        if acidic:
            # 对于酸性基团(如羧酸)
            ionization_fraction = self.calculate_ionization_fraction(pH, pKa)
            # 电离形式通常更亲水，因此表观logP降低
            logP_app = logP_neutral + np.log10(1 - ionization_fraction)
        else:
            # 对于碱性基团(如胺)
            ionization_fraction = 1.0 / (1.0 + 10**(pH - pKa))
            # 电离形式通常更亲水，因此表观logP降低
            logP_app = logP_neutral + np.log10(1 - ionization_fraction)
            
        return logP_app
    
    def calculate_distribution_coefficient(self, logP_app):
        
        return 10**logP_app
    
    def predict_logP_from_structure(self, smiles):
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无法从SMILES '{smiles}'创建分子")
        return Crippen.MolLogP(mol)
    
    def analyze_partition_mechanism(self, compound_name, smiles, logP_app, pH, pKa):
        """
        分析在给定 pH 和表观 logP 条件下的分配机制。

        设计目标：
        - 在输入不完整或无效时，不抛异常，而是返回结构化的提示信息，便于上层 LLM 纠错。
        """
        issues = []

        # 基本输入检查
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            issues.append("invalid_smiles")

        if logP_app is None or not isinstance(logP_app, (int, float)):
            issues.append("logP_app_missing_or_invalid")

        if pKa is None or not isinstance(pKa, (int, float)):
            issues.append("pKa_missing_or_invalid")

        if pH is None or not isinstance(pH, (int, float)):
            issues.append("pH_missing_or_invalid")

        # 如果有严重缺失，直接返回提示，不再继续计算
        if issues:
            return {
                "ok": False,
                "compound": compound_name,
                "smiles": smiles,
                "logP_apparent": logP_app,
                "pH": pH,
                "pKa": pKa,
                "issues": issues,
                "message": (
                    "analyze_partition_mechanism 需要有效的 SMILES、数值类型的 logP_app 和 pKa "
                    "才能进行机理分析；当前输入不完整或无效，请先使用其他工具计算/填充这些字段。"
                ),
            }

        # 计算分子描述符
        mw = Descriptors.MolWt(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)

        # 电离状态分析
        ionization_fraction = self.calculate_ionization_fraction(pH, pKa)

        # 分配行为分析
        if logP_app > 0:
            partition_preference = "疏水性，倾向于分配到有机相(正辛醇)"
        else:
            partition_preference = "亲水性，倾向于分配到水相"

        # 分子机制解释
        if ionization_fraction > 0.5:
            if logP_app < 0:
                mechanism = "在给定pH条件下，分子高度电离，带电荷形式优先分配到水相"
            else:
                mechanism = "尽管分子高度电离，但其疏水骨架使其仍倾向于分配到有机相"
        else:
            if logP_app > 0:
                mechanism = "分子主要以中性形式存在，疏水性使其优先分配到有机相"
            else:
                mechanism = "尽管分子主要以中性形式存在，但极性基团使其仍倾向于分配到水相"

        return {
            "ok": True,
            "compound": compound_name,
            "molecular_weight": mw,
            "hydrogen_bond_acceptors": hba,
            "hydrogen_bond_donors": hbd,
            "topological_polar_surface_area": tpsa,
            "ionization_fraction": ionization_fraction,
            "logP_apparent": logP_app,
            "partition_coefficient": 10**logP_app,
            "partition_preference": partition_preference,
            "molecular_mechanism": mechanism,
        }
    
    def visualize_pH_logP_profile(self, logP_neutral, pKa, pH_range=(0, 14), acidic=True, compound_name=""):
        
        pH_values = np.linspace(pH_range[0], pH_range[1], 100)
        logP_values = []
        
        for pH in pH_values:
            logP_app = self.calculate_apparent_logP(logP_neutral, pH, pKa, acidic)
            logP_values.append(logP_app)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pH_values, logP_values, 'b-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=pKa, color='r', linestyle='--', label=f'pKa = {pKa}')
        
        ax.set_xlabel('pH', fontsize=12)
        ax.set_ylabel('logP(apparent)', fontsize=12)
        ax.set_title(f'{compound_name} pH-logP Profile', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig

def parse_args():
    
    parser = argparse.ArgumentParser(description='计算和分析两相系统中的分配系数')
    parser.add_argument('--compound', type=str, required=True,
                        help='化合物名称')
    parser.add_argument('--smiles', type=str, required=True,
                        help='化合物的SMILES结构')
    parser.add_argument('--logP', type=float, required=True,
                        help='中性分子的logP值')
    parser.add_argument('--pKa', type=float, required=True,
                        help='化合物的pKa值')
    parser.add_argument('--pH', type=float, default=7.4,
                        help='溶液的pH值(默认为7.4)')
    parser.add_argument('--temperature', type=float, default=298.15,
                        help='系统温度，单位为K(默认为298.15K)')
    parser.add_argument('--acidic', action='store_true',
                        help='指示化合物是否含有酸性基团')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成pH-logP曲线图')
    parser.add_argument('--output', type=str, default='partition_analysis.png',
                        help='输出图表的文件名(默认为partition_analysis.png)')
    
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    calculator = PartitionCoefficientCalculator(temperature=args.temperature)
    
    # 计算表观logP
    logP_app = calculator.calculate_apparent_logP(
        args.logP, args.pH, args.pKa, acidic=args.acidic
    )
    
    # 计算分配系数
    K_ow = calculator.calculate_distribution_coefficient(logP_app)
    
    # 分析分配机制
    analysis = calculator.analyze_partition_mechanism(
        args.compound, args.smiles, logP_app, args.pH, args.pKa
    )
    
    # 打印结果
    print(f"化合物: {args.compound}")
    print(f"pH: {args.pH}")
    print(f"表观logP: {logP_app:.4f}")
    print(f"分配系数K_ow: {K_ow:.4f}")
    print(f"分配偏好: {analysis['partition_preference']}")
    print(f"分子机制: {analysis['molecular_mechanism']}")
    
    # 可视化
    if args.visualize:
        fig = calculator.visualize_pH_logP_profile(
            args.logP, args.pKa, acidic=args.acidic, compound_name=args.compound
        )
        fig.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"pH-logP曲线已保存为 {args.output}")

if __name__ == "__main__":
    main()
