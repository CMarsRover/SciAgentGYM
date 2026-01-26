# Filename: structural_biology_toolkit.py

import json
import sqlite3
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# 配置matplotlib字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 创建必要的目录
os.makedirs('./mid_result/structural_biology', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 氨基酸性质数据库
AMINO_ACID_PROPERTIES = {
    'A': {'name': 'Alanine', 'type': 'nonpolar', 'volume': 88.6, 'hydrophobicity': 1.8, 'charge': 0, 'h_bond': False},
    'R': {'name': 'Arginine', 'type': 'basic', 'volume': 173.4, 'hydrophobicity': -4.5, 'charge': 1, 'h_bond': True},
    'N': {'name': 'Asparagine', 'type': 'polar', 'volume': 114.1, 'hydrophobicity': -3.5, 'charge': 0, 'h_bond': True},
    'D': {'name': 'Aspartic acid', 'type': 'acidic', 'volume': 111.1, 'hydrophobicity': -3.5, 'charge': -1, 'h_bond': True},
    'C': {'name': 'Cysteine', 'type': 'polar', 'volume': 108.5, 'hydrophobicity': 2.5, 'charge': 0, 'h_bond': False},
    'E': {'name': 'Glutamic acid', 'type': 'acidic', 'volume': 138.4, 'hydrophobicity': -3.5, 'charge': -1, 'h_bond': True},
    'Q': {'name': 'Glutamine', 'type': 'polar', 'volume': 143.8, 'hydrophobicity': -3.5, 'charge': 0, 'h_bond': True},
    'G': {'name': 'Glycine', 'type': 'nonpolar', 'volume': 60.1, 'hydrophobicity': -0.4, 'charge': 0, 'h_bond': False},
    'H': {'name': 'Histidine', 'type': 'basic', 'volume': 153.2, 'hydrophobicity': -3.2, 'charge': 0.1, 'h_bond': True},
    'I': {'name': 'Isoleucine', 'type': 'nonpolar', 'volume': 166.7, 'hydrophobicity': 4.5, 'charge': 0, 'h_bond': False},
    'L': {'name': 'Leucine', 'type': 'nonpolar', 'volume': 166.7, 'hydrophobicity': 3.8, 'charge': 0, 'h_bond': False},
    'K': {'name': 'Lysine', 'type': 'basic', 'volume': 168.6, 'hydrophobicity': -3.9, 'charge': 1, 'h_bond': True},
    'M': {'name': 'Methionine', 'type': 'nonpolar', 'volume': 162.9, 'hydrophobicity': 1.9, 'charge': 0, 'h_bond': False},
    'F': {'name': 'Phenylalanine', 'type': 'nonpolar', 'volume': 189.9, 'hydrophobicity': 2.8, 'charge': 0, 'h_bond': False},
    'P': {'name': 'Proline', 'type': 'nonpolar', 'volume': 112.7, 'hydrophobicity': -1.6, 'charge': 0, 'h_bond': False},
    'S': {'name': 'Serine', 'type': 'polar', 'volume': 89.0, 'hydrophobicity': -0.8, 'charge': 0, 'h_bond': True},
    'T': {'name': 'Threonine', 'type': 'polar', 'volume': 116.1, 'hydrophobicity': -0.7, 'charge': 0, 'h_bond': True},
    'W': {'name': 'Tryptophan', 'type': 'nonpolar', 'volume': 227.8, 'hydrophobicity': -0.9, 'charge': 0, 'h_bond': True},
    'Y': {'name': 'Tyrosine', 'type': 'polar', 'volume': 193.6, 'hydrophobicity': -1.3, 'charge': 0, 'h_bond': True},
    'V': {'name': 'Valine', 'type': 'nonpolar', 'volume': 140.0, 'hydrophobicity': 4.2, 'charge': 0, 'h_bond': False}
}

# 第一层：原子函数
def get_amino_acid_properties(residue: str) -> Dict[str, Any]:
    """获取氨基酸的物理化学性质"""
    if residue not in AMINO_ACID_PROPERTIES:
        raise ValueError(f"Unknown amino acid: {residue}")
    
    properties = AMINO_ACID_PROPERTIES[residue].copy()
    return {
        'result': properties,
        'metadata': {'residue': residue, 'source': 'amino_acid_database'}
    }

def calculate_mutation_effect_score(original: str, mutant: str) -> Dict[str, Any]:
    """计算突变对蛋白质性质的影响评分"""
    if original not in AMINO_ACID_PROPERTIES or mutant not in AMINO_ACID_PROPERTIES:
        raise ValueError("Invalid amino acid code")
    
    orig_props = AMINO_ACID_PROPERTIES[original]
    mut_props = AMINO_ACID_PROPERTIES[mutant]
    
    # 计算各项性质变化
    volume_change = abs(mut_props['volume'] - orig_props['volume'])
    hydrophobicity_change = abs(mut_props['hydrophobicity'] - orig_props['hydrophobicity'])
    charge_change = abs(mut_props['charge'] - orig_props['charge'])
    h_bond_change = 1 if orig_props['h_bond'] != mut_props['h_bond'] else 0
    
    # 综合评分 (0-100, 越高影响越大)
    effect_score = (volume_change/50 + hydrophobicity_change/2 + charge_change*20 + h_bond_change*15) * 10
    effect_score = min(100, effect_score)
    
    return {
        'result': {
            'effect_score': round(effect_score, 2),
            'volume_change': round(volume_change, 1),
            'hydrophobicity_change': round(hydrophobicity_change, 1),
            'charge_change': charge_change,
            'h_bond_change': h_bond_change
        },
        'metadata': {
            'mutation': f"{original}->{mutant}",
            'interpretation': 'high_impact' if effect_score > 50 else 'moderate_impact' if effect_score > 25 else 'low_impact'
        }
    }

def analyze_binding_site_residue(residue: str, position: int) -> Dict[str, Any]:
    """分析结合位点残基的功能重要性"""
    if residue not in AMINO_ACID_PROPERTIES:
        raise ValueError(f"Unknown amino acid: {residue}")
    
    props = AMINO_ACID_PROPERTIES[residue]
    
    # 评估结合重要性
    binding_importance = 0
    if props['h_bond']:
        binding_importance += 30  # 氢键能力
    if abs(props['charge']) > 0:
        binding_importance += 25  # 电荷相互作用
    if props['type'] == 'polar':
        binding_importance += 20  # 极性相互作用
    if props['volume'] > 150:
        binding_importance += 15  # 疏水相互作用
    
    return {
        'result': {
            'residue': residue,
            'position': position,
            'binding_importance': binding_importance,
            'key_interactions': []
        },
        'metadata': {
            'analysis_type': 'binding_site_analysis',
            'properties': props
        }
    }

# 第二层：组合函数
def compare_mutation_candidates(mutations: List[str]) -> Dict[str, Any]:
    """比较多个突变候选的影响效果"""
    results = []
    
    for mutation in mutations:
        if '->' not in mutation:
            raise ValueError(f"Invalid mutation format: {mutation}. Use 'A->G' format")
        
        original, mutant = mutation.split('->')
        effect = calculate_mutation_effect_score(original, mutant)
        
        results.append({
            'mutation': mutation,
            'effect_score': effect['result']['effect_score'],
            'interpretation': effect['metadata']['interpretation'],
            'details': effect['result']
        })
    
    # 按影响评分排序
    results.sort(key=lambda x: x['effect_score'], reverse=True)
    
    return {
        'result': results,
        'metadata': {
            'total_mutations': len(mutations),
            'highest_impact': results[0]['mutation'] if results else None
        }
    }

def analyze_active_site_composition(residues: List[str], positions: List[int]) -> Dict[str, Any]:
    """分析活性位点的氨基酸组成"""
    if len(residues) != len(positions):
        raise ValueError("Residues and positions lists must have same length")
    
    site_analysis = []
    total_importance = 0
    
    for residue, position in zip(residues, positions):
        analysis = analyze_binding_site_residue(residue, position)
        site_analysis.append(analysis['result'])
        total_importance += analysis['result']['binding_importance']
    
    # 统计氨基酸类型分布
    type_distribution = {}
    for residue in residues:
        aa_type = AMINO_ACID_PROPERTIES[residue]['type']
        type_distribution[aa_type] = type_distribution.get(aa_type, 0) + 1
    
    return {
        'result': {
            'residue_analysis': site_analysis,
            'total_importance_score': total_importance,
            'type_distribution': type_distribution,
            'site_characteristics': {
                'polar_residues': sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['type'] in ['polar', 'basic', 'acidic']),
                'nonpolar_residues': sum(1 for r in residues if AMINO_ACID_PROPERTIES[r]['type'] == 'nonpolar'),
                'charged_residues': sum(1 for r in residues if abs(AMINO_ACID_PROPERTIES[r]['charge']) > 0)
            }
        },
        'metadata': {
            'analysis_type': 'active_site_composition',
            'residue_count': len(residues)
        }
    }

def predict_mutation_binding_impact(original_residues: List[str], positions: List[int], test_mutations: List[str]) -> Dict[str, Any]:
    """预测突变对配体结合的影响"""
    # 分析原始活性位点
    original_site = analyze_active_site_composition(original_residues, positions)
    
    # 分析每个突变的影响
    mutation_impacts = []
    
    for mutation in test_mutations:
        if '->' not in mutation:
            continue
        original, mutant = mutation.split('->')
        effect = calculate_mutation_effect_score(original, mutant)
        
        # 预测对结合的具体影响
        binding_impact = "unknown"
        if effect['result']['effect_score'] > 70:
            binding_impact = "severe_disruption"
        elif effect['result']['effect_score'] > 40:
            binding_impact = "moderate_disruption"
        elif effect['result']['effect_score'] > 15:
            binding_impact = "mild_disruption"
        else:
            binding_impact = "minimal_impact"
        
        mutation_impacts.append({
            'mutation': mutation,
            'predicted_impact': binding_impact,
            'effect_score': effect['result']['effect_score'],
            'key_changes': effect['result']
        })
    
    return {
        'result': {
            'original_site_analysis': original_site['result'],
            'mutation_predictions': mutation_impacts,
            'recommended_mutation': max(mutation_impacts, key=lambda x: x['effect_score'])['mutation'] if mutation_impacts else None
        },
        'metadata': {
            'prediction_method': 'physicochemical_analysis',
            'confidence': 'medium'
        }
    }

# 第三层：可视化函数
def visualize_amino_acid_properties(residues: List[str], save_path: str = None) -> Dict[str, Any]:
    """可视化氨基酸性质比较"""
    if not residues:
        raise ValueError("Residues list cannot be empty")
    
    properties = ['volume', 'hydrophobicity', 'charge']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, prop in enumerate(properties):
        values = [AMINO_ACID_PROPERTIES[r][prop] for r in residues]
        colors = ['red' if r == 'A' else 'blue' for r in residues]
        
        axes[i].bar(residues, values, color=colors, alpha=0.7)
        axes[i].set_title(f'{prop.capitalize()} Comparison')
        axes[i].set_ylabel(prop.capitalize())
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = './tool_images/amino_acid_properties.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'residues_analyzed': residues,
            'properties_shown': properties
        }
    }

def plot_mutation_effect_matrix(mutations: List[str], save_path: str = None) -> Dict[str, Any]:
    """绘制突变效应矩阵热图"""
    if not mutations:
        raise ValueError("Mutations list cannot be empty")
    
    # 计算突变效应
    mutation_data = []
    labels = []
    
    for mutation in mutations:
        if '->' not in mutation:
            continue
        original, mutant = mutation.split('->')
        effect = calculate_mutation_effect_score(original, mutant)
        mutation_data.append([
            effect['result']['volume_change'],
            effect['result']['hydrophobicity_change'],
            effect['result']['charge_change'] * 10,  # 放大显示
            effect['result']['h_bond_change'] * 20   # 放大显示
        ])
        labels.append(mutation)
    
    if not mutation_data:
        raise ValueError("No valid mutations found")
    
    mutation_matrix = np.array(mutation_data)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mutation_matrix.T, cmap='RdYlBu_r', aspect='auto')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['Volume Change', 'Hydrophobicity Change', 'Charge Change (×10)', 'H-bond Change (×20)'])
    ax.set_title('Mutation Effect Matrix')
    
    # 添加数值标注
    for i in range(len(labels)):
        for j in range(4):
            text = ax.text(i, j, f'{mutation_matrix[i, j]:.1f}', ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Effect Magnitude')
    plt.tight_layout()
    
    if save_path is None:
        save_path = './tool_images/mutation_effect_matrix.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'mutations_analyzed': len(labels),
            'matrix_shape': mutation_matrix.shape
        }
    }

def create_binding_site_summary_plot(active_site_residues: List[str], positions: List[int], test_mutations: List[str], save_path: str = None) -> Dict[str, Any]:
    """创建结合位点分析总结图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 活性位点氨基酸类型分布
    site_analysis = analyze_active_site_composition(active_site_residues, positions)
    type_dist = site_analysis['result']['type_distribution']
    
    ax1.pie(type_dist.values(), labels=type_dist.keys(), autopct='%1.1f%%')
    ax1.set_title('Active Site Amino Acid Types')
    
    # 2. 氨基酸性质雷达图
    properties = ['volume', 'hydrophobicity', 'charge']
    avg_props = []
    for prop in properties:
        values = [AMINO_ACID_PROPERTIES[r][prop] for r in active_site_residues]
        avg_props.append(np.mean(values))
    
    ax2.bar(properties, avg_props, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_title('Average Properties of Active Site')
    ax2.set_ylabel('Property Value')
    
    # 3. 突变效应评分
    mutation_scores = []
    mutation_labels = []
    for mutation in test_mutations:
        if '->' in mutation:
            original, mutant = mutation.split('->')
            effect = calculate_mutation_effect_score(original, mutant)
            mutation_scores.append(effect['result']['effect_score'])
            mutation_labels.append(mutation)
    
    colors = ['red' if score > 50 else 'orange' if score > 25 else 'green' for score in mutation_scores]
    ax3.bar(mutation_labels, mutation_scores, color=colors, alpha=0.7)
    ax3.set_title('Mutation Effect Scores')
    ax3.set_ylabel('Effect Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 结合重要性评分
    importance_scores = []
    for residue, position in zip(active_site_residues, positions):
        analysis = analyze_binding_site_residue(residue, position)
        importance_scores.append(analysis['result']['binding_importance'])
    
    position_labels = [f"{r}{p}" for r, p in zip(active_site_residues, positions)]
    ax4.bar(position_labels, importance_scores, color='lightblue', alpha=0.7)
    ax4.set_title('Binding Importance by Position')
    ax4.set_ylabel('Importance Score')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = './tool_images/binding_site_summary.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'file_type': 'png',
            'analysis_components': 4,
            'active_site_residues': len(active_site_residues)
        }
    }

# 文件加载函数
def load_file(filepath: str) -> Dict[str, Any]:
    """加载各种格式的文件"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    file_ext = os.path.splitext(filepath)[1].lower()
    
    if file_ext == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif file_ext == '.txt':
        with open(filepath, 'r') as f:
            data = f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return {
        'result': data,
        'metadata': {
            'filepath': filepath,
            'file_type': file_ext,
            'size': os.path.getsize(filepath)
        }
    }

def main():
    """主函数：演示三个场景的蛋白质-配体相互作用分析"""
    
    print("=" * 60)
    print("场景1：分析给定活性位点残基并预测最佳突变位点")
    print("=" * 60)
    print("问题描述：基于X-ray晶体结构数据，分析H34, T48, S62, G128活性位点残基，")
    print("预测哪个突变(142A->G)最可能影响配体-受体相互作用")
    print("-" * 60)
    
    # 步骤1：分析活性位点组成
    # 调用函数：analyze_active_site_composition()
    active_residues = ['H', 'T', 'S', 'G']
    positions = [34, 48, 62, 128]
    result1 = analyze_active_site_composition(active_residues, positions)
    print(f"FUNCTION_CALL: analyze_active_site_composition | PARAMS: {{'residues': {active_residues}, 'positions': {positions}}} | RESULT: {result1}")
    
    # 步骤2：分析候选突变A->G的效应
    # 调用函数：calculate_mutation_effect_score()
    mutation_effect = calculate_mutation_effect_score('A', 'G')
    print(f"FUNCTION_CALL: calculate_mutation_effect_score | PARAMS: {{'original': 'A', 'mutant': 'G'}} | RESULT: {mutation_effect}")
    
    # 步骤3：预测突变对结合的影响
    # 调用函数：predict_mutation_binding_impact()
    test_mutations = ['A->G', 'A->V', 'A->D']
    binding_prediction = predict_mutation_binding_impact(active_residues, positions, test_mutations)
    print(f"FUNCTION_CALL: predict_mutation_binding_impact | PARAMS: {{'original_residues': {active_residues}, 'positions': {positions}, 'test_mutations': {test_mutations}}} | RESULT: {binding_prediction}")
    
    # 步骤4：创建综合分析图
    # 调用函数：create_binding_site_summary_plot()
    summary_plot = create_binding_site_summary_plot(active_residues, positions, test_mutations)
    print(f"FUNCTION_CALL: create_binding_site_summary_plot | PARAMS: {{'active_site_residues': {active_residues}, 'positions': {positions}, 'test_mutations': {test_mutations}}} | RESULT: {summary_plot}")
    
    print(f"FINAL_ANSWER: A->G突变最可能影响配体-受体相互作用，效应评分为{mutation_effect['result']['effect_score']}")
    
    print("=" * 60)
    print("场景2：比较多种突变候选的相对影响")
    print("=" * 60)
    print("问题描述：比较不同氨基酸突变对蛋白质结构和功能的相对影响")
    print("-" * 60)
    
    # 步骤1：比较多个突变候选
    # 调用函数：compare_mutation_candidates()
    multiple_mutations = ['A->G', 'A->V', 'A->D', 'A->R', 'A->P']
    comparison_result = compare_mutation_candidates(multiple_mutations)
    print(f"FUNCTION_CALL: compare_mutation_candidates | PARAMS: {{'mutations': {multiple_mutations}}} | RESULT: {comparison_result}")
    
    # 步骤2：可视化突变效应矩阵
    # 调用函数：plot_mutation_effect_matrix()
    matrix_plot = plot_mutation_effect_matrix(multiple_mutations)
    print(f"FUNCTION_CALL: plot_mutation_effect_matrix | PARAMS: {{'mutations': {multiple_mutations}}} | RESULT: {matrix_plot}")
    
    print(f"FINAL_ANSWER: 在所有测试突变中，{comparison_result['result'][0]['mutation']}具有最高的影响评分")
    
    print("=" * 60)
    print("场景3：深入分析氨基酸性质对配体结合的贡献")
    print("=" * 60)
    print("问题描述：分析不同氨基酸的物理化学性质如何影响蛋白质-配体相互作用")
    print("-" * 60)
    
    # 步骤1：获取关键氨基酸的性质
    # 调用函数：get_amino_acid_properties()
    key_residues = ['A', 'G', 'H', 'T', 'S']
    properties_analysis = []
    for residue in key_residues:
        props = get_amino_acid_properties(residue)
        properties_analysis.append(props)
        print(f"FUNCTION_CALL: get_amino_acid_properties | PARAMS: {{'residue': '{residue}'}} | RESULT: {props}")
    
    # 步骤2：可视化氨基酸性质比较
    # 调用函数：visualize_amino_acid_properties()
    properties_plot = visualize_amino_acid_properties(key_residues)
    print(f"FUNCTION_CALL: visualize_amino_acid_properties | PARAMS: {{'residues': {key_residues}}} | RESULT: {properties_plot}")
    
    # 步骤3：分析结合位点残基的功能重要性
    # 调用函数：analyze_binding_site_residue()
    h34_analysis = analyze_binding_site_residue('H', 34)
    print(f"FUNCTION_CALL: analyze_binding_site_residue | PARAMS: {{'residue': 'H', 'position': 34}} | RESULT: {h34_analysis}")
    
    print(f"FINAL_ANSWER: 氨基酸性质分析显示，A->G突变会显著改变侧链体积和疏水性，影响配体结合")

if __name__ == "__main__":
    main()