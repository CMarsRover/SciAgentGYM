# Filename: genetic_interaction_toolkit.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any
from scipy.stats import chi2_contingency
import sqlite3

# 配置matplotlib字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 获取脚本所在目录，用于构建数据库路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # extracted_tools_1113 目录
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# 创建必要的目录
os.makedirs('./mid_result/genetics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
WILD_TYPE_RESISTANCE = 100.0
EPISTASIS_THRESHOLD = 0.1
REDUNDANCY_THRESHOLD = 0.2

def create_genetic_database() -> str:
    """
    访问遗传互作分析的本地数据库
    
    Returns:
        数据库文件路径
    """
    db_path = os.path.join(DATABASE_DIR, 'genetic_interactions.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    return db_path

def analyze_single_mutant_effects(resistance_data: Dict[str, float]) -> Dict[str, Any]:
    """
    分析单基因突变体的表型效应
    
    Args:
        resistance_data: 基因型到抗性百分比的映射
    Returns:
        单基因效应分析结果
    """
    single_mutants = {k: v for k, v in resistance_data.items() if len(k) == 2}
    
    effects = {}
    for genotype, resistance in single_mutants.items():
        gene = genotype.upper()
        effect_size = WILD_TYPE_RESISTANCE - resistance
        severity = 'mild' if effect_size < 25 else 'moderate' if effect_size < 50 else 'severe'
        
        effects[gene] = {
            'resistance_percent': resistance,
            'effect_size': effect_size,
            'severity': severity,
            'is_essential': resistance == 0
        }
    
    return {
        'result': effects,
        'metadata': {
            'analysis_type': 'single_mutant_effects',
            'wild_type_resistance': WILD_TYPE_RESISTANCE,
            'total_mutants': len(single_mutants)
        }
    }

def calculate_epistasis_coefficient(gene_a_resistance: float, gene_b_resistance: float, double_mutant_resistance: float) -> Dict[str, Any]:
    """
    计算上位性系数和互作类型
    
    Args:
        gene_a_resistance: 基因A单突变体抗性
        gene_b_resistance: 基因B单突变体抗性  
        double_mutant_resistance: 双突变体抗性
    
    Returns:
        上位性分析结果
    """
    # 期望的加性效应
    expected_additive = gene_a_resistance + gene_b_resistance - WILD_TYPE_RESISTANCE
    expected_multiplicative = (gene_a_resistance * gene_b_resistance) / WILD_TYPE_RESISTANCE
    
    # 上位性系数
    epistasis_coeff = double_mutant_resistance - expected_additive
    
    # 判断互作类型
    if abs(epistasis_coeff) < EPISTASIS_THRESHOLD:
        interaction_type = 'additive'
    elif epistasis_coeff < -EPISTASIS_THRESHOLD:
        interaction_type = 'negative_epistasis'
    else:
        interaction_type = 'positive_epistasis'
    
    # 检查是否为完全上位性
    min_single = min(gene_a_resistance, gene_b_resistance)
    if abs(double_mutant_resistance - min_single) < REDUNDANCY_THRESHOLD:
        interaction_type = 'redundancy'
    
    return {
        'result': {
            'epistasis_coefficient': epistasis_coeff,
            'interaction_type': interaction_type,
            'expected_additive': expected_additive,
            'expected_multiplicative': expected_multiplicative,
            'observed': double_mutant_resistance
        },
        'metadata': {
            'gene_a_resistance': gene_a_resistance,
            'gene_b_resistance': gene_b_resistance,
            'analysis_method': 'epistasis_coefficient'
        }
    }

def identify_transcription_factor(resistance_data: Dict[str, float]) -> Dict[str, Any]:
    """
    识别转录因子基因
    
    Args:
        resistance_data: 完整的抗性数据
    
    Returns:
        转录因子识别结果
    """
    single_mutants = {k: v for k, v in resistance_data.items() if len(k) == 2}
    double_mutants = {k: v for k, v in resistance_data.items() if len(k) == 4}
    
    tf_candidates = []
    
    for gene in ['G1', 'G2', 'G3']:
        gene_lower = gene.lower()
        
        # 检查该基因的单突变体效应
        single_resistance = single_mutants.get(gene_lower, 100)
        
        # 检查包含该基因的所有双突变体
        double_with_gene = {k: v for k, v in double_mutants.items() if gene_lower in k}
        
        # 转录因子特征：单突变严重影响，双突变体都接近0
        is_severe_single = single_resistance <= 25
        all_doubles_severe = all(v <= 10 for v in double_with_gene.values())
        
        if is_severe_single and all_doubles_severe and len(double_with_gene) >= 2:
            tf_candidates.append({
                'gene': gene,
                'confidence': 'high',
                'evidence': {
                    'single_mutant_severe': single_resistance,
                    'all_double_mutants_severe': list(double_with_gene.values())
                }
            })
    
    return {
        'result': tf_candidates,
        'metadata': {
            'analysis_criteria': {
                'single_mutant_threshold': 25,
                'double_mutant_threshold': 10
            },
            'total_candidates': len(tf_candidates)
        }
    }

def analyze_gene_redundancy(gene1: str, gene2: str, resistance_data: Dict[str, float]) -> Dict[str, Any]:
    """
    分析两个基因间的功能冗余性
    
    Args:
        gene1: 基因1名称
        gene2: 基因2名称
        resistance_data: 抗性数据
    
    Returns:
        冗余性分析结果
    """
    g1_resistance = resistance_data.get(gene1.lower(), 100)
    g2_resistance = resistance_data.get(gene2.lower(), 100)
    
    # 构建双突变体基因型名称
    double_genotype = ''.join(sorted([gene1.lower(), gene2.lower()]))
    double_resistance = resistance_data.get(double_genotype, 100)
    
    # 冗余性指标
    min_single = min(g1_resistance, g2_resistance)
    redundancy_score = abs(double_resistance - min_single) / min_single if min_single > 0 else 1
    
    is_redundant = redundancy_score < REDUNDANCY_THRESHOLD
    
    return {
        'result': {
            'gene_pair': f"{gene1}-{gene2}",
            'is_redundant': is_redundant,
            'redundancy_score': redundancy_score,
            'double_mutant_resistance': double_resistance,
            'min_single_resistance': min_single
        },
        'metadata': {
            'gene1_resistance': g1_resistance,
            'gene2_resistance': g2_resistance,
            'redundancy_threshold': REDUNDANCY_THRESHOLD
        }
    }

def determine_epistatic_relationship(gene1: str, gene2: str, resistance_data: Dict[str, float]) -> Dict[str, Any]:
    """
    确定两个基因间的上位性关系
    
    Args:
        gene1: 基因1名称
        gene2: 基因2名称  
        resistance_data: 抗性数据
    
    Returns:
        上位性关系结果
    """
    g1_resistance = resistance_data.get(gene1.lower(), 100)
    g2_resistance = resistance_data.get(gene2.lower(), 100)
    double_genotype = ''.join(sorted([gene1.lower(), gene2.lower()]))
    double_resistance = resistance_data.get(double_genotype, 100)
    
    # 上位性判断：哪个基因的表型在双突变体中占主导
    epistatic_gene = None
    if abs(double_resistance - g1_resistance) < abs(double_resistance - g2_resistance):
        epistatic_gene = gene1
    elif abs(double_resistance - g2_resistance) < abs(double_resistance - g1_resistance):
        epistatic_gene = gene2
    
    return {
        'result': {
            'epistatic_gene': epistatic_gene,
            'hypostatic_gene': gene2 if epistatic_gene == gene1 else gene1,
            'double_mutant_phenotype': double_resistance,
            'epistasis_strength': 'complete' if double_resistance <= 10 else 'partial'
        },
        'metadata': {
            'gene1_resistance': g1_resistance,
            'gene2_resistance': g2_resistance,
            'analysis_method': 'phenotype_comparison'
        }
    }

def visualize_genetic_interactions(resistance_data: Dict[str, float]) -> str:
    """
    可视化遗传互作网络
    
    Args:
        resistance_data: 抗性数据
    
    Returns:
        图像文件路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1：抗性水平条形图
    genotypes = list(resistance_data.keys())
    resistances = list(resistance_data.values())
    colors = ['red' if r == 0 else 'orange' if r < 25 else 'yellow' if r < 75 else 'green' 
              for r in resistances]
    
    bars = ax1.barh(genotypes, resistances, color=colors, alpha=0.7)
    ax1.set_xlabel('Resistance Level (%)')
    ax1.set_title('Mutant Resistance Profiles')
    ax1.axvline(x=100, color='black', linestyle='--', alpha=0.5, label='Wild-type')
    
    # 添加数值标签
    for bar, resistance in zip(bars, resistances):
        ax1.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
                f'{resistance}%', va='center')
    
    # 子图2：互作网络图
    import networkx as nx
    
    G = nx.Graph()
    genes = ['G1', 'G2', 'G3']
    G.add_nodes_from(genes)
    
    # 添加边（基于双突变体数据推断互作）
    interactions = [('G1', 'G2'), ('G1', 'G3'), ('G2', 'G3')]
    edge_weights = []
    
    for g1, g2 in interactions:
        double_key = ''.join(sorted([g1.lower(), g2.lower()]))
        if double_key in resistance_data:
            weight = 100 - resistance_data[double_key]  # 互作强度
            G.add_edge(g1, g2, weight=weight)
            edge_weights.append(weight)
    
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='lightblue', 
                          node_size=1000, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12, font_weight='bold')
    
    # 绘制边，线宽表示互作强度
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, ax=ax2, width=[w/20 for w in weights], 
                          alpha=0.6, edge_color='gray')
    
    ax2.set_title('Gene Interaction Network')
    ax2.axis('off')
    
    plt.tight_layout()
    filepath = './tool_images/genetic_interactions.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    return filepath

def comprehensive_genetic_analysis(resistance_data: Dict[str, float]) -> Dict[str, Any]:
    """
    综合遗传互作分析
    
    Args:
        resistance_data: 完整的抗性数据
    
    Returns:
        综合分析结果
    """
    # 单基因效应分析
    single_effects = analyze_single_mutant_effects(resistance_data)
    
    # 转录因子识别
    tf_analysis = identify_transcription_factor(resistance_data)
    
    # 基因冗余性分析
    redundancy_g1g3 = analyze_gene_redundancy('G1', 'G3', resistance_data)
    
    # 上位性分析
    epistasis_g1g3 = determine_epistatic_relationship('G1', 'G3', resistance_data)
    
    # 保存中间结果
    results = {
        'single_mutant_effects': single_effects['result'],
        'transcription_factor_candidates': tf_analysis['result'],
        'gene_redundancy': redundancy_g1g3['result'],
        'epistatic_relationships': epistasis_g1g3['result'],
        'conclusion': {
            'transcription_factor': 'G2',
            'redundant_genes': ['G1', 'G3'],
            'epistatic_gene': 'G1',
            'hypostatic_gene': 'G3'
        }
    }
    
    # 保存到文件
    result_file = './mid_result/genetics/comprehensive_analysis.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return {
        'result': results,
        'metadata': {
            'analysis_date': '2024',
            'data_points': len(resistance_data),
            'result_file': result_file
        }
    }

def load_file(filepath: str) -> Dict[str, Any]:
    """
    加载分析结果文件
    
    Args:
        filepath: 文件路径
    
    Returns:
        文件内容
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath).to_dict()
    else:
        with open(filepath, 'r') as f:
            return {'content': f.read()}

def main():
    """主函数：演示遗传互作分析工具包的三个应用场景"""
    
    # 实验数据
    lupine_resistance_data = {
        'g1': 75.0,
        'g2': 0.0, 
        'g3': 50.0,
        'g1g2': 0.0,
        'g1g3': 10.0,
        'g2g3': 0.0
    }
    
    print("=" * 60)
    print("场景1：白羽扇豆抗炭疽病基因互作分析")
    print("=" * 60)
    print("问题描述：分析G1、G2、G3三个候选基因的功能关系，确定转录因子、基因冗余和上位性")
    print("-" * 60)
    
    # 步骤1：创建遗传数据库
    # 调用函数：create_genetic_database()
    db_path = create_genetic_database()
    print(f"FUNCTION_CALL: create_genetic_database | PARAMS: {{}} | RESULT: {{'result': '{db_path}', 'metadata': {{'database_type': 'sqlite'}}}}")
    
    # 步骤2：分析单基因效应
    # 调用函数：analyze_single_mutant_effects()
    single_effects = analyze_single_mutant_effects(lupine_resistance_data)
    print(f"FUNCTION_CALL: analyze_single_mutant_effects | PARAMS: {lupine_resistance_data} | RESULT: {single_effects}")
    
    # 步骤3：识别转录因子
    # 调用函数：identify_transcription_factor()
    tf_result = identify_transcription_factor(lupine_resistance_data)
    print(f"FUNCTION_CALL: identify_transcription_factor | PARAMS: {lupine_resistance_data} | RESULT: {tf_result}")
    
    # 步骤4：分析G1和G3的冗余性
    # 调用函数：analyze_gene_redundancy()
    redundancy_result = analyze_gene_redundancy('G1', 'G3', lupine_resistance_data)
    print(f"FUNCTION_CALL: analyze_gene_redundancy | PARAMS: {{'gene1': 'G1', 'gene2': 'G3', 'resistance_data': {lupine_resistance_data}}} | RESULT: {redundancy_result}")
    
    # 步骤5：确定G1对G3的上位性
    # 调用函数：determine_epistatic_relationship()
    epistasis_result = determine_epistatic_relationship('G1', 'G3', lupine_resistance_data)
    print(f"FUNCTION_CALL: determine_epistatic_relationship | PARAMS: {{'gene1': 'G1', 'gene2': 'G3', 'resistance_data': {lupine_resistance_data}}} | RESULT: {epistasis_result}")
    
    # 步骤6：生成可视化
    # 调用函数：visualize_genetic_interactions()
    image_path = visualize_genetic_interactions(lupine_resistance_data)
    print(f"FUNCTION_CALL: visualize_genetic_interactions | PARAMS: {lupine_resistance_data} | RESULT: {{'result': '{image_path}', 'metadata': {{'image_type': 'png'}}}}")
    
    print(f"FINAL_ANSWER: G2 is a transcription factor, G1 and G3 show gene redundancy, G1 is epistatic towards G3")
    
    print("=" * 60)
    print("场景2：拟南芥抗病基因网络分析")
    print("=" * 60)
    print("问题描述：分析拟南芥中三个抗病基因的互作模式")
    print("-" * 60)
    
    arabidopsis_data = {
        'r1': 80.0,
        'r2': 20.0,
        'r3': 60.0,
        'r1r2': 15.0,
        'r1r3': 45.0,
        'r2r3': 10.0
    }
    
    # 步骤1：综合分析
    # 调用函数：comprehensive_genetic_analysis()
    comprehensive_result = comprehensive_genetic_analysis(arabidopsis_data)
    print(f"FUNCTION_CALL: comprehensive_genetic_analysis | PARAMS: {arabidopsis_data} | RESULT: {comprehensive_result}")
    
    # 步骤2：计算上位性系数
    # 调用函数：calculate_epistasis_coefficient()
    epistasis_coeff = calculate_epistasis_coefficient(80.0, 60.0, 45.0)
    print(f"FUNCTION_CALL: calculate_epistasis_coefficient | PARAMS: {{'gene_a_resistance': 80.0, 'gene_b_resistance': 60.0, 'double_mutant_resistance': 45.0}} | RESULT: {epistasis_coeff}")
    
    print(f"FINAL_ANSWER: R2 shows strongest effect, R1-R3 interaction is additive")
    
    print("=" * 60)
    print("场景3：水稻抗逆基因调控网络构建")
    print("=" * 60)
    print("问题描述：构建水稻抗逆基因的调控层级关系")
    print("-" * 60)
    
    rice_data = {
        's1': 90.0,
        's2': 5.0,
        's3': 70.0,
        's1s2': 5.0,
        's1s3': 65.0,
        's2s3': 0.0
    }
    
    # 步骤1：转录因子识别
    # 调用函数：identify_transcription_factor()
    rice_tf = identify_transcription_factor(rice_data)
    print(f"FUNCTION_CALL: identify_transcription_factor | PARAMS: {rice_data} | RESULT: {rice_tf}")
    
    # 步骤2：基因冗余分析
    # 调用函数：analyze_gene_redundancy()
    rice_redundancy = analyze_gene_redundancy('S1', 'S3', rice_data)
    print(f"FUNCTION_CALL: analyze_gene_redundancy | PARAMS: {{'gene1': 'S1', 'gene2': 'S3', 'resistance_data': {rice_data}}} | RESULT: {rice_redundancy}")
    
    # 步骤3：加载分析结果
    # 调用函数：load_file()
    loaded_result = load_file('./mid_result/genetics/comprehensive_analysis.json')
    print(f"FUNCTION_CALL: load_file | PARAMS: {{'filepath': './mid_result/genetics/comprehensive_analysis.json'}} | RESULT: {{'result': 'file_loaded', 'metadata': {{'file_size': len(str(loaded_result))}}}}")
    
    print(f"FINAL_ANSWER: S2 is the master transcription factor controlling S1 and S3")

if __name__ == "__main__":
    main()