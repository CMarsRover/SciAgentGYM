# Filename: sars_cov2_molecular_biology_toolkit.py

import json
import sqlite3
import os
from typing import Dict, List, Any, Optional
import requests
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

# 配置matplotlib字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 获取脚本所在目录，用于构建数据库路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # extracted_tools_1113 目录
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# 创建必要的目录
os.makedirs('./mid_result/virology', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
SARS_COV2_GENOME_LENGTH = 29903
NSP_PROTEINS = {
    'nsp1': {'start': 266, 'end': 805, 'function': 'Host shutoff'},
    'nsp10': {'start': 13025, 'end': 13441, 'function': 'Cofactor for nsp14 and nsp16'},
    'nsp14': {'start': 18040, 'end': 19620, 'function': '3\'-5\' exonuclease and N7-methyltransferase'},
    'nsp15': {'start': 19621, 'end': 20658, 'function': 'Endoribonuclease'},
    'nsp16': {'start': 20659, 'end': 21552, 'function': '2\'-O-methyltransferase'}
}

@dataclass
class ProteinComplex:
    name: str
    components: List[str]
    function: str
    substrate: str
    product: str

# 第一层：原子函数
def create_local_nsp_database() -> dict:
    """访问本地SARS-CoV-2非结构蛋白数据库"""
    db_path = os.path.join(DATABASE_DIR, 'sars_cov2_nsp.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询记录数
    cursor.execute("SELECT COUNT(*) FROM nsp_proteins")
    record_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'result': db_path,
        'metadata': {
            'database_type': 'SQLite',
            'table_count': 1,
            'record_count': record_count,
            'file_size': os.path.getsize(db_path) if os.path.exists(db_path) else 0
        }
    }

def query_nsp_protein_info(protein_name: str) -> dict:
    """查询特定非结构蛋白信息"""
    db_path = os.path.join(DATABASE_DIR, 'sars_cov2_nsp.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM nsp_proteins WHERE name = ?
    ''', (protein_name,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'result': {
                'name': result[0],
                'genomic_position': {'start': result[1], 'end': result[2]},
                'length': result[3],
                'function': result[4],
                'domain_structure': result[5],
                'cofactors': result[6]
            },
            'metadata': {
                'source': 'local_database',
                'query_type': 'single_protein'
            }
        }
    else:
        return {
            'result': None,
            'metadata': {
                'error': f'Protein {protein_name} not found',
                'available_proteins': ['nsp10', 'nsp14', 'nsp15', 'nsp16']
            }
        }

def analyze_protein_complex_formation(protein1: str, protein2: str) -> dict:
    """分析蛋白质复合体形成的分子机制"""
    
    # 获取蛋白质信息
    p1_info = query_nsp_protein_info(protein1)
    p2_info = query_nsp_protein_info(protein2)
    
    if not p1_info['result'] or not p2_info['result']:
        return {
            'result': None,
            'metadata': {'error': 'One or both proteins not found'}
        }
    
    # 分析已知的复合体
    known_complexes = {
        ('nsp10', 'nsp14'): {
            'complex_name': 'nsp10/nsp14-ExoN',
            'interaction_type': 'heterodimer',
            'binding_site': 'nsp10 binds to N-terminal ExoN domain of nsp14',
            'function': '3\'-5\' exonuclease activity for RNA proofreading',
            'substrate': 'ssRNA with 3\'-terminal mismatches',
            'mechanism': 'nsp10 acts as allosteric activator of nsp14 exonuclease',
            'biological_significance': 'RNA proofreading during replication'
        },
        ('nsp10', 'nsp16'): {
            'complex_name': 'nsp10/nsp16-2\'-O-MTase',
            'interaction_type': 'heterodimer',
            'binding_site': 'nsp10 binds to nsp16 methyltransferase domain',
            'function': '2\'-O-methylation of viral mRNA cap',
            'substrate': 'N7-methylated GpppA-RNA',
            'mechanism': 'nsp10 enhances nsp16 methyltransferase activity',
            'biological_significance': 'mRNA cap modification for immune evasion'
        }
    }
    
    complex_key = tuple(sorted([protein1, protein2]))
    if complex_key in known_complexes:
        complex_info = known_complexes[complex_key]
    else:
        complex_info = {
            'complex_name': f'{protein1}/{protein2}',
            'interaction_type': 'unknown',
            'function': 'No known direct interaction',
            'biological_significance': 'Not characterized'
        }
    
    return {
        'result': {
            'proteins': [p1_info['result'], p2_info['result']],
            'complex_properties': complex_info
        },
        'metadata': {
            'analysis_type': 'protein_complex_formation',
            'interaction_known': complex_key in known_complexes
        }
    }

def validate_molecular_statement(statement: str, protein_complex: str) -> dict:
    """验证分子生物学陈述的准确性"""
    
    # 获取复合体信息
    if protein_complex == 'nsp10/nsp14':
        proteins = protein_complex.split('/')
        complex_info = analyze_protein_complex_formation(proteins[0], proteins[1])
        
        if not complex_info['result']:
            return {
                'result': False,
                'metadata': {'error': 'Complex information not available'}
            }
        
        complex_data = complex_info['result']['complex_properties']
        
        # 分析陈述的关键要素
        statement_lower = statement.lower()
        
        validation_results = {
            'heterodimer_formation': 'heterodimer' in statement_lower and complex_data['interaction_type'] == 'heterodimer',
            'exonuclease_function': 'exonuclease' in statement_lower and 'exonuclease' in complex_data['function'],
            'nsp10_binding': 'nsp10' in statement_lower and 'bind' in statement_lower,
            'dsrna_substrate': 'dsrna' in statement_lower,
            'mismatch_repair': 'mismatch repair' in statement_lower
        }
        
        # 检查错误陈述
        errors_found = []
        
        # nsp10/nsp14复合体主要处理ssRNA，不是dsRNA
        if validation_results['dsrna_substrate']:
            errors_found.append('nsp10/nsp14 complex processes ssRNA, not dsRNA')
        
        # 该复合体进行RNA校对，不是经典的DNA错配修复
        if validation_results['mismatch_repair']:
            errors_found.append('nsp10/nsp14 performs RNA proofreading, not classical mismatch repair')
        
        # 复合体防止RNA降解是不准确的，它实际上是移除错误核苷酸
        if 'prevents breakdown' in statement_lower:
            errors_found.append('nsp10/nsp14 removes incorrect nucleotides, does not prevent RNA breakdown')
        
        is_correct = len(errors_found) == 0
        
        return {
            'result': {
                'statement_correct': is_correct,
                'validation_details': validation_results,
                'errors_identified': errors_found,
                'correct_mechanism': complex_data
            },
            'metadata': {
                'statement_analyzed': statement,
                'complex_analyzed': protein_complex,
                'validation_criteria': list(validation_results.keys())
            }
        }
    else:
        return {
            'result': False,
            'metadata': {'error': f'Validation not implemented for {protein_complex}'}
        }

# 第二层：组合函数
def comprehensive_nsp_analysis(target_proteins: List[str]) -> dict:
    """综合分析多个非结构蛋白的功能和相互作用"""
    
    results = {}
    interactions = []
    
    # 分析每个蛋白质
    for protein in target_proteins:
        protein_info = query_nsp_protein_info(protein)
        if protein_info['result']:
            results[protein] = protein_info['result']
    
    # 分析蛋白质间相互作用
    for i, p1 in enumerate(target_proteins):
        for p2 in target_proteins[i+1:]:
            interaction = analyze_protein_complex_formation(p1, p2)
            if interaction['result'] and interaction['metadata']['interaction_known']:
                interactions.append(interaction['result'])
    
    # 保存结果
    output_file = './mid_result/virology/nsp_comprehensive_analysis.json'
    analysis_data = {
        'individual_proteins': results,
        'protein_interactions': interactions,
        'analysis_summary': {
            'total_proteins': len(results),
            'known_interactions': len(interactions),
            'functional_categories': list(set([info['function'].split()[0] for info in results.values()]))
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return {
        'result': analysis_data,
        'metadata': {
            'output_file': output_file,
            'proteins_analyzed': len(results),
            'interactions_found': len(interactions)
        }
    }

def molecular_mechanism_validator(statements: List[str]) -> dict:
    """批量验证分子机制陈述"""
    
    validation_results = []
    
    for i, statement in enumerate(statements):
        # 识别陈述中涉及的蛋白质复合体
        if 'nsp10' in statement and 'nsp14' in statement:
            complex_name = 'nsp10/nsp14'
        elif 'nsp10' in statement and 'nsp16' in statement:
            complex_name = 'nsp10/nsp16'
        else:
            complex_name = 'unknown'
        
        if complex_name != 'unknown':
            validation = validate_molecular_statement(statement, complex_name)
            validation_results.append({
                'statement_id': i + 1,
                'statement': statement,
                'complex': complex_name,
                'validation': validation['result']
            })
    
    # 统计结果
    correct_count = sum(1 for r in validation_results if r['validation']['statement_correct'])
    incorrect_count = len(validation_results) - correct_count
    
    return {
        'result': {
            'validation_results': validation_results,
            'summary': {
                'total_statements': len(validation_results),
                'correct_statements': correct_count,
                'incorrect_statements': incorrect_count,
                'accuracy_rate': correct_count / len(validation_results) if validation_results else 0
            }
        },
        'metadata': {
            'validation_method': 'molecular_mechanism_analysis',
            'complexes_analyzed': list(set([r['complex'] for r in validation_results]))
        }
    }

# 第三层：可视化函数
def visualize_nsp_genomic_organization() -> dict:
    """可视化非结构蛋白在基因组中的组织"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 绘制基因组
    ax.plot([0, SARS_COV2_GENOME_LENGTH], [0, 0], 'k-', linewidth=3, label='SARS-CoV-2 Genome')
    
    # 绘制NSP蛋白
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    y_positions = [0.1, -0.1, 0.2, -0.2, 0.3]
    
    for i, (nsp, info) in enumerate(NSP_PROTEINS.items()):
        start, end = info['start'], info['end']
        color = colors[i % len(colors)]
        y_pos = y_positions[i % len(y_positions)]
        
        # 绘制蛋白质区域
        ax.plot([start, end], [y_pos, y_pos], color=color, linewidth=8, alpha=0.7)
        ax.text((start + end) / 2, y_pos + 0.05, nsp, ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text((start + end) / 2, y_pos - 0.05, f'{start}-{end}', ha='center', va='top', fontsize=8)
    
    ax.set_xlim(0, SARS_COV2_GENOME_LENGTH)
    ax.set_ylim(-0.4, 0.4)
    ax.set_xlabel('Genomic Position (nt)', fontsize=12)
    ax.set_title('SARS-CoV-2 Nonstructural Proteins Genomic Organization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 移除y轴刻度
    ax.set_yticks([])
    
    filepath = './tool_images/nsp_genomic_organization.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG',
            'proteins_visualized': len(NSP_PROTEINS),
            'genome_length': SARS_COV2_GENOME_LENGTH
        }
    }

def visualize_protein_complex_interactions() -> dict:
    """可视化蛋白质复合体相互作用网络"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # nsp10/nsp14复合体
    ax1.text(0.5, 0.8, 'nsp10/nsp14 Complex', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 绘制nsp10
    circle1 = plt.Circle((0.3, 0.5), 0.1, color='lightblue', alpha=0.7)
    ax1.add_patch(circle1)
    ax1.text(0.3, 0.5, 'nsp10\n(Cofactor)', ha='center', va='center', fontsize=10)
    
    # 绘制nsp14
    circle2 = plt.Circle((0.7, 0.5), 0.15, color='lightcoral', alpha=0.7)
    ax1.add_patch(circle2)
    ax1.text(0.7, 0.5, 'nsp14\n(ExoN)', ha='center', va='center', fontsize=10)
    
    # 绘制相互作用
    ax1.arrow(0.4, 0.5, 0.15, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax1.text(0.5, 0.55, 'Binding', ha='center', va='bottom', fontsize=9)
    
    # 功能描述
    ax1.text(0.5, 0.2, 'Function: 3\'-5\' exonuclease\nSubstrate: ssRNA\nRole: RNA proofreading', 
             ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # nsp10/nsp16复合体
    ax2.text(0.5, 0.8, 'nsp10/nsp16 Complex', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 绘制nsp10
    circle3 = plt.Circle((0.3, 0.5), 0.1, color='lightblue', alpha=0.7)
    ax2.add_patch(circle3)
    ax2.text(0.3, 0.5, 'nsp10\n(Cofactor)', ha='center', va='center', fontsize=10)
    
    # 绘制nsp16
    circle4 = plt.Circle((0.7, 0.5), 0.15, color='lightgreen', alpha=0.7)
    ax2.add_patch(circle4)
    ax2.text(0.7, 0.5, 'nsp16\n(2\'-O-MTase)', ha='center', va='center', fontsize=10)
    
    # 绘制相互作用
    ax2.arrow(0.4, 0.5, 0.15, 0, head_width=0.02, head_length=0.03, fc='black', ec='black')
    ax2.text(0.5, 0.55, 'Binding', ha='center', va='bottom', fontsize=9)
    
    # 功能描述
    ax2.text(0.5, 0.2, 'Function: 2\'-O-methylation\nSubstrate: mRNA cap\nRole: Immune evasion', 
             ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    filepath = './tool_images/protein_complex_interactions.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'PNG',
            'complexes_visualized': 2,
            'interaction_type': 'heterodimer_formation'
        }
    }

def main():
    """主函数：演示SARS-CoV-2分子生物学分析工具包"""
    
    print("=" * 60)
    print("场景1：验证SARS-CoV-2 nsp10/nsp14复合体机制陈述的准确性")
    print("=" * 60)
    print("问题描述：分析给定的分子生物学陈述是否正确，识别其中的错误")
    print("-" * 60)
    
    # 步骤1：创建本地数据库
    # 调用函数：create_local_nsp_database()
    db_result = create_local_nsp_database()
    print(f"FUNCTION_CALL: create_local_nsp_database | PARAMS: {{}} | RESULT: {db_result}")
    
    # 步骤2：分析nsp10/nsp14复合体
    # 调用函数：analyze_protein_complex_formation()
    complex_analysis = analyze_protein_complex_formation('nsp10', 'nsp14')
    print(f"FUNCTION_CALL: analyze_protein_complex_formation | PARAMS: {{'protein1': 'nsp10', 'protein2': 'nsp14'}} | RESULT: {complex_analysis}")
    
    # 步骤3：验证标准答案中的陈述
    # 调用函数：validate_molecular_statement()
    target_statement = "SARS-CoV-2 nsp10/nsp14-ExoN operates as heterodimers in a mismatch repair mechanism. The N-terminal ExoN domain of nsp14 could bind to nsp10 making an active exonuclease complex that prevents the breakdown of dsRNA."
    validation_result = validate_molecular_statement(target_statement, 'nsp10/nsp14')
    print(f"FUNCTION_CALL: validate_molecular_statement | PARAMS: {{'statement': '{target_statement[:50]}...', 'protein_complex': 'nsp10/nsp14'}} | RESULT: {validation_result}")
    
    # 分析结果
    is_correct = validation_result['result']['statement_correct']
    errors = validation_result['result']['errors_identified']
    
    print(f"FINAL_ANSWER: The statement is {'CORRECT' if is_correct else 'INCORRECT'}. Errors identified: {'; '.join(errors) if errors else 'None'}")
    
    print("=" * 60)
    print("场景2：综合分析SARS-CoV-2非结构蛋白功能网络")
    print("=" * 60)
    print("问题描述：分析多个NSP蛋白的功能和相互作用关系")
    print("-" * 60)
    
    # 步骤1：综合分析目标蛋白
    # 调用函数：comprehensive_nsp_analysis()
    target_nsps = ['nsp10', 'nsp14', 'nsp15', 'nsp16']
    comprehensive_result = comprehensive_nsp_analysis(target_nsps)
    print(f"FUNCTION_CALL: comprehensive_nsp_analysis | PARAMS: {{'target_proteins': {target_nsps}}} | RESULT: {comprehensive_result}")
    
    # 步骤2：可视化基因组组织
    # 调用函数：visualize_nsp_genomic_organization()
    genomic_viz = visualize_nsp_genomic_organization()
    print(f"FUNCTION_CALL: visualize_nsp_genomic_organization | PARAMS: {{}} | RESULT: {genomic_viz}")
    
    interaction_count = comprehensive_result['result']['analysis_summary']['total_proteins']
    print(f"FINAL_ANSWER: Analyzed {interaction_count} NSP proteins with {len(comprehensive_result['result']['protein_interactions'])} known interactions")
    

if __name__ == "__main__":
    main()