# Filename: embryonic_stem_cell_enhancer_toolkit.py

import sqlite3
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import requests
from dataclasses import dataclass
import os

# 获取脚本所在目录，用于构建数据库路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # extracted_tools_1113 目录
DATABASE_DIR = os.path.join(BASE_DIR, 'database')

# 配置matplotlib字体，避免乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False

# 全局常量
POLYCOMB_TARGETS = ['PRC1', 'PRC2', 'CBX7', 'EZH2', 'SUZ12', 'RING1B']
HISTONE_MARKS = ['H3K4me1', 'H3K4me3', 'H3K27me3', 'H2AK119ub']
ESC_MARKERS = ['OCT4', 'SOX2', 'NANOG', 'KLF4']

@dataclass
class ChromatinContact:
    """染色质接触数据结构"""
    chr1: str
    start1: int
    end1: int
    chr2: str
    start2: int
    end2: int
    contact_frequency: float
    distance: int

def create_esc_enhancer_database() -> Dict:
    """
    访问胚胎干细胞增强子本地数据库
    
    Returns:
        dict: 包含数据库路径的结果字典
    """
    db_path = os.path.join(DATABASE_DIR, 'esc_enhancer.db')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询表信息
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    # 查询记录数
    total_records = 0
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        total_records += cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'result': db_path,
        'metadata': {
            'tables': tables,
            'total_records': total_records
        }
    }

def query_polycomb_mediated_contacts(db_path: str, complex_type: str = None) -> Dict:
    """
    查询Polycomb复合体介导的增强子-启动子接触
    
    Args:
        db_path: 数据库路径
        complex_type: Polycomb复合体类型 (PRC1, PRC2, CBX7, EZH2等)
    
    Returns:
        dict: 包含接触信息的结果字典
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if complex_type:
        query = '''
            SELECT cc.*, e.name as enhancer_name, p.gene_name
            FROM chromatin_contacts cc
            JOIN enhancers e ON (cc.chr1 = e.chr AND cc.start1 >= e.start AND cc.end1 <= e.end)
            JOIN promoters p ON (cc.chr2 = p.chr AND cc.start2 >= p.start AND cc.end2 <= p.end)
            WHERE cc.mediated_by = ?
        '''
        cursor.execute(query, (complex_type,))
    else:
        query = '''
            SELECT cc.*, e.name as enhancer_name, p.gene_name
            FROM chromatin_contacts cc
            JOIN enhancers e ON (cc.chr1 = e.chr AND cc.start1 >= e.start AND cc.end1 <= e.end)
            JOIN promoters p ON (cc.chr2 = p.chr AND cc.start2 >= p.start AND cc.end2 <= p.end)
            WHERE cc.mediated_by IN ('PRC1', 'PRC2', 'CBX7', 'EZH2')
        '''
        cursor.execute(query)
    
    results = cursor.fetchall()
    conn.close()
    
    contacts = []
    for row in results:
        contacts.append({
            'contact_id': row[0],
            'enhancer_chr': row[1],
            'enhancer_pos': row[2],
            'promoter_chr': row[4],
            'promoter_pos': row[5],
            'contact_frequency': row[7],
            'distance': row[8],
            'mediated_by': row[9],
            'enhancer_name': row[10],
            'target_gene': row[11]
        })
    
    return {
        'result': contacts,
        'metadata': {
            'total_contacts': len(contacts),
            'complex_type': complex_type or 'all_polycomb',
            'avg_contact_frequency': np.mean([c['contact_frequency'] for c in contacts]) if contacts else 0
        }
    }

def analyze_enhancer_promoter_distance_distribution(db_path: str) -> Dict:
    """
    分析增强子-启动子距离分布
    
    Args:
        db_path: 数据库路径
    
    Returns:
        dict: 包含距离分布统计的结果字典
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = '''
        SELECT distance, contact_frequency, mediated_by
        FROM chromatin_contacts
        WHERE mediated_by IN ('PRC1', 'PRC2', 'CBX7', 'EZH2')
    '''
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    distances = [row[0] for row in results]
    frequencies = [row[1] for row in results]
    mediators = [row[2] for row in results]
    
    # 统计分析
    distance_stats = {
        'mean_distance': np.mean(distances),
        'median_distance': np.median(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances)
    }
    
    # 按介导因子分组
    mediator_stats = {}
    for mediator in set(mediators):
        med_distances = [d for d, m in zip(distances, mediators) if m == mediator]
        med_frequencies = [f for f, m in zip(frequencies, mediators) if m == mediator]
        mediator_stats[mediator] = {
            'count': len(med_distances),
            'avg_distance': np.mean(med_distances),
            'avg_frequency': np.mean(med_frequencies)
        }
    
    return {
        'result': {
            'distance_stats': distance_stats,
            'mediator_stats': mediator_stats,
            'raw_data': {'distances': distances, 'frequencies': frequencies, 'mediators': mediators}
        },
        'metadata': {
            'total_contacts': len(distances),
            'unique_mediators': len(set(mediators))
        }
    }

def calculate_polycomb_enrichment_score(db_path: str, gene_name: str) -> Dict:
    """
    计算特定基因的Polycomb富集评分
    
    Args:
        db_path: 数据库路径
        gene_name: 目标基因名称
    
    Returns:
        dict: 包含富集评分的结果字典
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取基因启动子信息
    cursor.execute('SELECT * FROM promoters WHERE gene_name = ?', (gene_name,))
    promoter = cursor.fetchone()
    
    if not promoter:
        conn.close()
        return {
            'result': None,
            'metadata': {'error': f'Gene {gene_name} not found in database'}
        }
    
    # 查找该基因相关的Polycomb接触
    cursor.execute('''
        SELECT cc.contact_frequency, cc.mediated_by, ps.h3k27me3_signal
        FROM chromatin_contacts cc
        JOIN polycomb_sites ps ON cc.mediated_by = ps.complex_type
        WHERE cc.chr2 = ? AND cc.start2 >= ? AND cc.end2 <= ?
    ''', (promoter[1], promoter[2], promoter[3]))
    
    contacts = cursor.fetchall()
    conn.close()
    
    if not contacts:
        enrichment_score = 0.0
        polycomb_activity = 'Low'
    else:
        # 计算富集评分：接触频率 × H3K27me3信号强度
        scores = [freq * h3k27me3 for freq, _, h3k27me3 in contacts]
        enrichment_score = np.mean(scores)
        
        if enrichment_score > 30:
            polycomb_activity = 'High'
        elif enrichment_score > 15:
            polycomb_activity = 'Medium'
        else:
            polycomb_activity = 'Low'
    
    return {
        'result': {
            'gene_name': gene_name,
            'enrichment_score': enrichment_score,
            'polycomb_activity': polycomb_activity,
            'contact_count': len(contacts)
        },
        'metadata': {
            'promoter_location': f"{promoter[1]}:{promoter[2]}-{promoter[3]}",
            'h3k4me3_signal': promoter[6],
            'expression_level': promoter[7]
        }
    }

def visualize_enhancer_promoter_network(db_path: str) -> str:
    """
    可视化增强子-启动子相互作用网络
    
    Args:
        db_path: 数据库路径
    
    Returns:
        str: 图像文件路径
    """
    os.makedirs('./tool_images', exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取所有接触数据
    cursor.execute('''
        SELECT cc.contact_frequency, cc.distance, cc.mediated_by, 
               e.name, p.gene_name
        FROM chromatin_contacts cc
        JOIN enhancers e ON (cc.chr1 = e.chr AND cc.start1 >= e.start AND cc.end1 <= e.end)
        JOIN promoters p ON (cc.chr2 = p.chr AND cc.start2 >= p.start AND cc.end2 <= p.end)
    ''')
    
    results = cursor.fetchall()
    conn.close()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 接触频率 vs 距离散点图
    frequencies = [r[0] for r in results]
    distances = [r[1]/1000000 for r in results]  # 转换为Mb
    mediators = [r[2] for r in results]
    
    colors = {'PRC1': 'red', 'PRC2': 'blue', 'CBX7': 'green', 'EZH2': 'orange'}
    for mediator in set(mediators):
        med_freq = [f for f, m in zip(frequencies, mediators) if m == mediator]
        med_dist = [d for d, m in zip(distances, mediators) if m == mediator]
        ax1.scatter(med_dist, med_freq, c=colors.get(mediator, 'gray'), 
                   label=mediator, alpha=0.7, s=100)
    
    ax1.set_xlabel('Distance (Mb)')
    ax1.set_ylabel('Contact Frequency')
    ax1.set_title('Enhancer-Promoter Contact vs Distance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Polycomb复合体介导的接触频率分布
    mediator_freq = {}
    for mediator in set(mediators):
        mediator_freq[mediator] = [f for f, m in zip(frequencies, mediators) if m == mediator]
    
    ax2.boxplot([mediator_freq[m] for m in mediator_freq.keys()], 
                labels=list(mediator_freq.keys()))
    ax2.set_ylabel('Contact Frequency')
    ax2.set_title('Contact Frequency by Polycomb Complex')
    ax2.grid(True, alpha=0.3)
    
    # 3. 基因-增强子连接网络
    genes = [r[4] for r in results]
    enhancers = [r[3] for r in results]
    gene_counts = {}
    for gene in set(genes):
        gene_counts[gene] = genes.count(gene)
    
    ax3.bar(gene_counts.keys(), gene_counts.values(), 
            color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax3.set_ylabel('Number of Enhancer Contacts')
    ax3.set_title('Enhancer Contacts per Gene')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 距离分布直方图
    ax4.hist(distances, bins=10, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Distance (Mb)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Enhancer-Promoter Distances')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = './tool_images/enhancer_promoter_network.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    return filepath

def simulate_polycomb_knockout_effect(db_path: str, knockout_complex: str) -> Dict:
    """
    模拟Polycomb复合体敲除对增强子-启动子接触的影响
    
    Args:
        db_path: 数据库路径
        knockout_complex: 要敲除的Polycomb复合体 (PRC1, PRC2, CBX7, EZH2)
    
    Returns:
        dict: 包含敲除效应分析的结果字典
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取野生型接触数据
    cursor.execute('SELECT * FROM chromatin_contacts WHERE mediated_by = ?', (knockout_complex,))
    wt_contacts = cursor.fetchall()
    
    # 获取其他复合体的接触数据作为对照
    cursor.execute('SELECT * FROM chromatin_contacts WHERE mediated_by != ?', (knockout_complex,))
    control_contacts = cursor.fetchall()
    
    conn.close()
    
    # 模拟敲除效应：假设敲除后接触频率下降70-90%
    knockout_effect = np.random.uniform(0.1, 0.3, len(wt_contacts))
    wt_frequencies = [c[7] for c in wt_contacts]
    ko_frequencies = [freq * effect for freq, effect in zip(wt_frequencies, knockout_effect)]
    
    # 计算统计指标
    wt_mean = np.mean(wt_frequencies) if wt_frequencies else 0
    ko_mean = np.mean(ko_frequencies) if ko_frequencies else 0
    fold_change = wt_mean / ko_mean if ko_mean > 0 else float('inf')
    
    # 对照组统计
    control_frequencies = [c[7] for c in control_contacts]
    control_mean = np.mean(control_frequencies) if control_frequencies else 0
    return {
        'result': {
            'knockout_complex': knockout_complex,
            'wt_mean_frequency': wt_mean,
            'ko_mean_frequency': ko_mean,
            'fold_change': fold_change,
            'control_mean_frequency': control_mean,
            'affected_contacts': len(wt_contacts),
            'unaffected_contacts': len(control_contacts)
        },
        'metadata': {
            'simulation_parameters': {
                'knockout_efficiency': '70-90%',
                'model': 'random_uniform_reduction'
            },
            'interpretation': 'High' if fold_change > 3 else 'Medium' if fold_change > 2 else 'Low'
        }
    }

def load_file(filepath: str) -> Dict:
    """
    加载文件内容的通用函数
    
    Args:
        filepath: 文件路径
    
    Returns:
        dict: 包含文件内容的结果字典
    """
    try:
        if filepath.endswith('.db'):
            # SQLite数据库文件
            conn = sqlite3.connect(filepath)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            return {
                'result': {'type': 'database', 'tables': tables},
                'metadata': {'file_type': 'sqlite', 'path': filepath}
            }
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            return {
                'result': data,
                'metadata': {'file_type': 'json', 'size': len(str(data))}
            }
        else:
            with open(filepath, 'r') as f:
                content = f.read()
            return {
                'result': content,
                'metadata': {'file_type': 'text', 'size': len(content)}
            }
    except Exception as e:
        return {
            'result': None,
            'metadata': {'error': str(e), 'file_type': 'unknown'}
        }

def main():
    """主函数：演示胚胎干细胞增强子分析的三个场景"""
    
    print("=" * 60)
    print("场景1：验证Polycomb复合体介导增强子-启动子长程接触")
    print("=" * 60)
    print("问题描述：分析胚胎干细胞中Polycomb复合体是否参与介导增强子与启动子之间的长程染色质接触")
    print("-" * 60)
    
    # 步骤1：创建胚胎干细胞增强子数据库
    # 调用函数：create_esc_enhancer_database()
    db_result = create_esc_enhancer_database()
    print(f"FUNCTION_CALL: create_esc_enhancer_database | PARAMS: {{}} | RESULT: {db_result}")
    
    db_path = db_result['result']
    
    # 步骤2：查询Polycomb介导的增强子-启动子接触
    # 调用函数：query_polycomb_mediated_contacts()
    contacts_result = query_polycomb_mediated_contacts(db_path)
    print(f"FUNCTION_CALL: query_polycomb_mediated_contacts | PARAMS: {{'db_path': '{db_path}'}} | RESULT: {contacts_result}")
    
    # 步骤3：分析接触距离分布
    # 调用函数：analyze_enhancer_promoter_distance_distribution()
    distance_result = analyze_enhancer_promoter_distance_distribution(db_path)
    print(f"FUNCTION_CALL: analyze_enhancer_promoter_distance_distribution | PARAMS: {{'db_path': '{db_path}'}} | RESULT: {distance_result}")
    
    polycomb_contacts = len(contacts_result['result'])
    avg_frequency = contacts_result['metadata']['avg_contact_frequency']
    
    print(f"FINAL_ANSWER: Polycomb complexes mediate {polycomb_contacts} long-range enhancer-promoter contacts with average frequency {avg_frequency:.3f}")
    
    print("=" * 60)
    print("场景2：评估不同Polycomb复合体对特定基因的调控强度")
    print("=" * 60)
    print("问题描述：计算OCT4基因的Polycomb富集评分，评估其受Polycomb调控的程度")
    print("-" * 60)
    
    # 步骤1：计算OCT4基因的Polycomb富集评分
    # 调用函数：calculate_polycomb_enrichment_score()
    oct4_score = calculate_polycomb_enrichment_score(db_path, 'OCT4')
    print(f"FUNCTION_CALL: calculate_polycomb_enrichment_score | PARAMS: {{'db_path': '{db_path}', 'gene_name': 'OCT4'}} | RESULT: {oct4_score}")
    
    # 步骤2：计算NANOG基因的Polycomb富集评分进行对比
    # 调用函数：calculate_polycomb_enrichment_score()
    nanog_score = calculate_polycomb_enrichment_score(db_path, 'NANOG')
    print(f"FUNCTION_CALL: calculate_polycomb_enrichment_score | PARAMS: {{'db_path': '{db_path}', 'gene_name': 'NANOG'}} | RESULT: {nanog_score}")
    
    oct4_activity = oct4_score['result']['polycomb_activity']
    nanog_activity = nanog_score['result']['polycomb_activity']
    
    print(f"FINAL_ANSWER: OCT4 shows {oct4_activity} Polycomb activity, NANOG shows {nanog_activity} Polycomb activity")
    
    print("=" * 60)
    print("场景3：模拟Polycomb复合体敲除对染色质结构的影响")
    print("=" * 60)
    print("问题描述：模拟PRC2复合体敲除后对增强子-启动子接触网络的影响并可视化结果")
    print("-" * 60)
    
    # 步骤1：可视化野生型增强子-启动子网络
    # 调用函数：visualize_enhancer_promoter_network()
    network_image = visualize_enhancer_promoter_network(db_path)
    print(f"FUNCTION_CALL: visualize_enhancer_promoter_network | PARAMS: {{'db_path': '{db_path}'}} | RESULT: {{'result': '{network_image}', 'metadata': {{'file_type': 'png'}}}}")
    
    # 步骤2：模拟PRC2敲除效应
    # 调用函数：simulate_polycomb_knockout_effect()
    knockout_result = simulate_polycomb_knockout_effect(db_path, 'PRC2')
    print(f"FUNCTION_CALL: simulate_polycomb_knockout_effect | PARAMS: {{'db_path': '{db_path}', 'knockout_complex': 'PRC2'}} | RESULT: {knockout_result}")
    
    fold_change = knockout_result['result']['fold_change']
    affected_contacts = knockout_result['result']['affected_contacts']
    
    print(f"FINAL_ANSWER: PRC2 knockout reduces enhancer-promoter contacts by {fold_change:.1f}-fold, affecting {affected_contacts} contact sites")

if __name__ == "__main__":
    main()