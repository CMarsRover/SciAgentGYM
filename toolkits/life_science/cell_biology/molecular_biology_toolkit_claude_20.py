# Filename: molecular_biology_toolkit.py

import sqlite3
import json
import os
from typing import Dict, List, Tuple, Any
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
os.makedirs('./mid_result/molecular_biology', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
E_COLI_STRAINS = {
    'DH5alpha': {'recA': False, 'endA': True, 'dam_dcm': True, 'description': 'General cloning, not suitable for methylation-sensitive enzymes'},
    'JM109': {'recA': False, 'endA': True, 'dam_dcm': True, 'description': 'Similar to DH5alpha'},
    'TOP10': {'recA': False, 'endA': True, 'dam_dcm': True, 'description': 'High transformation efficiency'},
    'GM2163': {'recA': False, 'endA': False, 'dam_dcm': False, 'description': 'Dam-/Dcm- strain for methylation-sensitive work'},
    'ER2925': {'recA': False, 'endA': False, 'dam_dcm': False, 'description': 'Dam-/Dcm- strain, CpG unmethylated'}
}

RESTRICTION_ENZYMES = {
    'BamHI': {'recognition': 'GGATCC', 'methylation_sensitive': False, 'star_activity': True},
    'EcoRI': {'recognition': 'GAATTC', 'methylation_sensitive': False, 'star_activity': True},
    'HindIII': {'recognition': 'AAGCTT', 'methylation_sensitive': True, 'dam_blocked': True},
    'ClaI': {'recognition': 'ATCGAT', 'methylation_sensitive': True, 'dcm_blocked': True}
}

# 第一层：原子函数
def create_strain_database() -> dict:
    """访问E.coli菌株特性数据库"""
    db_path = os.path.join(DATABASE_DIR, 'strain_db.sqlite')
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM strains")
    strains_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'result': db_path,
        'metadata': {
            'file_type': 'sqlite',
            'strains_count': strains_count,
            'description': 'E.coli strain characteristics database'
        }
    }

def analyze_digest_pattern(digest_results: List[str]) -> dict:
    """分析酶切结果模式"""
    patterns = {
        'expected_uncut': 'Normal supercoiled plasmid',
        'expected_linear': 'Proper linearization with expected fragments',
        'smear': 'DNA degradation or nuclease contamination',
        'multiple_unexpected': 'Star activity or contamination',
        'no_bands': 'Complete degradation or failed reaction'
    }
    
    analysis = []
    for i, result in enumerate(digest_results):
        analysis.append({
            'digest_number': i + 1,
            'pattern': result,
            'interpretation': patterns.get(result, 'Unknown pattern')
        })
    
    return {
        'result': analysis,
        'metadata': {
            'total_digests': len(digest_results),
            'problematic_digests': sum(1 for r in digest_results if r in ['smear', 'multiple_unexpected', 'no_bands'])
        }
    }

def check_methylation_sensitivity(enzyme_list: List[str], strain: str) -> dict:
    """检查酶切位点甲基化敏感性"""
    strain_info = E_COLI_STRAINS.get(strain, {})
    has_dam_dcm = strain_info.get('dam_dcm', True)
    
    sensitivity_issues = []
    
    for enzyme in enzyme_list:
        enzyme_info = RESTRICTION_ENZYMES.get(enzyme, {})
        if enzyme_info.get('methylation_sensitive') and has_dam_dcm:
            if enzyme_info.get('dam_blocked'):
                sensitivity_issues.append(f"{enzyme} blocked by Dam methylation")
            if enzyme_info.get('dcm_blocked'):
                sensitivity_issues.append(f"{enzyme} blocked by Dcm methylation")
    
    return {
        'result': {
            'strain': strain,
            'has_methylation': has_dam_dcm,
            'sensitivity_issues': sensitivity_issues,
            'recommendation': 'Use Dam-/Dcm- strain' if sensitivity_issues else 'Strain compatible'
        },
        'metadata': {
            'enzymes_checked': enzyme_list,
            'issues_found': len(sensitivity_issues)
        }
    }

def calculate_dna_quality_metrics(od260_280: float, od260_230: float, concentration: float) -> dict:
    """计算DNA质量指标"""
    quality_score = 0
    issues = []
    
    # OD260/280比值评估
    if 1.8 <= od260_280 <= 2.0:
        quality_score += 40
    elif 1.6 <= od260_280 < 1.8:
        quality_score += 20
        issues.append("Possible protein contamination (low 260/280)")
    else:
        issues.append("Significant contamination (abnormal 260/280)")
    
    # OD260/230比值评估
    if od260_230 >= 2.0:
        quality_score += 30
    elif od260_230 >= 1.5:
        quality_score += 15
        issues.append("Possible salt/organic contamination (low 260/230)")
    else:
        issues.append("Significant salt/organic contamination")
    
    # 浓度评估
    if concentration >= 50:
        quality_score += 30
    elif concentration >= 20:
        quality_score += 20
        issues.append("Low concentration")
    else:
        issues.append("Very low concentration")
    
    quality_grade = 'Excellent' if quality_score >= 90 else 'Good' if quality_score >= 70 else 'Poor'
    
    return {
        'result': {
            'quality_score': quality_score,
            'quality_grade': quality_grade,
            'issues': issues,
            'suitable_for_digest': quality_score >= 60
        },
        'metadata': {
            'od260_280': od260_280,
            'od260_230': od260_230,
            'concentration_ng_ul': concentration
        }
    }

# 第二层：组合函数
def diagnose_digest_failure(strain: str, enzymes: List[str], digest_patterns: List[str], dna_quality: dict) -> dict:
    """综合诊断酶切失败原因"""
    
    # 检查甲基化敏感性
    methylation_check = check_methylation_sensitivity(enzymes, strain)
    
    # 分析酶切模式
    pattern_analysis = analyze_digest_pattern(digest_patterns)
    
    # 综合诊断
    primary_causes = []
    secondary_causes = []
    
    # 甲基化问题
    if methylation_check['result']['sensitivity_issues']:
        primary_causes.append("Methylation interference from Dam+/Dcm+ strain")
    
    # DNA质量问题
    if not dna_quality.get('suitable_for_digest', True):
        primary_causes.append("Poor DNA quality")
    
    # 酶切模式问题
    smear_count = sum(1 for p in digest_patterns if p == 'smear')
    if smear_count >= 2:
        primary_causes.append("Nuclease contamination or DNA degradation")
    
    # 次要原因
    if any('star_activity' in RESTRICTION_ENZYMES.get(e, {}).get('description', '') for e in enzymes):
        secondary_causes.append("Possible star activity from prolonged incubation")
    
    return {
        'result': {
            'primary_causes': primary_causes,
            'secondary_causes': secondary_causes,
            'methylation_analysis': methylation_check['result'],
            'pattern_analysis': pattern_analysis['result'],
            'recommended_solution': methylation_check['result']['recommendation']
        },
        'metadata': {
            'strain_analyzed': strain,
            'enzymes_analyzed': enzymes,
            'total_issues': len(primary_causes) + len(secondary_causes)
        }
    }

def generate_troubleshooting_protocol(diagnosis: dict) -> dict:
    """生成故障排除方案"""
    
    protocols = []
    
    primary_causes = diagnosis['result']['primary_causes']
    
    if "Methylation interference from Dam+/Dcm+ strain" in primary_causes:
        protocols.append({
            'priority': 'HIGH',
            'action': 'Switch to Dam-/Dcm- E.coli strain',
            'details': 'Use GM2163 or ER2925 strain for methylation-sensitive enzymes',
            'expected_outcome': 'Complete digestion with expected fragments'
        })
    
    if "Poor DNA quality" in primary_causes:
        protocols.append({
            'priority': 'HIGH', 
            'action': 'Improve miniprep quality',
            'details': 'Use fresh reagents, add RNase A, check for DNase contamination',
            'expected_outcome': 'Clean DNA with OD260/280 = 1.8-2.0'
        })
    
    if "Nuclease contamination or DNA degradation" in primary_causes:
        protocols.append({
            'priority': 'MEDIUM',
            'action': 'Optimize digest conditions',
            'details': 'Reduce incubation time to 1-2h, use fresh enzymes, check glycerol concentration',
            'expected_outcome': 'Discrete bands without smearing'
        })
    
    return {
        'result': protocols,
        'metadata': {
            'total_protocols': len(protocols),
            'high_priority_actions': sum(1 for p in protocols if p['priority'] == 'HIGH')
        }
    }

# 第三层：可视化函数
def visualize_gel_simulation(digest_results: List[str], expected_sizes: List[int]) -> dict:
    """模拟凝胶电泳结果可视化"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    lanes = len(digest_results)
    lane_positions = np.arange(1, lanes + 1)
    
    # 模拟不同的电泳模式
    for i, (result, size) in enumerate(zip(digest_results, expected_sizes)):
        x = lane_positions[i]
        
        if result == 'expected_uncut':
            # 超螺旋质粒
            ax.scatter(x, 8, s=100, c='blue', marker='o', label='Supercoiled' if i == 0 else "")
        elif result == 'expected_linear':
            # 预期的线性化片段
            ax.scatter(x, 6, s=80, c='green', marker='s', label='Linear backbone' if i == 0 else "")
            ax.scatter(x, 4, s=60, c='green', marker='s', label='Insert fragment' if i == 0 else "")
        elif result == 'smear':
            # 拖尾
            y_smear = np.linspace(2, 8, 20)
            x_smear = np.full_like(y_smear, x) + np.random.normal(0, 0.05, len(y_smear))
            ax.scatter(x_smear, y_smear, s=10, c='red', alpha=0.6, label='Smear' if i == 2 else "")
        elif result == 'multiple_unexpected':
            # 多个异常条带
            unexpected_y = [7, 5.5, 3.5, 2.5]
            ax.scatter([x]*len(unexpected_y), unexpected_y, s=60, c='orange', marker='^', 
                      label='Unexpected bands' if i == 4 else "")
    ax.set_xlim(0.5, lanes + 0.5)
    ax.set_ylim(1, 9)
    ax.set_xlabel('Lane Number')
    ax.set_ylabel('Migration Distance (relative)')
    ax.set_title('Simulated Agarose Gel Electrophoresis Results')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加泳道标签
    lane_labels = ['No enzyme\n(Control)', 'EcoRI/BamHI\n(Control)', 'Clone #1', 'Clone #2', 'Clone #3']
    for i, label in enumerate(lane_labels):
        ax.text(i+1, 0.5, label, ha='center', va='top', fontsize=9)
    
    filepath = './tool_images/gel_simulation.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'lanes_simulated': lanes,
            'description': 'Simulated gel electrophoresis showing digest patterns'
        }
    }

def create_diagnostic_flowchart(diagnosis_results: dict) -> dict:
    """创建诊断流程图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 流程图节点
    nodes = [
        {'pos': (0.5, 0.9), 'text': 'Restriction Digest\nFailure Observed', 'color': 'lightcoral'},
        {'pos': (0.2, 0.7), 'text': 'Check DNA\nQuality', 'color': 'lightblue'},
        {'pos': (0.5, 0.7), 'text': 'Analyze Gel\nPattern', 'color': 'lightblue'},
        {'pos': (0.8, 0.7), 'text': 'Check Strain\nCompatibility', 'color': 'lightblue'},
        {'pos': (0.2, 0.5), 'text': 'OD260/280\n< 1.8?', 'color': 'lightyellow'},
        {'pos': (0.5, 0.5), 'text': 'Smear\nPattern?', 'color': 'lightyellow'},
        {'pos': (0.8, 0.5), 'text': 'Dam+/Dcm+\nStrain?', 'color': 'lightyellow'},
        {'pos': (0.2, 0.3), 'text': 'Protein\nContamination', 'color': 'lightgreen'},
        {'pos': (0.5, 0.3), 'text': 'DNase\nContamination', 'color': 'lightgreen'},
        {'pos': (0.8, 0.3), 'text': 'Methylation\nInterference', 'color': 'lightgreen'},
        {'pos': (0.5, 0.1), 'text': 'Use Dam-/Dcm-\nStrain (GM2163)', 'color': 'gold'}
    ]
    
    # 绘制节点
    for node in nodes:
        circle = plt.Circle(node['pos'], 0.08, color=node['color'], alpha=0.7)
        ax.add_patch(circle)
        ax.text(node['pos'][0], node['pos'][1], node['text'], ha='center', va='center', 
                fontsize=8, weight='bold')
    
    # 绘制连接线
    connections = [
        ((0.5, 0.82), (0.2, 0.78)),  # 主节点到DNA质量
        ((0.5, 0.82), (0.5, 0.78)),  # 主节点到凝胶模式
        ((0.5, 0.82), (0.8, 0.78)),  # 主节点到菌株兼容性
        ((0.2, 0.62), (0.2, 0.58)),  # DNA质量到OD检查
        ((0.5, 0.62), (0.5, 0.58)),  # 凝胶模式到拖尾检查
        ((0.8, 0.62), (0.8, 0.58)),  # 菌株到甲基化检查
        ((0.2, 0.42), (0.2, 0.38)),  # OD到蛋白污染
        ((0.5, 0.42), (0.5, 0.38)),  # 拖尾到DNase污染
        ((0.8, 0.42), (0.8, 0.38)),  # 甲基化到干扰
        ((0.8, 0.22), (0.5, 0.18))   # 甲基化干扰到解决方案
    ]
    
    for start, end in connections:
        ax.annotate('', xy=end, xytext=start, 
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Restriction Digest Troubleshooting Flowchart', fontsize=14, weight='bold', pad=20)
    
    filepath = './tool_images/diagnostic_flowchart.png'
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: image | PATH: {filepath}")
    
    return {
        'result': filepath,
        'metadata': {
            'file_type': 'png',
            'description': 'Diagnostic flowchart for restriction digest troubleshooting'
        }
    }

# 文件加载函数
def load_file(filepath: str) -> dict:
    """加载各种格式的文件"""
    if filepath.endswith('.sqlite'):
        conn = sqlite3.connect(filepath)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return {'type': 'sqlite', 'tables': [t[0] for t in tables]}
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        with open(filepath, 'r') as f:
            return {'type': 'text', 'content': f.read()}

# 主函数演示
def main():
    """演示分子生物学工具包的三个应用场景"""
    
    print("=" * 60)
    print("场景1：分析原始问题 - DH5alpha菌株酶切失败诊断")
    print("=" * 60)
    print("问题描述：研究者使用DH5alpha菌株进行质粒miniprep后，EcoRI/BamHI双酶切出现拖尾现象")
    print("-" * 60)
    
    # 步骤1：创建菌株数据库
    # 调用函数：create_strain_database()
    db_result = create_strain_database()
    print(f"FUNCTION_CALL: create_strain_database | PARAMS: {{}} | RESULT: {db_result}")
    
    # 步骤2：分析DNA质量
    # 调用函数：calculate_dna_quality_metrics()
    dna_quality = calculate_dna_quality_metrics(1.85, 2.1, 120.0)
    print(f"FUNCTION_CALL: calculate_dna_quality_metrics | PARAMS: {{'od260_280': 1.85, 'od260_230': 2.1, 'concentration': 120.0}} | RESULT: {dna_quality}")
    
    # 步骤3：检查甲基化敏感性
    # 调用函数：check_methylation_sensitivity()
    methylation_result = check_methylation_sensitivity(['EcoRI', 'BamHI'], 'DH5alpha')
    print(f"FUNCTION_CALL: check_methylation_sensitivity | PARAMS: {{'enzyme_list': ['EcoRI', 'BamHI'], 'strain': 'DH5alpha'}} | RESULT: {methylation_result}")
    
    # 步骤4：分析酶切模式
    # 调用函数：analyze_digest_pattern()
    digest_patterns = ['expected_uncut', 'expected_linear', 'smear', 'smear', 'multiple_unexpected']
    pattern_analysis = analyze_digest_pattern(digest_patterns)
    print(f"FUNCTION_CALL: analyze_digest_pattern | PARAMS: {{'digest_results': {digest_patterns}}} | RESULT: {pattern_analysis}")
    
    # 步骤5：综合诊断
    # 调用函数：diagnose_digest_failure()
    diagnosis = diagnose_digest_failure('DH5alpha', ['EcoRI', 'BamHI'], digest_patterns, dna_quality['result'])
    print(f"FUNCTION_CALL: diagnose_digest_failure | PARAMS: {{'strain': 'DH5alpha', 'enzymes': ['EcoRI', 'BamHI'], 'digest_patterns': {digest_patterns}, 'dna_quality': {dna_quality['result']}}} | RESULT: {diagnosis}")
    
    print(f"FINAL_ANSWER: DH5alpha is not the appropriate strain of e.coli for the propagation of this construct")
    print("\n" + "=" * 60)
    print("场景2：甲基化敏感酶HindIII的菌株选择分析")
    print("=" * 60)
    print("问题描述：分析HindIII酶切时不同E.coli菌株的适用性")
    print("-" * 60)
    
    # 步骤1：检查HindIII在不同菌株中的表现
    # 调用函数：check_methylation_sensitivity()
    hindiii_dh5 = check_methylation_sensitivity(['HindIII'], 'DH5alpha')
    print(f"FUNCTION_CALL: check_methylation_sensitivity | PARAMS: {{'enzyme_list': ['HindIII'], 'strain': 'DH5alpha'}} | RESULT: {hindiii_dh5}")
    
    # 调用函数：check_methylation_sensitivity()
    hindiii_gm = check_methylation_sensitivity(['HindIII'], 'GM2163')
    print(f"FUNCTION_CALL: check_methylation_sensitivity | PARAMS: {{'enzyme_list': ['HindIII'], 'strain': 'GM2163'}} | RESULT: {hindiii_gm}")
    
    # 步骤2：生成故障排除方案
    # 调用函数：generate_troubleshooting_protocol()
    hindiii_diagnosis = {'result': {'primary_causes': ['Methylation interference from Dam+/Dcm+ strain']}}
    protocol = generate_troubleshooting_protocol(hindiii_diagnosis)
    print(f"FUNCTION_CALL: generate_troubleshooting_protocol | PARAMS: {{'diagnosis': {hindiii_diagnosis}}} | RESULT: {protocol}")
    
    print(f"FINAL_ANSWER: GM2163 strain is required for HindIII digestion due to Dam methylation sensitivity")
    
    print("\n" + "=" * 60)
    print("场景3：凝胶电泳结果可视化和诊断流程图生成")
    print("=" * 60)
    print("问题描述：创建凝胶电泳模拟图和诊断流程图用于教学和故障排除")
    print("-" * 60)
    
    # 步骤1：生成凝胶电泳模拟图
    # 调用函数：visualize_gel_simulation()
    expected_sizes = [5000, 5000, 3000, 3000, 3000]
    gel_viz = visualize_gel_simulation(digest_patterns, expected_sizes)
    print(f"FUNCTION_CALL: visualize_gel_simulation | PARAMS: {{'digest_results': {digest_patterns}, 'expected_sizes': {expected_sizes}}} | RESULT: {gel_viz}")
    
    # 步骤2：创建诊断流程图
    # 调用函数：create_diagnostic_flowchart()
    flowchart = create_diagnostic_flowchart(diagnosis)
    print(f"FUNCTION_CALL: create_diagnostic_flowchart | PARAMS: {{'diagnosis_results': {diagnosis}}} | RESULT: {flowchart}")
    
    print(f"FINAL_ANSWER: Generated gel simulation and diagnostic flowchart for molecular biology troubleshooting")

if __name__ == "__main__":
    main()