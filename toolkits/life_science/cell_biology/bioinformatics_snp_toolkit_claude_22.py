# Filename: bioinformatics_snp_toolkit.py

import requests
import json
from typing import Dict, List, Optional, Tuple
import re
import os
from xml.etree import ElementTree as ET

# 创建输出目录
os.makedirs('./mid_result/bioinformatics', exist_ok=True)
os.makedirs('./tool_images', exist_ok=True)

# 全局常量
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ENSEMBL_BASE_URL = "https://rest.ensembl.org"

def fetch_snp_from_ncbi(rs_id: str) -> Dict:
    """
    从NCBI dbSNP获取SNP基本信息
    
    Args:
        rs_id: SNP的rs编号 (如 'rs113993960')
    
    Returns:
        dict: {'result': snp_info, 'metadata': {...}}
    """
    try:
        snp_id = rs_id.replace('rs', '')
        efetch_url = f"{NCBI_BASE_URL}/efetch.fcgi"
        params = {
            'db': 'snp',
            'id': snp_id,
            'retmode': 'xml'
        }
        
        response = requests.get(efetch_url, params=params, timeout=30)
        response.raise_for_status()
        
        # 解析XML
        root = ET.fromstring(response.content)
        
        # 提取基本信息
        snp_info = {
            'rs_id': rs_id,
            'chromosome': None,
            'position': None,
            'alleles': [],
            'gene': None
        }
        
        # 提取染色体和位置信息
        for assembly in root.findall('.//Assembly'):
            component = assembly.find('.//Component')
            if component is not None:
                snp_info['chromosome'] = component.get('chromosome', 'Unknown')
                snp_info['position'] = int(component.get('start', 0))
                break
        
        # 提取等位基因信息
        for allele in root.findall('.//Allele'):
            allele_text = allele.text
            if allele_text:
                snp_info['alleles'].append(allele_text)
        
        # 提取基因信息
        for gene in root.findall('.//Gene'):
            gene_symbol = gene.get('symbol')
            if gene_symbol:
                snp_info['gene'] = gene_symbol
                break
        
        return {
            'result': snp_info,
            'metadata': {
                'source': 'NCBI_dbSNP',
                'api_version': 'eutils',
                'timestamp': 'current'
            }
        }
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'error': str(e),
                'source': 'NCBI_dbSNP'
            }
        }

def fetch_flanking_sequence_ensembl(rs_id: str, window_size: int = 100) -> Dict:
    """
    从Ensembl获取SNP侧翼序列
    
    Args:
        rs_id: SNP的rs编号
        window_size: 上下游窗口大小
    
    Returns:
        dict: {'result': sequence_data, 'metadata': {...}}
    """
    try:
        # 获取SNP变异信息
        var_url = f"{ENSEMBL_BASE_URL}/variation/human/{rs_id}"
        headers = {"Content-Type": "application/json"}
        
        response = requests.get(var_url, headers=headers, timeout=30)
        response.raise_for_status()
        var_data = response.json()
        
        if 'mappings' not in var_data or len(var_data['mappings']) == 0:
            raise ValueError(f"No mappings found for {rs_id}")
        
        # 获取第一个映射
        mapping = var_data['mappings'][0]
        location_parts = mapping['location'].split(':')
        chrom = location_parts[0]
        start_pos = mapping['start']
        end_pos = mapping['end']
        
        # 计算侧翼区域
        flank_start = max(1, start_pos - window_size)
        flank_end = end_pos + window_size
        
        # 获取基因组序列
        seq_url = f"{ENSEMBL_BASE_URL}/sequence/region/human/{chrom}:{flank_start}..{flank_end}:1"
        seq_response = requests.get(seq_url, headers=headers, timeout=30)
        seq_response.raise_for_status()
        seq_data = seq_response.json()
        
        sequence_info = {
            'rs_id': rs_id,
            'chromosome': chrom,
            'snp_position': start_pos,
            'flanking_start': flank_start,
            'flanking_end': flank_end,
            'window_size': window_size,
            'sequence': seq_data.get('seq', '').upper(),
            'allele_string': mapping.get('allele_string', ''),
            'strand': mapping.get('strand', 1)
        }
        
        return {
            'result': sequence_info,
            'metadata': {
                'source': 'Ensembl_REST_API',
                'genome_build': 'GRCh38',
                'sequence_length': len(sequence_info['sequence'])
            }
        }
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'error': str(e),
                'source': 'Ensembl_REST_API'
            }
        }

def format_sequence_with_spacing(sequence: str, line_length: int = 50, group_size: int = 10) -> str:
    """
    格式化DNA序列，添加空格和换行
    
    Args:
        sequence: DNA序列字符串
        line_length: 每行字符数
        group_size: 每组字符数
    
    Returns:
        dict: {'result': formatted_sequence, 'metadata': {...}}
    """
    if not sequence or not isinstance(sequence, str):
        return {
            'result': '',
            'metadata': {'error': 'Invalid sequence input'}
        }
    
    # 移除所有空格和换行
    clean_seq = re.sub(r'\s+', '', sequence.upper())
    
    # 按组添加空格
    grouped = []
    for i in range(0, len(clean_seq), group_size):
        grouped.append(clean_seq[i:i+group_size])
    
    spaced_seq = ' '.join(grouped)
    
    # 按行分割
    lines = []
    words = spaced_seq.split(' ')
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + len(current_line) <= line_length:
            current_line.append(word)
            current_length += len(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    formatted = '\n'.join(lines)
    
    return {
        'result': formatted,
        'metadata': {
            'original_length': len(clean_seq),
            'formatted_lines': len(lines),
            'group_size': group_size
        }
    }

def analyze_sequence_composition(sequence: str) -> Dict:
    """
    分析DNA序列组成
    
    Args:
        sequence: DNA序列字符串
    
    Returns:
        dict: {'result': composition_stats, 'metadata': {...}}
    """
    if not sequence:
        return {
            'result': {},
            'metadata': {'error': 'Empty sequence'}
        }
    
    clean_seq = re.sub(r'\s+', '', sequence.upper())
    total_length = len(clean_seq)
    
    if total_length == 0:
        return {
            'result': {},
            'metadata': {'error': 'No valid nucleotides found'}
        }
    
    # 计算碱基组成
    composition = {
        'A': clean_seq.count('A'),
        'T': clean_seq.count('T'),
        'G': clean_seq.count('G'),
        'C': clean_seq.count('C'),
        'N': clean_seq.count('N')
    }
    
    # 计算百分比
    percentages = {
        base: (count / total_length) * 100 
        for base, count in composition.items()
    }
    
    # 计算GC含量
    gc_content = (composition['G'] + composition['C']) / total_length * 100
    
    stats = {
        'total_length': total_length,
        'composition': composition,
        'percentages': percentages,
        'gc_content': round(gc_content, 2),
        'at_content': round(100 - gc_content, 2)
    }
    
    return {
        'result': stats,
        'metadata': {
            'analysis_type': 'nucleotide_composition',
            'valid_bases': total_length - composition['N']
        }
    }

def extract_surrounding_nucleotides(rs_id: str, target_length: int = 200) -> Dict:
    """
    提取SNP周围指定长度的核苷酸序列
    
    Args:
        rs_id: SNP的rs编号
        target_length: 目标序列长度
    
    Returns:
        dict: {'result': surrounding_sequence, 'metadata': {...}}
    """
    # 计算需要的侧翼长度
    flank_size = target_length // 2
    # 获取侧翼序列
    seq_result = fetch_flanking_sequence_ensembl(rs_id, flank_size)
    
    if seq_result['result'] is None:
        return {
            'result': None,
            'metadata': {
                'error': 'Failed to fetch sequence',
                'rs_id': rs_id
            }
        }
    
    seq_data = seq_result['result']
    full_sequence = seq_data['sequence']
    
    # 确保序列长度符合要求
    if len(full_sequence) < target_length:
        # 如果序列太短，尝试获取更大的窗口
        extended_result = fetch_flanking_sequence_ensembl(rs_id, target_length)
        if extended_result['result']:
            full_sequence = extended_result['result']['sequence']
    
    # 截取目标长度
    if len(full_sequence) >= target_length:
        start_idx = (len(full_sequence) - target_length) // 2
        target_sequence = full_sequence[start_idx:start_idx + target_length]
    else:
        target_sequence = full_sequence
    
    return {
        'result': {
            'rs_id': rs_id,
            'sequence': target_sequence,
            'length': len(target_sequence),
            'snp_position_in_sequence': len(target_sequence) // 2,
            'chromosome': seq_data.get('chromosome'),
            'genomic_position': seq_data.get('snp_position')
        },
        'metadata': {
            'target_length': target_length,
            'actual_length': len(target_sequence),
            'source': 'Ensembl',
            'genome_build': 'GRCh38'
        }
    }

def save_sequence_to_file(sequence_data: Dict, filename: str) -> Dict:
    """
    保存序列数据到文件
    
    Args:
        sequence_data: 序列数据字典
        filename: 输出文件名
    
    Returns:
        dict: {'result': filepath, 'metadata': {...}}
    """
    filepath = f"./mid_result/bioinformatics/{filename}"
    
    try:
        with open(filepath, 'w') as f:
            f.write(f"RS ID: {sequence_data.get('rs_id', 'Unknown')}\n")
            f.write(f"Chromosome: {sequence_data.get('chromosome', 'Unknown')}\n")
            f.write(f"Position: {sequence_data.get('genomic_position', 'Unknown')}\n")
            f.write(f"Sequence Length: {sequence_data.get('length', 0)}\n")
            f.write(f"Sequence:\n")
            # 格式化序列
            format_result = format_sequence_with_spacing(sequence_data.get('sequence', ''))
            f.write(format_result['result'])
            f.write('\n')
        
        return {
            'result': filepath,
            'metadata': {
                'file_type': 'text',
                'size': os.path.getsize(filepath),
                'encoding': 'utf-8'
            }
        }
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'error': str(e),
                'attempted_filepath': filepath
            }
        }

def load_file(filepath: str) -> Dict:
    """
    加载文件内容
    
    Args:
        filepath: 文件路径
    
    Returns:
        dict: {'result': file_content, 'metadata': {...}}
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'result': content,
            'metadata': {
                'filepath': filepath,
                'size': len(content),
                'lines': len(content.split('\n'))
            }
        }
    except Exception as e:
        return {
            'result': None,
            'metadata': {
                'error': str(e),
                'filepath': filepath
            }
        }

def main():
    """演示生物信息学SNP分析工具包的三个场景"""
    
    print("=" * 60)
    print("场景1：获取rs113993960周围200个核苷酸序列")
    print("=" * 60)
    print("问题描述：分析SNP rs113993960周围200个核苷酸，验证标准答案")
    print("-" * 60)
    
    # 步骤1：获取SNP基本信息
    # 调用函数：fetch_snp_from_ncbi()
    snp_info = fetch_snp_from_ncbi("rs113993960")
    print(f"FUNCTION_CALL: fetch_snp_from_ncbi | PARAMS: {{'rs_id': 'rs113993960'}} | RESULT: {snp_info}")
    
    # 步骤2：提取200个核苷酸序列
    # 调用函数：extract_surrounding_nucleotides()
    sequence_result = extract_surrounding_nucleotides("rs113993960", 200)
    print(f"FUNCTION_CALL: extract_surrounding_nucleotides | PARAMS: {{'rs_id': 'rs113993960', 'target_length': 200}} | RESULT: {sequence_result}")
    
    # 步骤3：格式化序列显示
    # 调用函数：format_sequence_with_spacing()
    if sequence_result['result']:
        formatted_seq = format_sequence_with_spacing(sequence_result['result']['sequence'])
        print(f"FUNCTION_CALL: format_sequence_with_spacing | PARAMS: {{'sequence': 'DNA_SEQUENCE'}} | RESULT: {formatted_seq}")
        # 步骤4：保存序列到文件
        # 调用函数：save_sequence_to_file()
        save_result = save_sequence_to_file(sequence_result['result'], "rs113993960_200bp.txt")
        print(f"FUNCTION_CALL: save_sequence_to_file | PARAMS: {{'sequence_data': 'DICT', 'filename': 'rs113993960_200bp.txt'}} | RESULT: {save_result}")
        
        if save_result['result']:
            print(f"FILE_GENERATED: text | PATH: {save_result['result']}")
        
        print(f"FINAL_ANSWER: {formatted_seq['result']}")
    
    print("\n" + "=" * 60)
    print("场景2：分析序列组成和GC含量")
    print("=" * 60)
    print("问题描述：分析获取的200bp序列的核苷酸组成和GC含量")
    print("-" * 60)
    
    if sequence_result['result']:
        # 步骤1：分析序列组成 调用函数：analyze_sequence_composition()
        composition = analyze_sequence_composition(sequence_result['result']['sequence'])
        print(f"FUNCTION_CALL: analyze_sequence_composition | PARAMS: {{'sequence': 'DNA_SEQUENCE'}} | RESULT: {composition}")
        
        if composition['result']:
            stats = composition['result']
            print(f"FINAL_ANSWER: GC含量: {stats['gc_content']}%, 总长度: {stats['total_length']}bp")
    
    print("\n" + "=" * 60)
    print("场景3：比较不同窗口大小的序列获取")
    print("=" * 60)
    print("问题描述：获取rs113993960周围不同长度的序列进行比较分析")
    print("-" * 60)
    
    # 步骤1：获取100bp序列
    # 调用函数：extract_surrounding_nucleotides()
    seq_100 = extract_surrounding_nucleotides("rs113993960", 100)
    print(f"FUNCTION_CALL: extract_surrounding_nucleotides | PARAMS: {{'rs_id': 'rs113993960', 'target_length': 100}} | RESULT: {seq_100}")
    
    # 步骤2：获取300bp序列
    # 调用函数：extract_surrounding_nucleotides()
    seq_300 = extract_surrounding_nucleotides("rs113993960", 300)
    print(f"FUNCTION_CALL: extract_surrounding_nucleotides | PARAMS: {{'rs_id': 'rs113993960', 'target_length': 300}} | RESULT: {seq_300}")
    
    # 步骤3：比较序列长度
    lengths = []
    if seq_100['result']:
        lengths.append(f"100bp: {seq_100['result']['length']}")
    if sequence_result['result']:
        lengths.append(f"200bp: {sequence_result['result']['length']}")
    if seq_300['result']:
        lengths.append(f"300bp: {seq_300['result']['length']}")
    
    print(f"FINAL_ANSWER: 序列长度比较: {', '.join(lengths)}")

if __name__ == "__main__":
    main()