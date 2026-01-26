#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
structural_biology 工具注册模块
使用 gym.tool.EnvironmentTool 为 structural_biology 目录中的工具提供统一的注册与调用接口

本文件由 collect_and_register_tools.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# from toolkits.life_science.structural_biology.rna_structure_toolkit_0160002 import *  # 动态导入
# from toolkits.life_science.structural_biology.structural_biology_toolkit_0002_fixed import *  # 动态导入

# ==================== 工具类定义 ====================


@Toolbox.register(name="fetch_pdb_structure")
class FetchPdbStructureTool(EnvironmentTool):
    """从RCSB PDB数据库下载蛋白质结构文件"""
    
    name = "fetch_pdb_structure"
    description = "从RCSB PDB数据库下载蛋白质结构文件"
    arguments = {
        "pdb_id": {"type": "string", "description": "PDB数据库标识符，4字符代码（如'1ABC'）"},
        "save_dir": {"type": "string", "description": "本地保存目录，默认'./pdb_files/'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 fetch_pdb_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pdb_id = args.get("pdb_id")
            if pdb_id is None:
                return Observation(self.name, "错误: 缺少必需参数 pdb_id")
            save_dir = args.get("save_dir", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.structural_biology_toolkit_0002_fixed import fetch_pdb_structure
            
            # 调用函数
            result = fetch_pdb_structure(pdb_id, save_dir)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="count_ligand_chains")
class CountLigandChainsTool(EnvironmentTool):
    """统计PDB结构中的配体链数量"""
    
    name = "count_ligand_chains"
    description = "统计PDB结构中的配体链数量"
    arguments = {
        "pdb_file": {"type": "string", "description": "PDB文件路径"},
        "remove_hydrogen": {"type": "boolean", "description": "是否移除氢原子，默认True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 count_ligand_chains 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pdb_file = args.get("pdb_file")
            if pdb_file is None:
                return Observation(self.name, "错误: 缺少必需参数 pdb_file")
            remove_hydrogen = args.get("remove_hydrogen", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.structural_biology_toolkit_0002_fixed import count_ligand_chains
            
            # 调用函数
            result = count_ligand_chains(pdb_file, remove_hydrogen)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="analyze_structure_composition")
class AnalyzeStructureCompositionTool(EnvironmentTool):
    """全面分析PDB结构的组成成分"""
    
    name = "analyze_structure_composition"
    description = "全面分析PDB结构的组成成分"
    arguments = {
        "pdb_file": {"type": "string", "description": "PDB文件路径"},
        "remove_hydrogen": {"type": "boolean", "description": "是否移除氢原子，默认True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_structure_composition 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pdb_file = args.get("pdb_file")
            if pdb_file is None:
                return Observation(self.name, "错误: 缺少必需参数 pdb_file")
            remove_hydrogen = args.get("remove_hydrogen", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.structural_biology_toolkit_0002_fixed import analyze_structure_composition
            
            # 调用函数
            result = analyze_structure_composition(pdb_file, remove_hydrogen)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="batch_analyze_pdb_structures")
class BatchAnalyzePdbStructuresTool(EnvironmentTool):
    """批量分析多个PDB结构的配体链数量"""
    
    name = "batch_analyze_pdb_structures"
    description = "批量分析多个PDB结构的配体链数量"
    arguments = {
        "pdb_ids": {"type": "array", "description": "PDB ID列表（如['1A2B', '3C4D']）"},
        "remove_hydrogen": {"type": "boolean", "description": "是否移除氢原子，默认True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_analyze_pdb_structures 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pdb_ids = args.get("pdb_ids")
            if pdb_ids is None:
                return Observation(self.name, "错误: 缺少必需参数 pdb_ids")
            remove_hydrogen = args.get("remove_hydrogen", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.structural_biology_toolkit_0002_fixed import batch_analyze_pdb_structures
            
            # 调用函数
            result = batch_analyze_pdb_structures(pdb_ids, remove_hydrogen)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="visualize_domain_specific")
class VisualizeDomainSpecificTool(EnvironmentTool):
    """结构生物学专属可视化工具"""
    
    name = "visualize_domain_specific"
    description = "结构生物学专属可视化工具"
    arguments = {
        "data": {"type": "object", "description": "要可视化的数据（只包含基本数据类型）"},
        "domain": {"type": "string", "description": "领域类型 'structural_biology'"},
        "vis_type": {"type": "string", "description": "可视化类型：'composition_pie'(结构组成饼图)、'chain_comparison'(多结构配体链对比柱状图)、'residue_distribution'(残基类型分布图)"},
        "save_dir": {"type": "string", "description": "保存目录，默认'./images/'"},
        "filename": {"type": "string", "description": "文件名（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_domain_specific 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            data = args.get("data")
            if data is None:
                return Observation(self.name, "错误: 缺少必需参数 data")
            domain = args.get("domain")
            if domain is None:
                return Observation(self.name, "错误: 缺少必需参数 domain")
            vis_type = args.get("vis_type")
            if vis_type is None:
                return Observation(self.name, "错误: 缺少必需参数 vis_type")
            save_dir = args.get("save_dir", None)
            filename = args.get("filename", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.structural_biology_toolkit_0002_fixed import visualize_domain_specific
            
            # 调用函数
            result = visualize_domain_specific(data, domain, vis_type, save_dir, filename)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="parse_rna_sequence")
class ParseRnaSequenceTool(EnvironmentTool):
    """解析RNA序列并提取基础特征，验证序列合法性并计算GC含量、长度等基本参数"""
    
    name = "parse_rna_sequence"
    description = "解析RNA序列并提取基础特征，验证序列合法性并计算GC含量、长度等基本参数"
    arguments = {
        "sequence": {"type": "string", "description": "RNA序列字符串，仅包含A/U/G/C（如'AUGCGAU'）"},
        "validate": {"type": "boolean", "description": "是否验证序列合法性，默认True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_rna_sequence 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            sequence = args.get("sequence")
            if sequence is None:
                return Observation(self.name, "错误: 缺少必需参数 sequence")
            validate = args.get("validate", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.rna_structure_toolkit_0160002 import parse_rna_sequence
            
            # 调用函数
            result = parse_rna_sequence(sequence, validate)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="detect_base_pairs")
class DetectBasePairsTool(EnvironmentTool):
    """检测RNA序列中的碱基配对（简化版二级结构预测），使用动态规划算法寻找最大配对数，返回配对位置列表"""
    
    name = "detect_base_pairs"
    description = "检测RNA序列中的碱基配对（简化版二级结构预测），使用动态规划算法寻找最大配对数，返回配对位置列表"
    arguments = {
        "sequence": {"type": "string", "description": "RNA序列字符串"},
        "min_stem_length": {"type": "integer", "description": "最小茎区长度，默认3（连续配对数）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_base_pairs 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            sequence = args.get("sequence")
            if sequence is None:
                return Observation(self.name, "错误: 缺少必需参数 sequence")
            min_stem_length = args.get("min_stem_length", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.rna_structure_toolkit_0160002 import detect_base_pairs
            
            # 调用函数
            result = detect_base_pairs(sequence, min_stem_length)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="calculate_structure_complexity")
class CalculateStructureComplexityTool(EnvironmentTool):
    """计算RNA结构复杂度指标，基于配对模式分析结构特征：茎区数量、假结、长程相互作用等"""
    
    name = "calculate_structure_complexity"
    description = "计算RNA结构复杂度指标，基于配对模式分析结构特征：茎区数量、假结、长程相互作用等"
    arguments = {
        "pairs": {"type": "array", "description": "碱基配对列表 [(i, j), ...]，其中 i < j"},
        "sequence_length": {"type": "integer", "description": "序列总长度"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_structure_complexity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pairs = args.get("pairs")
            if pairs is None:
                return Observation(self.name, "错误: 缺少必需参数 pairs")
            sequence_length = args.get("sequence_length")
            if sequence_length is None:
                return Observation(self.name, "错误: 缺少必需参数 sequence_length")
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.rna_structure_toolkit_0160002 import calculate_structure_complexity
            
            # 调用函数
            result = calculate_structure_complexity(pairs, sequence_length)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="classify_rna_type")
class ClassifyRnaTypeTool(EnvironmentTool):
    """基于结构特征分类RNA类型，根据长度、茎区数量、三级结构等特征判断RNA类型（tRNA/rRNA/ribozyme/mRNA）"""
    
    name = "classify_rna_type"
    description = "基于结构特征分类RNA类型，根据长度、茎区数量、三级结构等特征判断RNA类型（tRNA/rRNA/ribozyme/mRNA）"
    arguments = {
        "sequence": {"type": "string", "description": "RNA序列字符串"},
        "structure_features": {"type": "object", "description": "可选的预计算结构特征（来自analyze_rna_structure）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 classify_rna_type 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            sequence = args.get("sequence")
            if sequence is None:
                return Observation(self.name, "错误: 缺少必需参数 sequence")
            structure_features = args.get("structure_features", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.rna_structure_toolkit_0160002 import classify_rna_type
            
            # 调用函数
            result = classify_rna_type(sequence, structure_features)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


@Toolbox.register(name="predict_catalytic_activity")
class PredictCatalyticActivityTool(EnvironmentTool):
    """预测RNA的催化活性可能性，基于结构复杂度、三级相互作用等特征评估是否具有催化功能（ribozyme特性）"""
    
    name = "predict_catalytic_activity"
    description = "预测RNA的催化活性可能性，基于结构复杂度、三级相互作用等特征评估是否具有催化功能（ribozyme特性）"
    arguments = {
        "sequence": {"type": "string", "description": "RNA序列字符串"},
        "structure_features": {"type": "object", "description": "可选的预计算结构特征"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 predict_catalytic_activity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            sequence = args.get("sequence")
            if sequence is None:
                return Observation(self.name, "错误: 缺少必需参数 sequence")
            structure_features = args.get("structure_features", None)
            
            # 导入并调用原始函数
            from toolkits.life_science.structural_biology.rna_structure_toolkit_0160002 import predict_catalytic_activity
            
            # 调用函数
            result = predict_catalytic_activity(sequence, structure_features)
            
            # 处理返回值
            if isinstance(result, (dict, list)):
                return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
            elif isinstance(result, tuple):
                # 将元组转换为列表以便 JSON 序列化
                result_dict = {"result": list(result)}
                return Observation(self.name, json.dumps(result_dict, ensure_ascii=False, indent=2))
            else:
                return Observation(self.name, json.dumps({"result": result}, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")


# ==================== 工具注册函数 ====================

def register_structural_biology_tools(environment):
    """
    将所有 structural_biology 工具注册到环境中
    
    Args:
        environment: RepoEnv 实例
    """
    # 工具已通过 @Toolbox.register 装饰器自动注册
    # 此函数保留用于兼容性
    pass

