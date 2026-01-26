#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
computational_chemistry 工具注册模块
使用 gym.tool.EnvironmentTool 为 computational_chemistry 目录中的工具提供统一的注册与调用接口

本文件由 register_tools_for_directory.py 自动生成
"""

import json
import traceback
from typing import Any, Dict, Optional
from gym.tool import EnvironmentTool
from gym.entities import Observation
from gym.toolbox import Toolbox

# 注意: 实际导入在工具类中动态进行，以避免循环依赖

# ==================== 工具类定义 ====================


# ==================== 新增工具类（由 register_tools_for_directory.py 自动添加） ====================


@Toolbox.register(name="ensure_dirs")
class EnsureDirsTool(EnvironmentTool):
    """确保所需的中间结果与图像目录存在。"""
    
    name = "ensure_dirs"
    description = "确保所需的中间结果与图像目录存在。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 ensure_dirs 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import ensure_dirs
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = ensure_dirs(**func_kwargs)
            
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


@Toolbox.register(name="save_mid_result")
class SaveMidResultTool(EnvironmentTool):
    """将中间结果以JSON格式保存到文件系统，并返回结构化信息。"""
    
    name = "save_mid_result"
    description = "将中间结果以JSON格式保存到文件系统，并返回结构化信息。"
    arguments = {
        "subject": {"type": "string", "description": "学科或主题标签，例如“chemistry”。"},
        "label": {"type": "string", "description": "中间结果标记，用于文件命名。"},
        "data": {"type": "object", "description": "要保存的中间结果数据（字典对象，内容可嵌套）。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 save_mid_result 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "subject" not in args or args["subject"] is None:
                return Observation(self.name, "错误: 缺少必需参数 subject")
            if "label" not in args or args["label"] is None:
                return Observation(self.name, "错误: 缺少必需参数 label")
            if "data" not in args or args["data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import save_mid_result
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["subject", "label", "data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = save_mid_result(**func_kwargs)
            
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


@Toolbox.register(name="load_file")
class LoadFileTool(EnvironmentTool):
    """通用文件加载器，支持JSON或文本文件内容读取。"""
    
    name = "load_file"
    description = "通用文件加载器，支持JSON或文本文件内容读取。"
    arguments = {
        "filepath": {"type": "string", "description": "文件路径；支持.json与文本文件。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 load_file 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "filepath" not in args or args["filepath"] is None:
                return Observation(self.name, "错误: 缺少必需参数 filepath")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import load_file
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["filepath"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = load_file(**func_kwargs)
            
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


@Toolbox.register(name="google_search_resources")
class GoogleSearchResourcesTool(EnvironmentTool):
    """Google 搜索工具桩函数：返回与化学对称性相关的API与文档的参考列表（仅元数据）。"""
    
    name = "google_search_resources"
    description = "Google 搜索工具桩函数：返回与化学对称性相关的API与文档的参考列表（仅元数据）。"
    arguments = {
        "query": {"type": "string", "description": "检索关键词，例如“molecular symmetry point groups; PubChem; RDKit”。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 google_search_resources 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "query" not in args or args["query"] is None:
                return Observation(self.name, "错误: 缺少必需参数 query")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import google_search_resources
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["query"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = google_search_resources(**func_kwargs)
            
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


@Toolbox.register(name="init_local_db")
class InitLocalDbTool(EnvironmentTool):
    """初始化本地SQLite数据库并创建分子对称性表结构。"""
    
    name = "init_local_db"
    description = "初始化本地SQLite数据库并创建分子对称性表结构。"
    arguments = {
        "db_path": {"type": "string", "description": "SQLite数据库文件路径。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 init_local_db 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "db_path" not in args or args["db_path"] is None:
                return Observation(self.name, "错误: 缺少必需参数 db_path")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import init_local_db
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["db_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = init_local_db(**func_kwargs)
            
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


@Toolbox.register(name="insert_local_entries")
class InsertLocalEntriesTool(EnvironmentTool):
    """将经整理的分子对称性条目插入本地数据库。"""
    
    name = "insert_local_entries"
    description = "将经整理的分子对称性条目插入本地数据库。"
    arguments = {
        "entries": {"type": "array", "description": "待插入的条目列表；每个条目为字典对象，通常包含name、smiles、point_group、has_c3_axis、has_sigma_h、has_c2_perp_axes、provenance等字段。"},
        "db_path": {"type": "string", "description": "SQLite数据库文件路径。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 insert_local_entries 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "entries" not in args or args["entries"] is None:
                return Observation(self.name, "错误: 缺少必需参数 entries")
            if "db_path" not in args or args["db_path"] is None:
                return Observation(self.name, "错误: 缺少必需参数 db_path")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import insert_local_entries
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["entries", "db_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = insert_local_entries(**func_kwargs)
            
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


@Toolbox.register(name="query_local_symmetry")
class QueryLocalSymmetryTool(EnvironmentTool):
    """按分子名称查询本地数据库中的对称性信息。"""
    
    name = "query_local_symmetry"
    description = "按分子名称查询本地数据库中的对称性信息。"
    arguments = {
        "name": {"type": "string", "description": "分子名称。"},
        "db_path": {"type": "string", "description": "SQLite数据库文件路径。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 query_local_symmetry 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            if "db_path" not in args or args["db_path"] is None:
                return Observation(self.name, "错误: 缺少必需参数 db_path")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import query_local_symmetry
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name", "db_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = query_local_symmetry(**func_kwargs)
            
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


@Toolbox.register(name="pubchem_fetch_cid_by_name")
class PubchemFetchCidByNameTool(EnvironmentTool):
    """通过化合物名称从PubChem获取CID。"""
    
    name = "pubchem_fetch_cid_by_name"
    description = "通过化合物名称从PubChem获取CID。"
    arguments = {
        "name": {"type": "string", "description": "化合物名称（非空）。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 pubchem_fetch_cid_by_name 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import pubchem_fetch_cid_by_name
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = pubchem_fetch_cid_by_name(**func_kwargs)
            
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


@Toolbox.register(name="pubchem_fetch_smiles")
class PubchemFetchSmilesTool(EnvironmentTool):
    """根据PubChem CID获取标准SMILES字符串。"""
    
    name = "pubchem_fetch_smiles"
    description = "根据PubChem CID获取标准SMILES字符串。"
    arguments = {
        "cid": {"type": "integer", "description": "PubChem CID（正整数）。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 pubchem_fetch_smiles 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "cid" not in args or args["cid"] is None:
                return Observation(self.name, "错误: 缺少必需参数 cid")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import pubchem_fetch_smiles
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cid"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = pubchem_fetch_smiles(**func_kwargs)
            
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


@Toolbox.register(name="rdkit_generate_3d")
class RdkitGenerate3dTool(EnvironmentTool):
    """使用RDKit从SMILES生成3D构象坐标与元素列表。"""
    
    name = "rdkit_generate_3d"
    description = "使用RDKit从SMILES生成3D构象坐标与元素列表。"
    arguments = {
        "smiles": {"type": "string", "description": "SMILES字符串（非空）。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 rdkit_generate_3d 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "smiles" not in args or args["smiles"] is None:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import rdkit_generate_3d
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["smiles"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = rdkit_generate_3d(**func_kwargs)
            
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


@Toolbox.register(name="compute_planarity")
class ComputePlanarityTool(EnvironmentTool):
    """计算点集相对于最佳拟合平面的平面度（RMS偏差）。"""
    
    name = "compute_planarity"
    description = "计算点集相对于最佳拟合平面的平面度（RMS偏差）。"
    arguments = {
        "coords": {"type": "array", "description": "三维坐标列表，形状为N×3；至少包含6个原子。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_planarity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coords" not in args or args["coords"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import compute_planarity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coords"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_planarity(**func_kwargs)
            
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


@Toolbox.register(name="detect_c3_axis")
class DetectC3AxisTool(EnvironmentTool):
    """基于投影角度分布的启发式方法检测分子的C3主轴。"""
    
    name = "detect_c3_axis"
    description = "基于投影角度分布的启发式方法检测分子的C3主轴。"
    arguments = {
        "coords": {"type": "array", "description": "三维坐标列表，形状为N×3。"},
        "centroid": {"type": "array", "description": "坐标质心 [cx, cy, cz]。"},
        "normal": {"type": "array", "description": "最佳拟合平面的法向量 [nx, ny, nz]。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_c3_axis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coords" not in args or args["coords"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            if "centroid" not in args or args["centroid"] is None:
                return Observation(self.name, "错误: 缺少必需参数 centroid")
            if "normal" not in args or args["normal"] is None:
                return Observation(self.name, "错误: 缺少必需参数 normal")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import detect_c3_axis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coords", "centroid", "normal"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = detect_c3_axis(**func_kwargs)
            
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


@Toolbox.register(name="detect_sigma_h")
class DetectSigmaHTool(EnvironmentTool):
    """依据平面度阈值检测是否存在水平镜面（σh）。"""
    
    name = "detect_sigma_h"
    description = "依据平面度阈值检测是否存在水平镜面（σh）。"
    arguments = {
        "planarity_rms": {"type": "number", "description": "平面度RMS偏差（单位：Å）。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_sigma_h 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "planarity_rms" not in args or args["planarity_rms"] is None:
                return Observation(self.name, "错误: 缺少必需参数 planarity_rms")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import detect_sigma_h
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["planarity_rms"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = detect_sigma_h(**func_kwargs)
            
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


@Toolbox.register(name="detect_c2_perp_axes")
class DetectC2PerpAxesTool(EnvironmentTool):
    """启发式检测与C3轴垂直的多个C2轴，用于区分D3h与C3h。"""
    
    name = "detect_c2_perp_axes"
    description = "启发式检测与C3轴垂直的多个C2轴，用于区分D3h与C3h。"
    arguments = {
        "coords": {"type": "array", "description": "三维坐标列表，形状为N×3。"},
        "centroid": {"type": "array", "description": "坐标质心 [cx, cy, cz]。"},
        "normal": {"type": "array", "description": "最佳拟合平面的法向量 [nx, ny, nz]。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 detect_c2_perp_axes 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coords" not in args or args["coords"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            if "centroid" not in args or args["centroid"] is None:
                return Observation(self.name, "错误: 缺少必需参数 centroid")
            if "normal" not in args or args["normal"] is None:
                return Observation(self.name, "错误: 缺少必需参数 normal")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import detect_c2_perp_axes
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coords", "centroid", "normal"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = detect_c2_perp_axes(**func_kwargs)
            
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


@Toolbox.register(name="infer_point_group")
class InferPointGroupTool(EnvironmentTool):
    """根据几何特征推断简化的分子点群（C3h、D3h、C3v、C3或C1）。"""
    
    name = "infer_point_group"
    description = "根据几何特征推断简化的分子点群（C3h、D3h、C3v、C3或C1）。"
    arguments = {
        "features": {"type": "object", "description": "几何与对称性特征字典。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 infer_point_group 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "features" not in args or args["features"] is None:
                return Observation(self.name, "错误: 缺少必需参数 features")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import infer_point_group
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["features"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = infer_point_group(**func_kwargs)
            
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


@Toolbox.register(name="assess_symmetry_by_name")
class AssessSymmetryByNameTool(EnvironmentTool):
    """基于名称尝试判定分子点群：优先使用PubChem与RDKit的几何启发式，失败时回退至本地数据库。"""
    
    name = "assess_symmetry_by_name"
    description = "基于名称尝试判定分子点群：优先使用PubChem与RDKit的几何启发式，失败时回退至本地数据库。"
    arguments = {
        "name": {"type": "string", "description": "分子名称（非空）。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 assess_symmetry_by_name 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import assess_symmetry_by_name
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = assess_symmetry_by_name(**func_kwargs)
            
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

@Toolbox.register(name="visualize_molecule_3d")
class VisualizeMolecule3dTool(EnvironmentTool):
    """使用py3Dmol按XYZ格式渲染分子，并保存文本文件路径。"""
    
    name = "visualize_molecule_3d"
    description = "使用py3Dmol按XYZ格式渲染分子，并保存文本文件路径。"
    arguments = {
        "coords": {"type": "array", "description": "三维坐标列表，形状为N×3；与elements长度一致。"},
        "elements": {"type": "array", "description": "元素符号列表，例如“C”、“H”、“O”；与coords长度一致。"},
        "name": {"type": "string", "description": "分子名称，用于输出文件命名。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_molecule_3d 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coords" not in args or args["coords"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coords")
            if "elements" not in args or args["elements"] is None:
                return Observation(self.name, "错误: 缺少必需参数 elements")
            if "name" not in args or args["name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 name")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import visualize_molecule_3d
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coords", "elements", "name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_molecule_3d(**func_kwargs)
            
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


@Toolbox.register(name="prepare_local_db_entries")
class PrepareLocalDbEntriesTool(EnvironmentTool):
    """插入与题目相关的经整理分子及其点群分类到本地数据库，并记录溯源信息。"""
    
    name = "prepare_local_db_entries"
    description = "插入与题目相关的经整理分子及其点群分类到本地数据库，并记录溯源信息。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 prepare_local_db_entries 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import prepare_local_db_entries
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = prepare_local_db_entries(**func_kwargs)
            
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


@Toolbox.register(name="batch_assess")
class BatchAssessTool(EnvironmentTool):
    """批量评估多个分子名称的点群并返回映射字典。"""
    
    name = "batch_assess"
    description = "批量评估多个分子名称的点群并返回映射字典。"
    arguments = {
        "names": {"type": "array", "description": "分子名称列表。"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_assess 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "names" not in args or args["names"] is None:
                return Observation(self.name, "错误: 缺少必需参数 names")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.gpqa_physics_chemsirty_16 import batch_assess
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["names"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = batch_assess(**func_kwargs)
            
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


@Toolbox.register(name="main")
class MainTool(EnvironmentTool):
    """主函数"""
    
    name = "main"
    description = "主函数"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 main 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import main
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = main(**func_kwargs)
            
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


@Toolbox.register(name="calculate_h2_energy")
class CalculateH2EnergyTool(EnvironmentTool):
    """计算给定键长的氢分子(H₂)能量。"""
    
    name = "calculate_h2_energy"
    description = "计算给定键长的氢分子(H₂)能量。"
    arguments = {
        "bond_length": {"type": "number", "description": "氢分子的键长，单位为埃(Å)"},
        "basis": {"type": "string", "description": "计算使用的基组，默认为'sto-3g'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_h2_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "bond_length" not in args or args["bond_length"] is None:
                return Observation(self.name, "错误: 缺少必需参数 bond_length")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry import calculate_h2_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["bond_length", "basis"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_h2_energy(**func_kwargs)
            
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


@Toolbox.register(name="find_equilibrium_bond_length")
class FindEquilibriumBondLengthTool(EnvironmentTool):
    """通过计算一系列键长下的能量，找出平衡键长。"""
    
    name = "find_equilibrium_bond_length"
    description = "通过计算一系列键长下的能量，找出平衡键长。"
    arguments = {
        "bond_lengths": {"type": "array", "description": "要计算的键长数组，单位为埃(Å)"},
        "basis": {"type": "string", "description": "计算使用的基组，默认为'sto-3g'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_equilibrium_bond_length 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "bond_lengths" not in args or args["bond_lengths"] is None:
                return Observation(self.name, "错误: 缺少必需参数 bond_lengths")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry import find_equilibrium_bond_length
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["bond_lengths", "basis"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = find_equilibrium_bond_length(**func_kwargs)
            
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


@Toolbox.register(name="plot_potential_energy_curve")
class PlotPotentialEnergyCurveTool(EnvironmentTool):
    """绘制势能曲线。"""
    
    name = "plot_potential_energy_curve"
    description = "绘制势能曲线。"
    arguments = {
        "bond_lengths": {"type": "array", "description": "键长数组，单位为埃(Å)"},
        "energies": {"type": "array", "description": "对应的能量数组，单位为哈特里(Hartree)"},
        "equilibrium_length": {"type": "number", "description": "平衡键长，如果提供则会在图上标记"},
        "min_energy": {"type": "number", "description": "最小能量，如果提供则会在图上标记"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_potential_energy_curve 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "bond_lengths" not in args or args["bond_lengths"] is None:
                return Observation(self.name, "错误: 缺少必需参数 bond_lengths")
            if "energies" not in args or args["energies"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energies")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry import plot_potential_energy_curve
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["bond_lengths", "energies", "equilibrium_length", "min_energy"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_potential_energy_curve(**func_kwargs)
            
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


@Toolbox.register(name="parse_args")
class ParseArgsTool(EnvironmentTool):
    """解析命令行参数"""
    
    name = "parse_args"
    description = "解析命令行参数"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_args 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import parse_args
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_args(**func_kwargs)
            
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


@Toolbox.register(name="setup_molecule")
class SetupMoleculeTool(EnvironmentTool):
    """设置分子计算模型"""
    
    name = "setup_molecule"
    description = "设置分子计算模型"
    arguments = {
        "molecule_spec": {"type": "string", "description": "分子规格，可以是原子坐标列表或XYZ文件路径"},
        "basis": {"type": "string", "description": "基组名称，默认为'cc-pvdz'"},
        "charge": {"type": "integer", "description": "分子电荷，默认为0"},
        "spin": {"type": "integer", "description": "自旋多重度减1，默认为0（单重态）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 setup_molecule 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "molecule_spec" not in args or args["molecule_spec"] is None:
                return Observation(self.name, "错误: 缺少必需参数 molecule_spec")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import setup_molecule
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["molecule_spec", "basis", "charge", "spin"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = setup_molecule(**func_kwargs)
            
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


@Toolbox.register(name="parse_xyz")
class ParseXyzTool(EnvironmentTool):
    """解析XYZ文件"""
    
    name = "parse_xyz"
    description = "解析XYZ文件"
    arguments = {
        "xyz_file": {"type": "string", "description": "XYZ文件路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_xyz 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "xyz_file" not in args or args["xyz_file"] is None:
                return Observation(self.name, "错误: 缺少必需参数 xyz_file")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import parse_xyz
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["xyz_file"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_xyz(**func_kwargs)
            
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


@Toolbox.register(name="run_scf_calculation")
class RunScfCalculationTool(EnvironmentTool):
    """执行自洽场计算"""
    
    name = "run_scf_calculation"
    description = "执行自洽场计算"
    arguments = {
        "molecule_spec": {"type": "string", "description": "分子规格，可以是原子坐标列表或XYZ文件路径"},
        "basis": {"type": "string", "description": "基组名称，默认为'cc-pvdz'"},
        "charge": {"type": "integer", "description": "分子电荷，默认为0"},
        "spin": {"type": "integer", "description": "自旋多重度减1，默认为0（单重态）"},
        "method": {"type": "string", "description": "计算方法，可选'RHF'（限制性Hartree-Fock）或'DFT'（密度泛函理论），默认为'RHF'"},
        "xc": {"type": "string", "description": "DFT计算中使用的交换关联泛函，默认为None"},
        "grid_level": {"type": "integer", "description": "DFT计算中的网格精度，默认为3"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 run_scf_calculation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "molecule_spec" not in args or args["molecule_spec"] is None:
                return Observation(self.name, "错误: 缺少必需参数 molecule_spec")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import run_scf_calculation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["molecule_spec", "basis", "charge", "spin", "method", "xc", "grid_level"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = run_scf_calculation(**func_kwargs)
            
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


@Toolbox.register(name="optimize_geometry")
class OptimizeGeometryTool(EnvironmentTool):
    """优化分子几何结构"""
    
    name = "optimize_geometry"
    description = "优化分子几何结构"
    arguments = {
        "molecule_spec": {"type": "string", "description": "分子规格，可以是原子坐标列表或XYZ文件路径"},
        "basis": {"type": "string", "description": "基组名称，默认为'cc-pvdz'"},
        "charge": {"type": "integer", "description": "分子电荷，默认为0"},
        "spin": {"type": "integer", "description": "自旋多重度减1，默认为0（单重态）"},
        "method": {"type": "string", "description": "计算方法，默认为'RHF'"},
        "xc": {"type": "string", "description": "DFT计算中使用的交换关联泛函，默认为None"},
        "grid_level": {"type": "integer", "description": "DFT计算中的网格精度，默认为3"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 optimize_geometry 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "molecule_spec" not in args or args["molecule_spec"] is None:
                return Observation(self.name, "错误: 缺少必需参数 molecule_spec")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import optimize_geometry
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["molecule_spec", "basis", "charge", "spin", "method", "xc", "grid_level"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = optimize_geometry(**func_kwargs)
            
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


@Toolbox.register(name="analyze_molecular_properties")
class AnalyzeMolecularPropertiesTool(EnvironmentTool):
    """分析分子性质"""
    
    name = "analyze_molecular_properties"
    description = "分析分子性质"
    arguments = {
        "molecule_spec": {"type": "string", "description": "分子规格，可以是原子坐标列表或XYZ文件路径"},
        "basis": {"type": "string", "description": "基组名称，默认为'cc-pvdz'"},
        "charge": {"type": "integer", "description": "分子电荷，默认为0"},
        "spin": {"type": "integer", "description": "自旋多重度减1，默认为0（单重态）"},
        "method": {"type": "string", "description": "SCF计算方法，可选'RHF'（限制性Hartree-Fock）或'DFT'（密度泛函理论），默认为'RHF'"},
        "xc": {"type": "string", "description": "DFT计算中使用的交换关联泛函，默认为None"},
        "grid_level": {"type": "integer", "description": "DFT计算中的网格精度，默认为3"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_molecular_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "molecule_spec" not in args or args["molecule_spec"] is None:
                return Observation(self.name, "错误: 缺少必需参数 molecule_spec")
            
            # 导入并调用原始函数
            from toolkits.chemistry.computational_chemistry.quantum_chemistry_solver import analyze_molecular_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["molecule_spec", "basis", "charge", "spin", "method", "xc", "grid_level"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_molecular_properties(**func_kwargs)
            
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

