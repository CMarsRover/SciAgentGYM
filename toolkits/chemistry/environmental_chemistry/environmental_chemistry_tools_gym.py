#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
environmental_chemistry 工具注册模块
使用 gym.tool.EnvironmentTool 为 environmental_chemistry 目录中的工具提供统一的注册与调用接口

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


@Toolbox.register(name="calculate_dissolved_oxygen")
class CalculateDissolvedOxygenTool(EnvironmentTool):
    """根据碘量法滴定结果计算溶解氧浓度。原理：Winkler碘量法，通过Na2S2O3标准溶液滴定I2，反应关系：1 mol O2 ≡ 4 mol Na2S2O3。"""
    
    name = "calculate_dissolved_oxygen"
    description = "根据碘量法滴定结果计算溶解氧浓度。原理：Winkler碘量法，通过Na2S2O3标准溶液滴定I2，反应关系：1 mol O2 ≡ 4 mol Na2S2O3。"
    arguments = {
        "na2s2o3_volume": {"type": "number", "description": "Na2S2O3标准溶液消耗体积，单位：mL"},
        "na2s2o3_concentration": {"type": "number", "description": "Na2S2O3标准溶液浓度，单位：mol/L"},
        "sample_volume": {"type": "number", "description": "水样体积，单位：mL"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_dissolved_oxygen 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "na2s2o3_volume" not in args or args["na2s2o3_volume"] is None:
                return Observation(self.name, "错误: 缺少必需参数 na2s2o3_volume")
            if "na2s2o3_concentration" not in args or args["na2s2o3_concentration"] is None:
                return Observation(self.name, "错误: 缺少必需参数 na2s2o3_concentration")
            if "sample_volume" not in args or args["sample_volume"] is None:
                return Observation(self.name, "错误: 缺少必需参数 sample_volume")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import calculate_dissolved_oxygen
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["na2s2o3_volume", "na2s2o3_concentration", "sample_volume"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_dissolved_oxygen(**func_kwargs)
            
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


@Toolbox.register(name="calculate_oxygen_depletion")
class CalculateOxygenDepletionTool(EnvironmentTool):
    """计算样品的净氧消耗量（扣除空白对照）"""
    
    name = "calculate_oxygen_depletion"
    description = "计算样品的净氧消耗量（扣除空白对照）"
    arguments = {
        "do_initial": {"type": "number", "description": "样品初始溶解氧 (mg/L)"},
        "do_final": {"type": "number", "description": "样品第5天溶解氧 (mg/L)"},
        "blank_do_initial": {"type": "number", "description": "空白对照初始溶解氧 (mg/L)"},
        "blank_do_final": {"type": "number", "description": "空白对照第5天溶解氧 (mg/L)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_oxygen_depletion 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "do_initial" not in args or args["do_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 do_initial")
            if "do_final" not in args or args["do_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 do_final")
            if "blank_do_initial" not in args or args["blank_do_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blank_do_initial")
            if "blank_do_final" not in args or args["blank_do_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blank_do_final")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import calculate_oxygen_depletion
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["do_initial", "do_final", "blank_do_initial", "blank_do_final"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_oxygen_depletion(**func_kwargs)
            
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


@Toolbox.register(name="apply_dilution_factor")
class ApplyDilutionFactorTool(EnvironmentTool):
    """应用稀释倍数校正，得到原水样的BOD5值"""
    
    name = "apply_dilution_factor"
    description = "应用稀释倍数校正，得到原水样的BOD5值"
    arguments = {
        "oxygen_depletion": {"type": "number", "description": "稀释水样的净氧消耗量 (mg/L)"},
        "dilution_factor": {"type": "number", "description": "稀释倍数（原水样体积/稀释后总体积）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 apply_dilution_factor 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "oxygen_depletion" not in args or args["oxygen_depletion"] is None:
                return Observation(self.name, "错误: 缺少必需参数 oxygen_depletion")
            if "dilution_factor" not in args or args["dilution_factor"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dilution_factor")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import apply_dilution_factor
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["oxygen_depletion", "dilution_factor"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = apply_dilution_factor(**func_kwargs)
            
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


@Toolbox.register(name="validate_bod5_measurement")
class ValidateBod5MeasurementTool(EnvironmentTool):
    """验证BOD5测定是否符合质量控制标准。根据HJ 505-2009标准：培养后溶解氧应≥2 mg/L；氧消耗量应≥2 mg/L；氧消耗量不应超过初始DO的70%。"""
    
    name = "validate_bod5_measurement"
    description = "验证BOD5测定是否符合质量控制标准。根据HJ 505-2009标准：培养后溶解氧应≥2 mg/L；氧消耗量应≥2 mg/L；氧消耗量不应超过初始DO的70%。"
    arguments = {
        "do_initial": {"type": "number", "description": "初始溶解氧 (mg/L)"},
        "do_final": {"type": "number", "description": "第5天溶解氧 (mg/L)"},
        "min_do_threshold": {"type": "number", "description": "最小剩余DO阈值 (mg/L)，默认2.0"},
        "min_depletion_threshold": {"type": "number", "description": "最小氧消耗阈值 (mg/L)，默认2.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 validate_bod5_measurement 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "do_initial" not in args or args["do_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 do_initial")
            if "do_final" not in args or args["do_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 do_final")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import validate_bod5_measurement
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["do_initial", "do_final", "min_do_threshold", "min_depletion_threshold"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = validate_bod5_measurement(**func_kwargs)
            
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


@Toolbox.register(name="calculate_bod5_from_titration")
class CalculateBod5FromTitrationTool(EnvironmentTool):
    """从滴定数据计算BOD5值（完整流程）"""
    
    name = "calculate_bod5_from_titration"
    description = "从滴定数据计算BOD5值（完整流程）"
    arguments = {
        "sample_data": {"type": "object", "description": "样品数据，包含稀释倍数、样品体积(mL)、Na2S2O3浓度(mol/L)、第0天和第5天Na2S2O3用量(mL)等字段"},
        "blank_data": {"type": "object", "description": "空白对照数据，结构同sample_data（dilution_factor为0）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_bod5_from_titration 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "sample_data" not in args or args["sample_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 sample_data")
            if "blank_data" not in args or args["blank_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blank_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import calculate_bod5_from_titration
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["sample_data", "blank_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_bod5_from_titration(**func_kwargs)
            
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


@Toolbox.register(name="batch_calculate_bod5")
class BatchCalculateBod5Tool(EnvironmentTool):
    """批量计算多个样品的BOD5值"""
    
    name = "batch_calculate_bod5"
    description = "批量计算多个样品的BOD5值"
    arguments = {
        "samples": {"type": "array", "description": "多个样品数据列表；每个样品为字典对象，包含样品相关参数"},
        "blank": {"type": "object", "description": "空白对照数据；字典对象，包含空白对照相关参数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 batch_calculate_bod5 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "samples" not in args or args["samples"] is None:
                return Observation(self.name, "错误: 缺少必需参数 samples")
            if "blank" not in args or args["blank"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blank")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import batch_calculate_bod5
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["samples", "blank"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = batch_calculate_bod5(**func_kwargs)
            
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


@Toolbox.register(name="estimate_optimal_dilution")
class EstimateOptimalDilutionTool(EnvironmentTool):
    """根据COD估算最佳稀释倍数。经验关系：BOD5 ≈ 0.5-0.7 × COD（对于生活污水），目标：使氧消耗量在2-6 mg/L之间。"""
    
    name = "estimate_optimal_dilution"
    description = "根据COD估算最佳稀释倍数。经验关系：BOD5 ≈ 0.5-0.7 × COD（对于生活污水），目标：使氧消耗量在2-6 mg/L之间。"
    arguments = {
        "estimated_cod": {"type": "number", "description": "估算的COD值 (mg/L)"},
        "target_do_depletion_range": {"type": "array", "description": "目标氧消耗范围 (mg/L)，默认为(2.0, 6.0)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 estimate_optimal_dilution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "estimated_cod" not in args or args["estimated_cod"] is None:
                return Observation(self.name, "错误: 缺少必需参数 estimated_cod")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import estimate_optimal_dilution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["estimated_cod", "target_do_depletion_range"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = estimate_optimal_dilution(**func_kwargs)
            
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


@Toolbox.register(name="plot_do_time_series")
class PlotDoTimeSeriesTool(EnvironmentTool):
    """绘制溶解氧随时间变化曲线"""
    
    name = "plot_do_time_series"
    description = "绘制溶解氧随时间变化曲线"
    arguments = {
        "sample_do_day0": {"type": "number", "description": "样品第0天DO (mg/L)"},
        "sample_do_day5": {"type": "number", "description": "样品第5天DO (mg/L)"},
        "blank_do_day0": {"type": "number", "description": "空白第0天DO (mg/L)"},
        "blank_do_day5": {"type": "number", "description": "空白第5天DO (mg/L)"},
        "sample_id": {"type": "string", "description": "样品标识，默认为Sample A"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_do_time_series 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "sample_do_day0" not in args or args["sample_do_day0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 sample_do_day0")
            if "sample_do_day5" not in args or args["sample_do_day5"] is None:
                return Observation(self.name, "错误: 缺少必需参数 sample_do_day5")
            if "blank_do_day0" not in args or args["blank_do_day0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blank_do_day0")
            if "blank_do_day5" not in args or args["blank_do_day5"] is None:
                return Observation(self.name, "错误: 缺少必需参数 blank_do_day5")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import plot_do_time_series
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["sample_do_day0", "sample_do_day5", "blank_do_day0", "blank_do_day5", "sample_id"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_do_time_series(**func_kwargs)
            
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


@Toolbox.register(name="plot_bod5_comparison")
class PlotBod5ComparisonTool(EnvironmentTool):
    """绘制多个样品的BOD5对比柱状图"""
    
    name = "plot_bod5_comparison"
    description = "绘制多个样品的BOD5对比柱状图"
    arguments = {
        "bod5_values": {"type": "array", "description": "BOD5值列表 (mg/L)"},
        "sample_ids": {"type": "array", "description": "样品标识列表"},
        "water_quality_standards": {"type": "object", "description": "水质标准参考值，如 {'Class I': 3, 'Class II': 6, 'Class III': 10}"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_bod5_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "bod5_values" not in args or args["bod5_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 bod5_values")
            if "sample_ids" not in args or args["sample_ids"] is None:
                return Observation(self.name, "错误: 缺少必需参数 sample_ids")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import plot_bod5_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["bod5_values", "sample_ids", "water_quality_standards"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_bod5_comparison(**func_kwargs)
            
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


@Toolbox.register(name="plot_dilution_optimization")
class PlotDilutionOptimizationTool(EnvironmentTool):
    """绘制稀释倍数优化曲线，展示不同COD值对应的推荐稀释倍数。"""
    
    name = "plot_dilution_optimization"
    description = "绘制稀释倍数优化曲线，展示不同COD值对应的推荐稀释倍数。"
    arguments = {
        "cod_range": {"type": "array", "description": "COD范围 (mg/L)，形式为(cod_min, cod_max)"},
        "num_points": {"type": "integer", "description": "曲线采样点数，默认为50"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_dilution_optimization 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "cod_range" not in args or args["cod_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 cod_range")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.environmental_chemistry_toolkit_claude_654 import plot_dilution_optimization
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cod_range", "num_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_dilution_optimization(**func_kwargs)
            
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
    """Demonstrate the wastewater nitrogen analysis toolkit with three scenarios."""
    
    name = "main"
    description = "Demonstrate the wastewater nitrogen analysis toolkit with three scenarios."
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
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import main
            
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


@Toolbox.register(name="validate_waste_component")
class ValidateWasteComponentTool(EnvironmentTool):
    """验证单个废物组分的数据。"""
    
    name = "validate_waste_component"
    description = "验证单个废物组分的数据。"
    arguments = {
        "component_name": {"type": "string", "description": "废物组分的名称"},
        "weight_percentage": {"type": "number", "description": "重量百分比（0-100）"},
        "heat_value": {"type": "number", "description": "热值，单位：kJ/kg（>= 0）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 validate_waste_component 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "component_name" not in args or args["component_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 component_name")
            if "weight_percentage" not in args or args["weight_percentage"] is None:
                return Observation(self.name, "错误: 缺少必需参数 weight_percentage")
            if "heat_value" not in args or args["heat_value"] is None:
                return Observation(self.name, "错误: 缺少必需参数 heat_value")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import validate_waste_component
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["component_name", "weight_percentage", "heat_value"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = validate_waste_component(**func_kwargs)
            
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


@Toolbox.register(name="calculate_component_contribution")
class CalculateComponentContributionTool(EnvironmentTool):
    """计算单个组分的热值贡献。公式：贡献 = (重量百分比 / 100) * 热值"""
    
    name = "calculate_component_contribution"
    description = "计算单个组分的热值贡献。公式：贡献 = (重量百分比 / 100) * 热值"
    arguments = {
        "weight_percentage": {"type": "number", "description": "组分的重量百分比（0-100）"},
        "heat_value": {"type": "number", "description": "热值，单位：kJ/kg"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_component_contribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "weight_percentage" not in args or args["weight_percentage"] is None:
                return Observation(self.name, "错误: 缺少必需参数 weight_percentage")
            if "heat_value" not in args or args["heat_value"] is None:
                return Observation(self.name, "错误: 缺少必需参数 heat_value")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import calculate_component_contribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["weight_percentage", "heat_value"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_component_contribution(**func_kwargs)
            
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


@Toolbox.register(name="sum_contributions")
class SumContributionsTool(EnvironmentTool):
    """计算所有组分贡献值的总和，得到总热值。"""
    
    name = "sum_contributions"
    description = "计算所有组分贡献值的总和，得到总热值。"
    arguments = {
        "contributions": {"type": "array", "description": "各组分贡献值列表，单位：kJ/kg"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 sum_contributions 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "contributions" not in args or args["contributions"] is None:
                return Observation(self.name, "错误: 缺少必需参数 contributions")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import sum_contributions
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["contributions"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = sum_contributions(**func_kwargs)
            
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


@Toolbox.register(name="validate_total_percentage")
class ValidateTotalPercentageTool(EnvironmentTool):
    """验证总重量百分比之和是否约等于100%。"""
    
    name = "validate_total_percentage"
    description = "验证总重量百分比之和是否约等于100%。"
    arguments = {
        "weight_percentages": {"type": "array", "description": "重量百分比列表"},
        "tolerance": {"type": "number", "description": "与100%的可接受偏差，默认为0.01"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 validate_total_percentage 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "weight_percentages" not in args or args["weight_percentages"] is None:
                return Observation(self.name, "错误: 缺少必需参数 weight_percentages")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import validate_total_percentage
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["weight_percentages", "tolerance"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = validate_total_percentage(**func_kwargs)
            
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


@Toolbox.register(name="calculate_waste_calorific_value")
class CalculateWasteCalorificValueTool(EnvironmentTool):
    """计算城市固体废物的加权平均热值。该函数结合验证、贡献计算和求和来计算总体热值。"""
    
    name = "calculate_waste_calorific_value"
    description = "计算城市固体废物的加权平均热值。该函数结合验证、贡献计算和求和来计算总体热值。"
    arguments = {
        "waste_data": {"type": "array", "description": "废物数据列表，每个元素包含组分名称、重量百分比和热值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_waste_calorific_value 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "waste_data" not in args or args["waste_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 waste_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import calculate_waste_calorific_value
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["waste_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_waste_calorific_value(**func_kwargs)
            
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


@Toolbox.register(name="analyze_waste_composition")
class AnalyzeWasteCompositionTool(EnvironmentTool):
    """对废物成分进行综合分析。"""
    
    name = "analyze_waste_composition"
    description = "对废物成分进行综合分析。"
    arguments = {
        "waste_data": {"type": "array", "description": "废物组分字典列表，每个字典包含组分名称、重量百分比、热值等字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_waste_composition 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "waste_data" not in args or args["waste_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 waste_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import analyze_waste_composition
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["waste_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_waste_composition(**func_kwargs)
            
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


@Toolbox.register(name="save_calculation_report")
class SaveCalculationReportTool(EnvironmentTool):
    """将详细的计算报告保存到JSON文件。"""
    
    name = "save_calculation_report"
    description = "将详细的计算报告保存到JSON文件。"
    arguments = {
        "waste_data": {"type": "array", "description": "废物组分字典列表，每个字典的键为字符串，值为浮点数"},
        "output_filename": {"type": "string", "description": "输出文件名，默认为waste_calorific_calculation.json"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 save_calculation_report 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "waste_data" not in args or args["waste_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 waste_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import save_calculation_report
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["waste_data", "output_filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = save_calculation_report(**func_kwargs)
            
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


@Toolbox.register(name="plot_waste_composition_pie")
class PlotWasteCompositionPieTool(EnvironmentTool):
    """创建饼图展示垃圾成分按重量百分比的组成。"""
    
    name = "plot_waste_composition_pie"
    description = "创建饼图展示垃圾成分按重量百分比的组成。"
    arguments = {
        "waste_data": {"type": "array", "description": "垃圾成分字典列表，每个字典包含component（成分名称）、weight_percentage（重量百分比）、heat_value（热值）等字段"},
        "output_filename": {"type": "string", "description": "输出图像文件名，默认为waste_composition_pie.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_waste_composition_pie 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "waste_data" not in args or args["waste_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 waste_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import plot_waste_composition_pie
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["waste_data", "output_filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_waste_composition_pie(**func_kwargs)
            
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


@Toolbox.register(name="plot_calorific_contribution_bar")
class PlotCalorificContributionBarTool(EnvironmentTool):
    """创建显示每个组分热值贡献的柱状图。"""
    
    name = "plot_calorific_contribution_bar"
    description = "创建显示每个组分热值贡献的柱状图。"
    arguments = {
        "waste_data": {"type": "array", "description": "废物组分字典列表；每个字典包含组分名称、重量百分比、热值等字段"},
        "output_filename": {"type": "string", "description": "输出图像文件名，默认为calorific_contribution_bar.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_calorific_contribution_bar 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "waste_data" not in args or args["waste_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 waste_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import plot_calorific_contribution_bar
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["waste_data", "output_filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_calorific_contribution_bar(**func_kwargs)
            
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


@Toolbox.register(name="plot_heat_value_vs_percentage")
class PlotHeatValueVsPercentageTool(EnvironmentTool):
    """创建散点图，展示热值与重量百分比之间的关系。"""
    
    name = "plot_heat_value_vs_percentage"
    description = "创建散点图，展示热值与重量百分比之间的关系。"
    arguments = {
        "waste_data": {"type": "array", "description": "废物组分字典列表，每个字典包含组分名称、热值、重量百分比等字段"},
        "output_filename": {"type": "string", "description": "输出图像文件名，默认为 heat_value_vs_percentage.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_heat_value_vs_percentage 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "waste_data" not in args or args["waste_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 waste_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import plot_heat_value_vs_percentage
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["waste_data", "output_filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_heat_value_vs_percentage(**func_kwargs)
            
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


@Toolbox.register(name="load_json_file")
class LoadJsonFileTool(EnvironmentTool):
    """加载并解析JSON文件。"""
    
    name = "load_json_file"
    description = "加载并解析JSON文件。"
    arguments = {
        "filepath": {"type": "string", "description": "JSON文件的路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 load_json_file 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "filepath" not in args or args["filepath"] is None:
                return Observation(self.name, "错误: 缺少必需参数 filepath")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.waste_calorific_value_toolkit_claude_650 import load_json_file
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["filepath"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = load_json_file(**func_kwargs)
            
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


@Toolbox.register(name="calculate_total_nitrogen")
class CalculateTotalNitrogenTool(EnvironmentTool):
    """计算总氮浓度，通过累加有机氮、氨氮、亚硝酸盐氮和硝酸盐氮的浓度。"""
    
    name = "calculate_total_nitrogen"
    description = "计算总氮浓度，通过累加有机氮、氨氮、亚硝酸盐氮和硝酸盐氮的浓度。"
    arguments = {
        "organic_n": {"type": "number", "description": "有机氮浓度，单位：mg/L"},
        "nh3_n": {"type": "number", "description": "氨氮浓度，单位：mg/L"},
        "no2_n": {"type": "number", "description": "亚硝酸盐氮浓度，单位：mg/L"},
        "no3_n": {"type": "number", "description": "硝酸盐氮浓度，单位：mg/L"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_total_nitrogen 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "organic_n" not in args or args["organic_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 organic_n")
            if "nh3_n" not in args or args["nh3_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 nh3_n")
            if "no2_n" not in args or args["no2_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 no2_n")
            if "no3_n" not in args or args["no3_n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 no3_n")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import calculate_total_nitrogen
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["organic_n", "nh3_n", "no2_n", "no3_n"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_total_nitrogen(**func_kwargs)
            
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


@Toolbox.register(name="calculate_removal_efficiency")
class CalculateRemovalEfficiencyTool(EnvironmentTool):
    """计算去除效率百分比，基于进水和出水浓度。"""
    
    name = "calculate_removal_efficiency"
    description = "计算去除效率百分比，基于进水和出水浓度。"
    arguments = {
        "inflow": {"type": "number", "description": "进水浓度，单位：mg/L"},
        "outflow": {"type": "number", "description": "出水浓度，单位：mg/L"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_removal_efficiency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "inflow" not in args or args["inflow"] is None:
                return Observation(self.name, "错误: 缺少必需参数 inflow")
            if "outflow" not in args or args["outflow"] is None:
                return Observation(self.name, "错误: 缺少必需参数 outflow")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import calculate_removal_efficiency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["inflow", "outflow"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_removal_efficiency(**func_kwargs)
            
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


@Toolbox.register(name="calculate_mass_balance")
class CalculateMassBalanceTool(EnvironmentTool):
    """计算氮的质量平衡，包括总输入、总输出和总去除量。"""
    
    name = "calculate_mass_balance"
    description = "计算氮的质量平衡，包括总输入、总输出和总去除量。"
    arguments = {
        "inflow": {"type": "object", "description": "进水数据字典，包含organic_n、nh3_n、no2_n、no3_n字段"},
        "outflow": {"type": "object", "description": "出水数据字典，包含organic_n、nh3_n、no2_n、no3_n字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_mass_balance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "inflow" not in args or args["inflow"] is None:
                return Observation(self.name, "错误: 缺少必需参数 inflow")
            if "outflow" not in args or args["outflow"] is None:
                return Observation(self.name, "错误: 缺少必需参数 outflow")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import calculate_mass_balance
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["inflow", "outflow"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_mass_balance(**func_kwargs)
            
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


@Toolbox.register(name="analyze_nitrogen_removal")
class AnalyzeNitrogenRemovalTool(EnvironmentTool):
    """综合分析污水处理系统的氮去除效率，包括总氮、有机氮、无机氮及各氮形态的去除率。"""
    
    name = "analyze_nitrogen_removal"
    description = "综合分析污水处理系统的氮去除效率，包括总氮、有机氮、无机氮及各氮形态的去除率。"
    arguments = {
        "wastewater_data": {"type": "object", "description": "污水数据字典，包含'inflow'和'outflow'两个子字典，每个子字典包含organic_n、nh3_n、no2_n、no3_n字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_nitrogen_removal 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wastewater_data" not in args or args["wastewater_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wastewater_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import analyze_nitrogen_removal
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wastewater_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_nitrogen_removal(**func_kwargs)
            
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


@Toolbox.register(name="evaluate_nitrification_denitrification")
class EvaluateNitrificationDenitrificationTool(EnvironmentTool):
    """评估污水处理过程中的硝化和反硝化过程，量化氮的转化情况。"""
    
    name = "evaluate_nitrification_denitrification"
    description = "评估污水处理过程中的硝化和反硝化过程，量化氮的转化情况。"
    arguments = {
        "wastewater_data": {"type": "object", "description": "污水数据字典，包含'inflow'和'outflow'两个子字典，每个子字典包含organic_n、nh3_n、no2_n、no3_n字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 evaluate_nitrification_denitrification 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wastewater_data" not in args or args["wastewater_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wastewater_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import evaluate_nitrification_denitrification
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wastewater_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = evaluate_nitrification_denitrification(**func_kwargs)
            
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


@Toolbox.register(name="visualize_nitrogen_distribution")
class VisualizeNitrogenDistributionTool(EnvironmentTool):
    """可视化进水和出水中各氮形态的分布情况，生成柱状图。"""
    
    name = "visualize_nitrogen_distribution"
    description = "可视化进水和出水中各氮形态的分布情况，生成柱状图。"
    arguments = {
        "wastewater_data": {"type": "object", "description": "污水数据字典，包含'inflow'和'outflow'两个子字典，每个子字典包含organic_n、nh3_n、no2_n、no3_n字段"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_nitrogen_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wastewater_data" not in args or args["wastewater_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wastewater_data")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import visualize_nitrogen_distribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wastewater_data"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_nitrogen_distribution(**func_kwargs)
            
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


@Toolbox.register(name="visualize_removal_efficiency")
class VisualizeRemovalEfficiencyTool(EnvironmentTool):
    """可视化各氮形态的去除效率，生成条形图。"""
    
    name = "visualize_removal_efficiency"
    description = "可视化各氮形态的去除效率，生成条形图。"
    arguments = {
        "analysis_result": {"type": "object", "description": "氮去除分析结果，由analyze_nitrogen_removal函数返回的结果对象"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_removal_efficiency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "analysis_result" not in args or args["analysis_result"] is None:
                return Observation(self.name, "错误: 缺少必需参数 analysis_result")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import visualize_removal_efficiency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["analysis_result"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_removal_efficiency(**func_kwargs)
            
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


@Toolbox.register(name="visualize_nitrogen_transformation")
class VisualizeNitrogenTransformationTool(EnvironmentTool):
    """可视化氮的转化过程，包括硝化和反硝化过程的流程图。"""
    
    name = "visualize_nitrogen_transformation"
    description = "可视化氮的转化过程，包括硝化和反硝化过程的流程图。"
    arguments = {
        "wastewater_data": {"type": "object", "description": "污水数据字典，包含'inflow'和'outflow'两个子字典，每个子字典包含organic_n、nh3_n、no2_n、no3_n字段"},
        "bio_process_result": {"type": "object", "description": "生物过程评估结果，由evaluate_nitrification_denitrification函数返回的结果对象"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_nitrogen_transformation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wastewater_data" not in args or args["wastewater_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wastewater_data")
            if "bio_process_result" not in args or args["bio_process_result"] is None:
                return Observation(self.name, "错误: 缺少必需参数 bio_process_result")
            
            # 导入并调用原始函数
            from toolkits.chemistry.environmental_chemistry.wastewater_nitrogen_toolkit_claude_652 import visualize_nitrogen_transformation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wastewater_data", "bio_process_result"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_nitrogen_transformation(**func_kwargs)
            
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

