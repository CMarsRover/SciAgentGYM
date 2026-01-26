#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
acoustics 工具注册模块
使用 gym.tool.EnvironmentTool 为 acoustics 目录中的工具提供统一的注册与调用接口

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


@Toolbox.register(name="hydrogen_wavefunction_radial")
class HydrogenWavefunctionRadialTool(EnvironmentTool):
    """计算氢原子波函数的径向部分 R_nl(r)。"""
    
    name = "hydrogen_wavefunction_radial"
    description = "计算氢原子波函数的径向部分 R_nl(r)。"
    arguments = {
        "n": {"type": "integer", "description": "主量子数 (n >= 1)"},
        "l": {"type": "integer", "description": "轨道角动量量子数 (0 <= l < n)"},
        "r_over_a0": {"type": "number", "description": "以玻尔半径为单位的径向距离 (r/a0)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 hydrogen_wavefunction_radial 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n" not in args or args["n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n")
            if "l" not in args or args["l"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l")
            if "r_over_a0" not in args or args["r_over_a0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 r_over_a0")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import hydrogen_wavefunction_radial
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n", "l", "r_over_a0"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = hydrogen_wavefunction_radial(**func_kwargs)
            
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


@Toolbox.register(name="dipole_matrix_element_radial")
class DipoleMatrixElementRadialTool(EnvironmentTool):
    """计算电偶极矩阵元的径向部分 <n1,l1|r|n2,l2>。这是径向积分：∫ R_{n1,l1}(r) * r * R_{n2,l2}(r) * r² dr"""
    
    name = "dipole_matrix_element_radial"
    description = "计算电偶极矩阵元的径向部分 <n1,l1|r|n2,l2>。这是径向积分：∫ R_{n1,l1}(r) * r * R_{n2,l2}(r) * r² dr"
    arguments = {
        "n1": {"type": "integer", "description": "初始态主量子数"},
        "l1": {"type": "integer", "description": "初始态轨道角量子数"},
        "n2": {"type": "integer", "description": "末态主量子数"},
        "l2": {"type": "integer", "description": "末态轨道角量子数"},
        "num_points": {"type": "integer", "description": "积分点数，默认为1000"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 dipole_matrix_element_radial 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n1" not in args or args["n1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            if "l1" not in args or args["l1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l1")
            if "n2" not in args or args["n2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            if "l2" not in args or args["l2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l2")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import dipole_matrix_element_radial
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n1", "l1", "n2", "l2", "num_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = dipole_matrix_element_radial(**func_kwargs)
            
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


@Toolbox.register(name="stark_shift_first_order")
class StarkShiftFirstOrderTool(EnvironmentTool):
    """计算氢原子态 |n,l,m> 的一阶斯塔克位移。一阶位移仅存在于简并态（n >= 2）。"""
    
    name = "stark_shift_first_order"
    description = "计算氢原子态 |n,l,m> 的一阶斯塔克位移。一阶位移仅存在于简并态（n >= 2）。"
    arguments = {
        "n": {"type": "integer", "description": "主量子数"},
        "l": {"type": "integer", "description": "轨道角动量量子数"},
        "m": {"type": "integer", "description": "磁量子数"},
        "E0_SI": {"type": "number", "description": "直流电场强度，单位：V/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 stark_shift_first_order 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n" not in args or args["n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n")
            if "l" not in args or args["l"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l")
            if "m" not in args or args["m"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m")
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import stark_shift_first_order
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n", "l", "m", "E0_SI"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = stark_shift_first_order(**func_kwargs)
            
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


@Toolbox.register(name="stark_shift_second_order")
class StarkShiftSecondOrderTool(EnvironmentTool):
    """计算氢原子的二阶（平方）Stark位移。"""
    
    name = "stark_shift_second_order"
    description = "计算氢原子的二阶（平方）Stark位移。"
    arguments = {
        "n": {"type": "integer", "description": "主量子数"},
        "l": {"type": "integer", "description": "轨道角动量量子数"},
        "E0_SI": {"type": "number", "description": "直流电场振幅，单位：V/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 stark_shift_second_order 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n" not in args or args["n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n")
            if "l" not in args or args["l"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l")
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import stark_shift_second_order
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n", "l", "E0_SI"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = stark_shift_second_order(**func_kwargs)
            
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


@Toolbox.register(name="transition_frequency_with_stark")
class TransitionFrequencyWithStarkTool(EnvironmentTool):
    """计算包含斯塔克位移的两个态之间的跃迁频率。"""
    
    name = "transition_frequency_with_stark"
    description = "计算包含斯塔克位移的两个态之间的跃迁频率。"
    arguments = {
        "n1": {"type": "integer", "description": "初态主量子数"},
        "l1": {"type": "integer", "description": "初态轨道角动量量子数"},
        "n2": {"type": "integer", "description": "末态主量子数"},
        "l2": {"type": "integer", "description": "末态轨道角动量量子数"},
        "E0_SI": {"type": "number", "description": "直流电场强度，单位：V/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 transition_frequency_with_stark 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n1" not in args or args["n1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            if "l1" not in args or args["l1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l1")
            if "n2" not in args or args["n2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            if "l2" not in args or args["l2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l2")
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import transition_frequency_with_stark
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n1", "l1", "n2", "l2", "E0_SI"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = transition_frequency_with_stark(**func_kwargs)
            
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


@Toolbox.register(name="rabi_frequency")
class RabiFrequencyTool(EnvironmentTool):
    """计算原子-场相互作用的拉比频率。公式：Ω = μ·E / ℏ，其中μ是跃迁偶极矩，E是场振幅。"""
    
    name = "rabi_frequency"
    description = "计算原子-场相互作用的拉比频率。公式：Ω = μ·E / ℏ，其中μ是跃迁偶极矩，E是场振幅。"
    arguments = {
        "dipole_moment_au": {"type": "number", "description": "跃迁偶极矩，单位：原子单位（e*a0）"},
        "field_amplitude_SI": {"type": "number", "description": "电场振幅，单位：V/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 rabi_frequency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "dipole_moment_au" not in args or args["dipole_moment_au"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dipole_moment_au")
            if "field_amplitude_SI" not in args or args["field_amplitude_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 field_amplitude_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import rabi_frequency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["dipole_moment_au", "field_amplitude_SI"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = rabi_frequency(**func_kwargs)
            
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


@Toolbox.register(name="absorption_probability_two_level")
class AbsorptionProbabilityTwoLevelTool(EnvironmentTool):
    """计算二能级系统的吸收概率。"""
    
    name = "absorption_probability_two_level"
    description = "计算二能级系统的吸收概率。"
    arguments = {
        "detuning": {"type": "number", "description": "频率失谐 (ω - ω₀)，单位：rad/s"},
        "rabi_freq": {"type": "number", "description": "拉比频率，单位：rad/s"},
        "interaction_time": {"type": "number", "description": "相互作用时间，单位：秒"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 absorption_probability_two_level 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "detuning" not in args or args["detuning"] is None:
                return Observation(self.name, "错误: 缺少必需参数 detuning")
            if "rabi_freq" not in args or args["rabi_freq"] is None:
                return Observation(self.name, "错误: 缺少必需参数 rabi_freq")
            if "interaction_time" not in args or args["interaction_time"] is None:
                return Observation(self.name, "错误: 缺少必需参数 interaction_time")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import absorption_probability_two_level
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["detuning", "rabi_freq", "interaction_time"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = absorption_probability_two_level(**func_kwargs)
            
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


@Toolbox.register(name="polarization_selection_rule")
class PolarizationSelectionRuleTool(EnvironmentTool):
    """检查不同偏振下电偶极跃迁的选择定则。"""
    
    name = "polarization_selection_rule"
    description = "检查不同偏振下电偶极跃迁的选择定则。"
    arguments = {
        "l_initial": {"type": "integer", "description": "初始轨道量子数"},
        "m_initial": {"type": "integer", "description": "初始磁量子数"},
        "l_final": {"type": "integer", "description": "末态轨道量子数"},
        "m_final": {"type": "integer", "description": "末态磁量子数"},
        "polarization_type": {"type": "string", "description": "偏振类型，可选值：'linear_z'（线偏振，π跃迁）、'circular_plus'（σ+圆偏振）或'circular_minus'（σ-圆偏振）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 polarization_selection_rule 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "l_initial" not in args or args["l_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_initial")
            if "m_initial" not in args or args["m_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_initial")
            if "l_final" not in args or args["l_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_final")
            if "m_final" not in args or args["m_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_final")
            if "polarization_type" not in args or args["polarization_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 polarization_type")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import polarization_selection_rule
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["l_initial", "m_initial", "l_final", "m_final", "polarization_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = polarization_selection_rule(**func_kwargs)
            
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


@Toolbox.register(name="calculate_1s_2p_transition_dipole")
class Calculate1s2pTransitionDipoleTool(EnvironmentTool):
    """计算氢原子1s到2p跃迁的跃迁偶极矩。径向矩阵元约为0.7449*a0，角向部分取决于m_final。"""
    
    name = "calculate_1s_2p_transition_dipole"
    description = "计算氢原子1s到2p跃迁的跃迁偶极矩。径向矩阵元约为0.7449*a0，角向部分取决于m_final。"
    arguments = {
        "m_final": {"type": "integer", "description": "末态磁量子数，取值为-1、0或1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_1s_2p_transition_dipole 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "m_final" not in args or args["m_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_final")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import calculate_1s_2p_transition_dipole
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["m_final"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_1s_2p_transition_dipole(**func_kwargs)
            
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


@Toolbox.register(name="analyze_dual_field_absorption")
class AnalyzeDualFieldAbsorptionTool(EnvironmentTool):
    """分析双电磁场对1s->2p跃迁的吸收概率。"""
    
    name = "analyze_dual_field_absorption"
    description = "分析双电磁场对1s->2p跃迁的吸收概率。"
    arguments = {
        "E1": {"type": "number", "description": "场1的振幅，单位：V/m"},
        "E2": {"type": "number", "description": "场2的振幅，单位：V/m"},
        "E0_SI": {"type": "number", "description": "直流电场振幅，单位：V/m"},
        "k": {"type": "number", "description": "波矢幅度，单位：m^-1"},
        "w1": {"type": "number", "description": "场1的频率，单位：rad/s"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_dual_field_absorption 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "E1" not in args or args["E1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E1")
            if "E2" not in args or args["E2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E2")
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            if "k" not in args or args["k"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k")
            if "w1" not in args or args["w1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 w1")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import analyze_dual_field_absorption
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["E1", "E2", "E0_SI", "k", "w1"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_dual_field_absorption(**func_kwargs)
            
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


@Toolbox.register(name="calculate_w2_for_equal_absorption")
class CalculateW2ForEqualAbsorptionTool(EnvironmentTool):
    """计算场#2的频率w2，使得吸收概率相等。对于相等的最大吸收，两个场必须与各自的跃迁共振，且拉比频率应相等（或吸收截面相等）。关键洞察：斯塔克效应根据m值对2p态产生不同的移位。对于圆偏振，频率必须匹配斯塔克移位后的2p(m=±1)跃迁。"""
    
    name = "calculate_w2_for_equal_absorption"
    description = "计算场#2的频率w2，使得吸收概率相等。对于相等的最大吸收，两个场必须与各自的跃迁共振，且拉比频率应相等（或吸收截面相等）。关键洞察：斯塔克效应根据m值对2p态产生不同的移位。对于圆偏振，频率必须匹配斯塔克移位后的2p(m=±1)跃迁。"
    arguments = {
        "E1": {"type": "number", "description": "场#1的振幅，单位：V/m"},
        "E2": {"type": "number", "description": "场#2的振幅，单位：V/m"},
        "E0_SI": {"type": "number", "description": "直流电场振幅，单位：V/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_w2_for_equal_absorption 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "E1" not in args or args["E1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E1")
            if "E2" not in args or args["E2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E2")
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import calculate_w2_for_equal_absorption
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["E1", "E2", "E0_SI"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_w2_for_equal_absorption(**func_kwargs)
            
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


@Toolbox.register(name="derive_stark_frequency_formula")
class DeriveStarkFrequencyFormulaTool(EnvironmentTool):
    """推导w2关于基本常数的解析公式。"""
    
    name = "derive_stark_frequency_formula"
    description = "推导w2关于基本常数的解析公式。"
    arguments = {
        "E0_SI": {"type": "number", "description": "直流电场振幅，单位：V/m"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 derive_stark_frequency_formula 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import derive_stark_frequency_formula
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["E0_SI"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = derive_stark_frequency_formula(**func_kwargs)
            
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


@Toolbox.register(name="plot_stark_energy_levels")
class PlotStarkEnergyLevelsTool(EnvironmentTool):
    """绘制显示1s和2p态Stark位移的能级图。"""
    
    name = "plot_stark_energy_levels"
    description = "绘制显示1s和2p态Stark位移的能级图。"
    arguments = {
        "E0_range_SI": {"type": "array", "description": "直流场强度列表，单位：V/m"},
        "save_path": {"type": "string", "description": "保存图形的可选路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_stark_energy_levels 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "E0_range_SI" not in args or args["E0_range_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_range_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import plot_stark_energy_levels
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["E0_range_SI", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_stark_energy_levels(**func_kwargs)
            
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


@Toolbox.register(name="plot_absorption_vs_frequency")
class PlotAbsorptionVsFrequencyTool(EnvironmentTool):
    """绘制两个场的吸收概率随频率变化的图。"""
    
    name = "plot_absorption_vs_frequency"
    description = "绘制两个场的吸收概率随频率变化的图。"
    arguments = {
        "w_range": {"type": "array", "description": "频率列表，单位：rad/s"},
        "E1": {"type": "number", "description": "场1的振幅，单位：V/m"},
        "E2": {"type": "number", "description": "场2的振幅，单位：V/m"},
        "E0_SI": {"type": "number", "description": "直流场振幅，单位：V/m"},
        "save_path": {"type": "string", "description": "保存图形的可选路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_absorption_vs_frequency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "w_range" not in args or args["w_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 w_range")
            if "E1" not in args or args["E1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E1")
            if "E2" not in args or args["E2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E2")
            if "E0_SI" not in args or args["E0_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import plot_absorption_vs_frequency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["w_range", "E1", "E2", "E0_SI", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_absorption_vs_frequency(**func_kwargs)
            
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


@Toolbox.register(name="plot_w2_vs_E0")
class PlotW2VsE0Tool(EnvironmentTool):
    """绘制所需w2频率随直流电场强度变化的图。"""
    
    name = "plot_w2_vs_E0"
    description = "绘制所需w2频率随直流电场强度变化的图。"
    arguments = {
        "E0_range_SI": {"type": "array", "description": "直流场强度列表，单位：V/m"},
        "save_path": {"type": "string", "description": "保存图形的可选路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_w2_vs_E0 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "E0_range_SI" not in args or args["E0_range_SI"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E0_range_SI")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.hydrogen_stark_spectroscopy_toolkit_claude_303 import plot_w2_vs_E0
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["E0_range_SI", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_w2_vs_E0(**func_kwargs)
            
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
    """Demonstrate the acoustics toolkit with three scenarios."""
    
    name = "main"
    description = "Demonstrate the acoustics toolkit with three scenarios."
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
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import main
            
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


@Toolbox.register(name="convert_spl_to_pressure")
class ConvertSplToPressureTool(EnvironmentTool):
    """将声压级（SPL）转换为有效声压值。"""
    
    name = "convert_spl_to_pressure"
    description = "将声压级（SPL）转换为有效声压值。"
    arguments = {
        "spl": {"type": "number", "description": "声压级值，单位：dB"},
        "p_ref": {"type": "number", "description": "参考声压，单位：Pa，默认为2e-5 Pa（空气中标准参考声压）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 convert_spl_to_pressure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spl" not in args or args["spl"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import convert_spl_to_pressure
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spl", "p_ref"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = convert_spl_to_pressure(**func_kwargs)
            
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


@Toolbox.register(name="convert_pressure_to_spl")
class ConvertPressureToSplTool(EnvironmentTool):
    """将有效声压值转换为声压级（SPL）。"""
    
    name = "convert_pressure_to_spl"
    description = "将有效声压值转换为声压级（SPL）。"
    arguments = {
        "pressure": {"type": "number", "description": "有效声压值，单位：Pa"},
        "p_ref": {"type": "number", "description": "参考声压，单位：Pa，默认为2e-5 Pa（空气中标准参考声压）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 convert_pressure_to_spl 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "pressure" not in args or args["pressure"] is None:
                return Observation(self.name, "错误: 缺少必需参数 pressure")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import convert_pressure_to_spl
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["pressure", "p_ref"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = convert_pressure_to_spl(**func_kwargs)
            
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


@Toolbox.register(name="get_spl_increase_from_table")
class GetSplIncreaseFromTableTool(EnvironmentTool):
    """根据声压级差值从查找表获取增量值，使用线性插值处理表项之间的值。"""
    
    name = "get_spl_increase_from_table"
    description = "根据声压级差值从查找表获取增量值，使用线性插值处理表项之间的值。"
    arguments = {
        "spl_difference": {"type": "number", "description": "两个声压级值之间的绝对差值（单位：dB），必须为非负数"},
        "lookup_table": {"type": "object", "description": "可选的自定义查找表，键为整数差值，值为对应的增量（默认使用全局表）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_spl_increase_from_table 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spl_difference" not in args or args["spl_difference"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl_difference")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import get_spl_increase_from_table
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spl_difference", "lookup_table"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_spl_increase_from_table(**func_kwargs)
            
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


@Toolbox.register(name="add_two_spl_values")
class AddTwoSplValuesTool(EnvironmentTool):
    """使用查表法或精确公式叠加两个声压级。"""
    
    name = "add_two_spl_values"
    description = "使用查表法或精确公式叠加两个声压级。"
    arguments = {
        "spl1_db": {"type": "number", "description": "第一个声压级，单位：dB"},
        "spl2_db": {"type": "number", "description": "第二个声压级，单位：dB"},
        "use_table": {"type": "boolean", "description": "若为True，使用查表法；若为False，使用精确公式，默认为True"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 add_two_spl_values 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spl1_db" not in args or args["spl1_db"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl1_db")
            if "spl2_db" not in args or args["spl2_db"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl2_db")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import add_two_spl_values
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spl1_db", "spl2_db", "use_table"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = add_two_spl_values(**func_kwargs)
            
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


@Toolbox.register(name="calculate_total_spl_multiple_sources")
class CalculateTotalSplMultipleSourcesTool(EnvironmentTool):
    """计算多个声源的总声压级。"""
    
    name = "calculate_total_spl_multiple_sources"
    description = "计算多个声源的总声压级。"
    arguments = {
        "spl_list": {"type": "array", "description": "声压级列表（单位：dB）"},
        "use_table": {"type": "boolean", "description": "如果为True，使用查找表；如果为False，使用精确公式"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_total_spl_multiple_sources 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spl_list" not in args or args["spl_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl_list")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import calculate_total_spl_multiple_sources
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spl_list", "use_table"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_total_spl_multiple_sources(**func_kwargs)
            
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


@Toolbox.register(name="compare_table_vs_exact_methods")
class CompareTableVsExactMethodsTool(EnvironmentTool):
    """比较查表法与精确公式法的计算结果。"""
    
    name = "compare_table_vs_exact_methods"
    description = "比较查表法与精确公式法的计算结果。"
    arguments = {
        "spl_list": {"type": "array", "description": "声压级列表，单位：分贝（dB）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_table_vs_exact_methods 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spl_list" not in args or args["spl_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl_list")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import compare_table_vs_exact_methods
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spl_list"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_table_vs_exact_methods(**func_kwargs)
            
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


@Toolbox.register(name="calculate_spl_at_distance")
class CalculateSplAtDistanceTool(EnvironmentTool):
    """计算声源在不同距离处的声压级（SPL）。自由场：SPL2 = SPL1 - 20*log10(r2/r1)；扩散场：SPL2 = SPL1 - 10*log10(r2/r1)"""
    
    name = "calculate_spl_at_distance"
    description = "计算声源在不同距离处的声压级（SPL）。自由场：SPL2 = SPL1 - 20*log10(r2/r1)；扩散场：SPL2 = SPL1 - 10*log10(r2/r1)"
    arguments = {
        "source_spl": {"type": "number", "description": "声源距离处的声压级，单位：dB"},
        "source_distance": {"type": "number", "description": "测量source_spl的距离，单位：米"},
        "target_distance": {"type": "number", "description": "需要计算声压级的目标距离，单位：米"},
        "environment": {"type": "string", "description": "声场环境类型，可选值：'free_field'（自由场）或 'diffuse_field'（扩散场），默认为'free_field'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_spl_at_distance 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "source_spl" not in args or args["source_spl"] is None:
                return Observation(self.name, "错误: 缺少必需参数 source_spl")
            if "source_distance" not in args or args["source_distance"] is None:
                return Observation(self.name, "错误: 缺少必需参数 source_distance")
            if "target_distance" not in args or args["target_distance"] is None:
                return Observation(self.name, "错误: 缺少必需参数 target_distance")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import calculate_spl_at_distance
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["source_spl", "source_distance", "target_distance", "environment"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_spl_at_distance(**func_kwargs)
            
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


@Toolbox.register(name="plot_spl_combination_process")
class PlotSplCombinationProcessTool(EnvironmentTool):
    """可视化声压级逐步组合过程。"""
    
    name = "plot_spl_combination_process"
    description = "可视化声压级逐步组合过程。"
    arguments = {
        "spl_list": {"type": "array", "description": "声压级列表（单位：dB）"},
        "use_table": {"type": "boolean", "description": "如果为True，使用查找表；如果为False，使用精确公式"},
        "save_path": {"type": "string", "description": "可选的自定义保存路径"},
        "precomputed_steps": {"type": "array", "description": "预计算的步骤列表；每个步骤为字典对象，包含step和result等字段"},
        "sorted_spl": {"type": "array", "description": "已排序的声压级列表"},
        "total_spl": {"type": "number", "description": "总声压级值"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_spl_combination_process 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spl_list" not in args or args["spl_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spl_list")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import plot_spl_combination_process
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spl_list", "use_table", "save_path", "precomputed_steps", "sorted_spl", "total_spl"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_spl_combination_process(**func_kwargs)
            
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


@Toolbox.register(name="plot_spl_distance_attenuation")
class PlotSplDistanceAttenuationTool(EnvironmentTool):
    """绘制自由场和扩散场中声压级随距离衰减的曲线图。"""
    
    name = "plot_spl_distance_attenuation"
    description = "绘制自由场和扩散场中声压级随距离衰减的曲线图。"
    arguments = {
        "source_spl": {"type": "number", "description": "声源距离处的声压级，单位：dB"},
        "source_distance": {"type": "number", "description": "测量source_spl的距离，单位：米"},
        "max_distance": {"type": "number", "description": "绘图的最大距离，单位：米，默认为100.0"},
        "save_path": {"type": "string", "description": "可选的自定义保存路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_spl_distance_attenuation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "source_spl" not in args or args["source_spl"] is None:
                return Observation(self.name, "错误: 缺少必需参数 source_spl")
            if "source_distance" not in args or args["source_distance"] is None:
                return Observation(self.name, "错误: 缺少必需参数 source_distance")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import plot_spl_distance_attenuation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["source_spl", "source_distance", "max_distance", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_spl_distance_attenuation(**func_kwargs)
            
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


@Toolbox.register(name="plot_lookup_table_visualization")
class PlotLookupTableVisualizationTool(EnvironmentTool):
    """可视化SPL查找表及其插值曲线。"""
    
    name = "plot_lookup_table_visualization"
    description = "可视化SPL查找表及其插值曲线。"
    arguments = {
        "save_path": {"type": "string", "description": "可选的自定义保存路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_lookup_table_visualization 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.acoustics_toolkit_claude_653 import plot_lookup_table_visualization
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_lookup_table_visualization(**func_kwargs)
            
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


@Toolbox.register(name="calculate_doppler_shift")
class CalculateDopplerShiftTool(EnvironmentTool):
    """计算多普勒频移"""
    
    name = "calculate_doppler_shift"
    description = "计算多普勒频移"
    arguments = {
        "freq_emit": {"type": "number", "description": "发射频率，单位为Hz"},
        "freq_received": {"type": "number", "description": "接收频率，单位为Hz"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_doppler_shift 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "freq_emit" not in args or args["freq_emit"] is None:
                return Observation(self.name, "错误: 缺少必需参数 freq_emit")
            if "freq_received" not in args or args["freq_received"] is None:
                return Observation(self.name, "错误: 缺少必需参数 freq_received")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import calculate_doppler_shift
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["freq_emit", "freq_received"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_doppler_shift(**func_kwargs)
            
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


@Toolbox.register(name="calculate_blood_velocity")
class CalculateBloodVelocityTool(EnvironmentTool):
    """计算血流速度（doppler_blood_velocity的别名）。"""
    
    name = "calculate_blood_velocity"
    description = "计算血流速度（doppler_blood_velocity的别名）。"
    arguments = {
        "freq_emit": {"type": "number", "description": "发射频率，单位为Hz"},
        "freq_received": {"type": "number", "description": "接收频率，单位为Hz"},
        "v_sound": {"type": "number", "description": "声波在组织中的传播速度，默认值为1540 m/s"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_blood_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "freq_emit" not in args or args["freq_emit"] is None:
                return Observation(self.name, "错误: 缺少必需参数 freq_emit")
            if "freq_received" not in args or args["freq_received"] is None:
                return Observation(self.name, "错误: 缺少必需参数 freq_received")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import calculate_blood_velocity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["freq_emit", "freq_received", "v_sound"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_blood_velocity(**func_kwargs)
            
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


@Toolbox.register(name="analyze_doppler_spectrum")
class AnalyzeDopplerSpectrumTool(EnvironmentTool):
    """分析多普勒频谱"""
    
    name = "analyze_doppler_spectrum"
    description = "分析多普勒频谱"
    arguments = {
        "frequencies": {"type": "array", "description": "频率数组"},
        "amplitudes": {"type": "array", "description": "幅度数组"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_doppler_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "frequencies" not in args or args["frequencies"] is None:
                return Observation(self.name, "错误: 缺少必需参数 frequencies")
            if "amplitudes" not in args or args["amplitudes"] is None:
                return Observation(self.name, "错误: 缺少必需参数 amplitudes")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import analyze_doppler_spectrum
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["frequencies", "amplitudes"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_doppler_spectrum(**func_kwargs)
            
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


@Toolbox.register(name="doppler_blood_velocity")
class DopplerBloodVelocityTool(EnvironmentTool):
    """计算多普勒血流速度。基于多普勒效应原理，通过发射频率和接收频率的差值计算血流速度。适用于医学超声多普勒血流检测。"""
    
    name = "doppler_blood_velocity"
    description = "计算多普勒血流速度。基于多普勒效应原理，通过发射频率和接收频率的差值计算血流速度。适用于医学超声多普勒血流检测。"
    arguments = {
        "freq_emit": {"type": "number", "description": "发射频率，单位为Hz。超声探头发射的原始频率"},
        "freq_received": {"type": "number", "description": "接收频率，单位为Hz。从流动血液反射回来的频率"},
        "v_sound": {"type": "number", "description": "声波在组织中的传播速度，默认值为1540 m/s（人体软组织的典型声速）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 doppler_blood_velocity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "freq_emit" not in args or args["freq_emit"] is None:
                return Observation(self.name, "错误: 缺少必需参数 freq_emit")
            if "freq_received" not in args or args["freq_received"] is None:
                return Observation(self.name, "错误: 缺少必需参数 freq_received")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import doppler_blood_velocity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["freq_emit", "freq_received", "v_sound"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = doppler_blood_velocity(**func_kwargs)
            
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


@Toolbox.register(name="thin_film_interference_thickness")
class ThinFilmInterferenceThicknessTool(EnvironmentTool):
    """计算薄膜干涉条件下的薄膜厚度。基于薄膜干涉理论，计算在给定波长、折射率、干涉级次和入射角条件下，产生相长干涉或相消干涉所需的薄膜厚度。"""
    
    name = "thin_film_interference_thickness"
    description = "计算薄膜干涉条件下的薄膜厚度。基于薄膜干涉理论，计算在给定波长、折射率、干涉级次和入射角条件下，产生相长干涉或相消干涉所需的薄膜厚度。"
    arguments = {
        "wavelength": {"type": "number", "description": "入射光波长，单位为nm或m。用于干涉计算的光波波长"},
        "n": {"type": "number", "description": "薄膜材料的折射率。薄膜相对于周围介质的折射率比值"},
        "m": {"type": "integer", "description": "干涉级次，默认为1。表示第m级干涉（m=1为一级干涉，m=2为二级干涉等）"},
        "theta": {"type": "number", "description": "入射角，单位为弧度，默认为0（垂直入射）。光线与薄膜法线的夹角"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 thin_film_interference_thickness 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wavelength" not in args or args["wavelength"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wavelength")
            if "n" not in args or args["n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n")
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import thin_film_interference_thickness
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wavelength", "n", "m", "theta"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = thin_film_interference_thickness(**func_kwargs)
            
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


@Toolbox.register(name="maxwell_field_relation")
class MaxwellFieldRelationTool(EnvironmentTool):
    """基于麦克斯韦方程计算电磁场关系"""
    
    name = "maxwell_field_relation"
    description = "基于麦克斯韦方程计算电磁场关系"
    arguments = {
        "changing_E": {"type": "number", "description": "changing_E参数，默认值为None"},
        "changing_B": {"type": "number", "description": "changing_B参数，默认值为None"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 maxwell_field_relation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import maxwell_field_relation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["changing_E", "changing_B"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = maxwell_field_relation(**func_kwargs)
            
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


@Toolbox.register(name="uv_remote_sensing_capability")
class UvRemoteSensingCapabilityTool(EnvironmentTool):
    """计算uv_remote_sensing_capability相关结果"""
    
    name = "uv_remote_sensing_capability"
    description = "计算uv_remote_sensing_capability相关结果"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 uv_remote_sensing_capability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import uv_remote_sensing_capability
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = uv_remote_sensing_capability(**func_kwargs)
            
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


@Toolbox.register(name="visualize_doppler_ultrasound")
class VisualizeDopplerUltrasoundTool(EnvironmentTool):
    """基于多普勒效应计算血流速度"""
    
    name = "visualize_doppler_ultrasound"
    description = "基于多普勒效应计算血流速度"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_doppler_ultrasound 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import visualize_doppler_ultrasound
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_doppler_ultrasound(**func_kwargs)
            
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


@Toolbox.register(name="visualize_thin_film_interference")
class VisualizeThinFilmInterferenceTool(EnvironmentTool):
    """计算薄膜干涉相关参数"""
    
    name = "visualize_thin_film_interference"
    description = "计算薄膜干涉相关参数"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_thin_film_interference 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import visualize_thin_film_interference
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_thin_film_interference(**func_kwargs)
            
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


@Toolbox.register(name="visualize_maxwell_fields")
class VisualizeMaxwellFieldsTool(EnvironmentTool):
    """基于麦克斯韦方程计算电磁场关系"""
    
    name = "visualize_maxwell_fields"
    description = "基于麦克斯韦方程计算电磁场关系"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_maxwell_fields 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import visualize_maxwell_fields
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_maxwell_fields(**func_kwargs)
            
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


@Toolbox.register(name="visualize_uv_remote_sensing")
class VisualizeUvRemoteSensingTool(EnvironmentTool):
    """生成可视化图表"""
    
    name = "visualize_uv_remote_sensing"
    description = "生成可视化图表"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_uv_remote_sensing 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import visualize_uv_remote_sensing
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_uv_remote_sensing(**func_kwargs)
            
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


@Toolbox.register(name="analyze_physics_statements")
class AnalyzePhysicsStatementsTool(EnvironmentTool):
    """执行分析计算"""
    
    name = "analyze_physics_statements"
    description = "执行分析计算"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_physics_statements 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.acoustics.doppler_blood_velocity import analyze_physics_statements
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_physics_statements(**func_kwargs)
            
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

