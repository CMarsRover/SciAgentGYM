#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
atomic_and_molecular_physics 工具注册模块
使用 gym.tool.EnvironmentTool 为 atomic_and_molecular_physics 目录中的工具提供统一的注册与调用接口

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


@Toolbox.register(name="pauli_matrices")
class PauliMatricesTool(EnvironmentTool):
    """返回Pauli矩阵（I, X, Y, Z）"""
    
    name = "pauli_matrices"
    description = "返回Pauli矩阵（I, X, Y, Z）"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 pauli_matrices 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import pauli_matrices
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = pauli_matrices(**func_kwargs)
            
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


@Toolbox.register(name="standard_cnot_gate")
class StandardCnotGateTool(EnvironmentTool):
    """构造标准CNOT门（控制比特为|1⟩时翻转目标比特）"""
    
    name = "standard_cnot_gate"
    description = "构造标准CNOT门（控制比特为|1⟩时翻转目标比特）"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 standard_cnot_gate 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import standard_cnot_gate
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = standard_cnot_gate(**func_kwargs)
            
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


@Toolbox.register(name="anti_controlled_not_gate")
class AntiControlledNotGateTool(EnvironmentTool):
    """构造反控制NOT门（控制比特为|0⟩时翻转目标比特）"""
    
    name = "anti_controlled_not_gate"
    description = "构造反控制NOT门（控制比特为|0⟩时翻转目标比特）"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 anti_controlled_not_gate 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import anti_controlled_not_gate
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = anti_controlled_not_gate(**func_kwargs)
            
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


@Toolbox.register(name="custom_controlled_gate")
class CustomControlledGateTool(EnvironmentTool):
    """构造自定义控制门"""
    
    name = "custom_controlled_gate"
    description = "构造自定义控制门"
    arguments = {
        "control_state": {"type": "integer", "description": "控制态 (0 或 1)"},
        "target_operation": {"type": "array", "description": "目标操作的矩阵表示"},
        "n_qubits": {"type": "integer", "description": "量子比特数（默认2）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 custom_controlled_gate 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "control_state" not in args or args["control_state"] is None:
                return Observation(self.name, "错误: 缺少必需参数 control_state")
            if "target_operation" not in args or args["target_operation"] is None:
                return Observation(self.name, "错误: 缺少必需参数 target_operation")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import custom_controlled_gate
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["control_state", "target_operation", "n_qubits"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = custom_controlled_gate(**func_kwargs)
            
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


@Toolbox.register(name="verify_gate_properties")
class VerifyGatePropertiesTool(EnvironmentTool):
    """验证量子门的性质（幺正性、厄米性等）"""
    
    name = "verify_gate_properties"
    description = "验证量子门的性质（幺正性、厄米性等）"
    arguments = {
        "gate_matrix": {"type": "array", "description": "量子门矩阵"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 verify_gate_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gate_matrix" not in args or args["gate_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gate_matrix")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import verify_gate_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gate_matrix"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = verify_gate_properties(**func_kwargs)
            
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


@Toolbox.register(name="visualize_gate_matrix")
class VisualizeGateMatrixTool(EnvironmentTool):
    """可视化量子门矩阵"""
    
    name = "visualize_gate_matrix"
    description = "可视化量子门矩阵"
    arguments = {
        "gate_matrix": {"type": "array", "description": "量子门矩阵"},
        "gate_name": {"type": "string", "description": "门的名称"},
        "basis_labels": {"type": "array", "description": "基态标签列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_gate_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gate_matrix" not in args or args["gate_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gate_matrix")
            if "gate_name" not in args or args["gate_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gate_name")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import visualize_gate_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gate_matrix", "gate_name", "basis_labels"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_gate_matrix(**func_kwargs)
            
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


@Toolbox.register(name="visualize_quantum_circuit")
class VisualizeQuantumCircuitTool(EnvironmentTool):
    """可视化量子电路"""
    
    name = "visualize_quantum_circuit"
    description = "可视化量子电路"
    arguments = {
        "gates": {"type": "array", "description": "门操作列表，每个元素为 {'type': str, 'qubits': list, 'name': str}"},
        "n_qubits": {"type": "integer", "description": "量子比特数量"},
        "circuit_name": {"type": "string", "description": "电路名称"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_quantum_circuit 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gates" not in args or args["gates"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gates")
            if "n_qubits" not in args or args["n_qubits"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_qubits")
            if "circuit_name" not in args or args["circuit_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 circuit_name")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import visualize_quantum_circuit
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gates", "n_qubits", "circuit_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_quantum_circuit(**func_kwargs)
            
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


@Toolbox.register(name="generate_latex_report")
class GenerateLatexReportTool(EnvironmentTool):
    """生成量子门的LaTeX报告"""
    
    name = "generate_latex_report"
    description = "生成量子门的LaTeX报告"
    arguments = {
        "gate_info": {"type": "object", "description": "包含门信息的字典"},
        "output_name": {"type": "string", "description": "输出文件名"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 generate_latex_report 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gate_info" not in args or args["gate_info"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gate_info")
            if "output_name" not in args or args["output_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 output_name")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_gates_toolkit_claude_159 import generate_latex_report
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gate_info", "output_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = generate_latex_report(**func_kwargs)
            
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


@Toolbox.register(name="calculate_normalization_constant")
class CalculateNormalizationConstantTool(EnvironmentTool):
    """计算归一化常数 A，使得 ∫|ψ|² * 4πr² dr = 1。"""
    
    name = "calculate_normalization_constant"
    description = "计算归一化常数 A，使得 ∫|ψ|² * 4πr² dr = 1。"
    arguments = {
        "r_min": {"type": "number", "description": "内球半径"},
        "r_max": {"type": "number", "description": "外球半径"},
        "wave_function_type": {"type": "string", "description": "波函数类型"},
        "params": {"type": "object", "description": "波函数参数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_normalization_constant 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "r_min" not in args or args["r_min"] is None:
                return Observation(self.name, "错误: 缺少必需参数 r_min")
            if "r_max" not in args or args["r_max"] is None:
                return Observation(self.name, "错误: 缺少必需参数 r_max")
            if "wave_function_type" not in args or args["wave_function_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_type")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import calculate_normalization_constant
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["r_min", "r_max", "wave_function_type", "params"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_normalization_constant(**func_kwargs)
            
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


@Toolbox.register(name="parse_superposition_state")
class ParseSuperpositionStateTool(EnvironmentTool):
    """解析并验证叠加态。检查归一化并返回态信息。"""
    
    name = "parse_superposition_state"
    description = "解析并验证叠加态。检查归一化并返回态信息。"
    arguments = {
        "coefficients": {"type": "array", "description": "态系数列表（实数或复数的实部）"},
        "states": {"type": "array", "description": "量子数列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_superposition_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            if "states" not in args or args["states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 states")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_harmonic_oscillator_toolkit_claude_265 import parse_superposition_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients", "states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_superposition_state(**func_kwargs)
            
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


@Toolbox.register(name="calculate_operator_aa_dagger_expectation")
class CalculateOperatorAaDaggerExpectationTool(EnvironmentTool):
    """计算aa†算符的期望值。aa†|n> = (N + 1)|n>，其中N是粒子数算符。因此<psi|aa†|psi> = <N> + 1"""
    
    name = "calculate_operator_aa_dagger_expectation"
    description = "计算aa†算符的期望值。aa†|n> = (N + 1)|n>，其中N是粒子数算符。因此<psi|aa†|psi> = <N> + 1"
    arguments = {
        "coefficients": {"type": "array", "description": "态系数列表"},
        "states": {"type": "array", "description": "量子数列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_operator_aa_dagger_expectation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            if "states" not in args or args["states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 states")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_harmonic_oscillator_toolkit_claude_265 import calculate_operator_aa_dagger_expectation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients", "states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_operator_aa_dagger_expectation(**func_kwargs)
            
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


@Toolbox.register(name="calculate_hamiltonian_expectation")
class CalculateHamiltonianExpectationTool(EnvironmentTool):
    """计算哈密顿量H = (aa† + 1/2)ℏω的期望值。由于aa† = N + 1，因此<H> = (<N> + 3/2)ℏω"""
    
    name = "calculate_hamiltonian_expectation"
    description = "计算哈密顿量H = (aa† + 1/2)ℏω的期望值。由于aa† = N + 1，因此<H> = (<N> + 3/2)ℏω"
    arguments = {
        "coefficients": {"type": "array", "description": "态系数列表"},
        "states": {"type": "array", "description": "量子数列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_hamiltonian_expectation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            if "states" not in args or args["states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 states")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_harmonic_oscillator_toolkit_claude_265 import calculate_hamiltonian_expectation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients", "states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_hamiltonian_expectation(**func_kwargs)
            
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


@Toolbox.register(name="plot_superposition_state")
class PlotSuperpositionStateTool(EnvironmentTool):
    """绘制叠加态的波函数和概率密度图像。"""
    
    name = "plot_superposition_state"
    description = "绘制叠加态的波函数和概率密度图像。"
    arguments = {
        "coefficients": {"type": "array", "description": "态系数列表"},
        "states": {"type": "array", "description": "量子数列表"},
        "x_range": {"type": "array", "description": "x坐标范围，(x_min, x_max)"},
        "num_points": {"type": "integer", "description": "绘图点数，默认为500"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_superposition_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            if "states" not in args or args["states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 states")
            if "x_range" not in args or args["x_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 x_range")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_harmonic_oscillator_toolkit_claude_265 import plot_superposition_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients", "states", "x_range", "num_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_superposition_state(**func_kwargs)
            
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


@Toolbox.register(name="plot_expectation_values")
class PlotExpectationValuesTool(EnvironmentTool):
    """绘制叠加态下各种算符的期望值图像。"""
    
    name = "plot_expectation_values"
    description = "绘制叠加态下各种算符的期望值图像。"
    arguments = {
        "coefficients": {"type": "array", "description": "态系数列表"},
        "states": {"type": "array", "description": "量子数列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_expectation_values 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            if "states" not in args or args["states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 states")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_harmonic_oscillator_toolkit_claude_265 import plot_expectation_values
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients", "states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_expectation_values(**func_kwargs)
            
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


@Toolbox.register(name="setup_matplotlib_chinese_fonts")
class SetupMatplotlibChineseFontsTool(EnvironmentTool):
    """配置中文字体（如缺失将回退英文），避免绘图中文乱码。"""
    
    name = "setup_matplotlib_chinese_fonts"
    description = "配置中文字体（如缺失将回退英文），避免绘图中文乱码。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 setup_matplotlib_chinese_fonts 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import setup_matplotlib_chinese_fonts
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = setup_matplotlib_chinese_fonts(**func_kwargs)
            
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


@Toolbox.register(name="get_default_molecule")
class GetDefaultMoleculeTool(EnvironmentTool):
    """返回分子与基组的默认设置（示例为 HF 分子）。"""
    
    name = "get_default_molecule"
    description = "返回分子与基组的默认设置（示例为 HF 分子）。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_default_molecule 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import get_default_molecule
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_default_molecule(**func_kwargs)
            
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


@Toolbox.register(name="get_fd_default_grid")
class GetFdDefaultGridTool(EnvironmentTool):
    """有限差分动能矩阵的默认网格参数。"""
    
    name = "get_fd_default_grid"
    description = "有限差分动能矩阵的默认网格参数。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_fd_default_grid 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import get_fd_default_grid
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_fd_default_grid(**func_kwargs)
            
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


@Toolbox.register(name="get_planewave_defaults")
class GetPlanewaveDefaultsTool(EnvironmentTool):
    """平面波动能曲线的默认参数。"""
    
    name = "get_planewave_defaults"
    description = "平面波动能曲线的默认参数。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_planewave_defaults 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import get_planewave_defaults
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_planewave_defaults(**func_kwargs)
            
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


@Toolbox.register(name="coding_func_molecular_kinetic")
class CodingFuncMolecularKineticTool(EnvironmentTool):
    """返回分子动能积分矩阵与分子对象。"""
    
    name = "coding_func_molecular_kinetic"
    description = "返回分子动能积分矩阵与分子对象。"
    arguments = {
        "atom": {"type": "string", "description": "分子原子坐标描述，格式为'元素 x y z; 元素 x y z'，默认为'H 0 0 0; F 0 0 1.1'"},
        "basis": {"type": "string", "description": "基组名称，默认为'sto-3g'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 coding_func_molecular_kinetic 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import coding_func_molecular_kinetic
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["atom", "basis"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = coding_func_molecular_kinetic(**func_kwargs)
            
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


@Toolbox.register(name="visual_func_molecular_kinetic")
class VisualFuncMolecularKineticTool(EnvironmentTool):
    """可视化分子动能积分矩阵的热力图。"""
    
    name = "visual_func_molecular_kinetic"
    description = "可视化分子动能积分矩阵的热力图。"
    arguments = {
        "kinetic_matrix": {"type": "array", "description": "动能积分矩阵，二维数组"},
        "title": {"type": "string", "description": "图表标题，默认为'Kinetic Energy Integral Matrix'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visual_func_molecular_kinetic 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "kinetic_matrix" not in args or args["kinetic_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kinetic_matrix")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import visual_func_molecular_kinetic
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["kinetic_matrix", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visual_func_molecular_kinetic(**func_kwargs)
            
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


@Toolbox.register(name="math_func_gaussian_kinetic_integral")
class MathFuncGaussianKineticIntegralTool(EnvironmentTool):
    """1D高斯波函数的动能积分解析结果。"""
    
    name = "math_func_gaussian_kinetic_integral"
    description = "1D高斯波函数的动能积分解析结果。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 math_func_gaussian_kinetic_integral 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import math_func_gaussian_kinetic_integral
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = math_func_gaussian_kinetic_integral(**func_kwargs)
            
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


@Toolbox.register(name="coding_func_finite_difference_kinetic")
class CodingFuncFiniteDifferenceKineticTool(EnvironmentTool):
    """构造一维二阶导数算子的有限差分矩阵，对应动能算子 -1/2 ∂²/∂x²。"""
    
    name = "coding_func_finite_difference_kinetic"
    description = "构造一维二阶导数算子的有限差分矩阵，对应动能算子 -1/2 ∂²/∂x²。"
    arguments = {
        "N": {"type": "integer", "description": "网格点数，默认为200"},
        "L": {"type": "number", "description": "空间范围长度，默认为10.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 coding_func_finite_difference_kinetic 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import coding_func_finite_difference_kinetic
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["N", "L"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = coding_func_finite_difference_kinetic(**func_kwargs)
            
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


@Toolbox.register(name="visual_func_infinite_well_eigenstates")
class VisualFuncInfiniteWellEigenstatesTool(EnvironmentTool):
    """可视化无限深势阱的本征态概率密度分布"""
    
    name = "visual_func_infinite_well_eigenstates"
    description = "可视化无限深势阱的本征态概率密度分布"
    arguments = {
        "T": {"type": "array", "description": "动能算符矩阵，二维数组"},
        "x": {"type": "array", "description": "空间坐标点数组"},
        "num_states": {"type": "integer", "description": "要显示的本征态数量，默认为5"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visual_func_infinite_well_eigenstates 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "T" not in args or args["T"] is None:
                return Observation(self.name, "错误: 缺少必需参数 T")
            if "x" not in args or args["x"] is None:
                return Observation(self.name, "错误: 缺少必需参数 x")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import visual_func_infinite_well_eigenstates
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["T", "x", "num_states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visual_func_infinite_well_eigenstates(**func_kwargs)
            
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


@Toolbox.register(name="math_func_infinite_well_analytical")
class MathFuncInfiniteWellAnalyticalTool(EnvironmentTool):
    """无限深势阱动能本征值公式：E_n = n²π²/(2L²)。"""
    
    name = "math_func_infinite_well_analytical"
    description = "无限深势阱动能本征值公式：E_n = n²π²/(2L²)。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 math_func_infinite_well_analytical 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import math_func_infinite_well_analytical
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = math_func_infinite_well_analytical(**func_kwargs)
            
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


@Toolbox.register(name="coding_func_planewave_kinetic")
class CodingFuncPlanewaveKineticTool(EnvironmentTool):
    """一维平面波自由电子模型：E(k,G)=0.5*(k+G*b)^2。"""
    
    name = "coding_func_planewave_kinetic"
    description = "一维平面波自由电子模型：E(k,G)=0.5*(k+G*b)^2。"
    arguments = {
        "k_points": {"type": "integer", "description": "k点数量，默认为100"},
        "G_max": {"type": "integer", "description": "倒格子矢量G的最大值，默认为5"},
        "a": {"type": "number", "description": "晶格常数，默认为5.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 coding_func_planewave_kinetic 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import coding_func_planewave_kinetic
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["k_points", "G_max", "a"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = coding_func_planewave_kinetic(**func_kwargs)
            
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


@Toolbox.register(name="visual_func_band_structure")
class VisualFuncBandStructureTool(EnvironmentTool):
    """可视化一维自由电子能带结构。"""
    
    name = "visual_func_band_structure"
    description = "可视化一维自由电子能带结构。"
    arguments = {
        "k_list": {"type": "array", "description": "波矢k的列表或数组"},
        "E_bands": {"type": "array", "description": "能带能量数组，形状为(k点数, 能带数)，每列代表一个能带"},
        "title": {"type": "string", "description": "图表标题，默认为'1D Free-electron Band Structure'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visual_func_band_structure 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "k_list" not in args or args["k_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 k_list")
            if "E_bands" not in args or args["E_bands"] is None:
                return Observation(self.name, "错误: 缺少必需参数 E_bands")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import visual_func_band_structure
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["k_list", "E_bands", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visual_func_band_structure(**func_kwargs)
            
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


@Toolbox.register(name="math_func_planewave_kinetic_expression")
class MathFuncPlanewaveKineticExpressionTool(EnvironmentTool):
    """计算并打印平面波动能在倒格子空间的数学表达式。"""
    
    name = "math_func_planewave_kinetic_expression"
    description = "计算并打印平面波动能在倒格子空间的数学表达式。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 math_func_planewave_kinetic_expression 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.coding_func_molecular_kinetic import math_func_planewave_kinetic_expression
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = math_func_planewave_kinetic_expression(**func_kwargs)
            
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


@Toolbox.register(name="normalize_state_vector")
class NormalizeStateVectorTool(EnvironmentTool):
    """归一化量子态矢量"""
    
    name = "normalize_state_vector"
    description = "归一化量子态矢量"
    arguments = {
        "coefficients": {"type": "array", "description": "复数系数列表，如 [(1+1j), (2-1j)]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 normalize_state_vector 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_228 import normalize_state_vector
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = normalize_state_vector(**func_kwargs)
            
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


@Toolbox.register(name="construct_operator_matrix")
class ConstructOperatorMatrixTool(EnvironmentTool):
    """根据矩阵元素规则构造算符矩阵"""
    
    name = "construct_operator_matrix"
    description = "根据矩阵元素规则构造算符矩阵"
    arguments = {
        "matrix_elements": {"type": "object", "description": "矩阵元素规则字典，包含 diagonal（对角元素值）和 off_diagonal（非对角元素值）字段"},
        "dimension": {"type": "integer", "description": "矩阵维度"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_operator_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_elements" not in args or args["matrix_elements"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_elements")
            if "dimension" not in args or args["dimension"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dimension")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_228 import construct_operator_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_elements", "dimension"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_operator_matrix(**func_kwargs)
            
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


@Toolbox.register(name="solve_eigenvalue_problem")
class SolveEigenvalueProblemTool(EnvironmentTool):
    """求解算符矩阵的本征值和本征态"""
    
    name = "solve_eigenvalue_problem"
    description = "求解算符矩阵的本征值和本征态"
    arguments = {
        "matrix": {"type": "array", "description": "算符矩阵（列表形式）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_eigenvalue_problem 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix" not in args or args["matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_228 import solve_eigenvalue_problem
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve_eigenvalue_problem(**func_kwargs)
            
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


@Toolbox.register(name="compute_inner_product")
class ComputeInnerProductTool(EnvironmentTool):
    """计算两个复数向量的内积 <v1|v2>"""
    
    name = "compute_inner_product"
    description = "计算两个复数向量的内积 <v1|v2>"
    arguments = {
        "vector1": {"type": "array", "description": "第一个复数向量"},
        "vector2": {"type": "array", "description": "第二个复数向量"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_inner_product 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "vector1" not in args or args["vector1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vector1")
            if "vector2" not in args or args["vector2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vector2")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_toolkit_claude_139 import compute_inner_product
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["vector1", "vector2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_inner_product(**func_kwargs)
            
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


@Toolbox.register(name="calculate_measurement_probabilities")
class CalculateMeasurementProbabilitiesTool(EnvironmentTool):
    """计算在给定本征态基底下的测量概率"""
    
    name = "calculate_measurement_probabilities"
    description = "计算在给定本征态基底下的测量概率"
    arguments = {
        "initial_state": {"type": "array", "description": "初始量子态（归一化）"},
        "eigenstates": {"type": "array", "description": "本征态列表（每个本征态是一个列表）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_measurement_probabilities 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "initial_state" not in args or args["initial_state"] is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_state")
            if "eigenstates" not in args or args["eigenstates"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenstates")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_228 import calculate_measurement_probabilities
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["initial_state", "eigenstates"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_measurement_probabilities(**func_kwargs)
            
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


@Toolbox.register(name="project_state_to_eigenbasis")
class ProjectStateToEigenbasisTool(EnvironmentTool):
    """将量子态投影到本征态基底，并计算各分量"""
    
    name = "project_state_to_eigenbasis"
    description = "将量子态投影到本征态基底，并计算各分量"
    arguments = {
        "state": {"type": "array", "description": "初始量子态"},
        "eigenstates": {"type": "array", "description": "本征态列表"},
        "eigenvalues": {"type": "array", "description": "对应的本征值列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 project_state_to_eigenbasis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state" not in args or args["state"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state")
            if "eigenstates" not in args or args["eigenstates"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenstates")
            if "eigenvalues" not in args or args["eigenvalues"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenvalues")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_228 import project_state_to_eigenbasis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state", "eigenstates", "eigenvalues"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = project_state_to_eigenbasis(**func_kwargs)
            
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


@Toolbox.register(name="visualize_probability_distribution")
class VisualizeProbabilityDistributionTool(EnvironmentTool):
    """可视化测量概率分布"""
    
    name = "visualize_probability_distribution"
    description = "可视化测量概率分布"
    arguments = {
        "probabilities": {"type": "array", "description": "概率列表"},
        "eigenvalues": {"type": "array", "description": "对应的本征值"},
        "title": {"type": "string", "description": "图表标题，默认为 Measurement Probability Distribution"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_probability_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "probabilities" not in args or args["probabilities"] is None:
                return Observation(self.name, "错误: 缺少必需参数 probabilities")
            if "eigenvalues" not in args or args["eigenvalues"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenvalues")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_228 import visualize_probability_distribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["probabilities", "eigenvalues", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_probability_distribution(**func_kwargs)
            
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


@Toolbox.register(name="visualize_quantum_state")
class VisualizeQuantumStateTool(EnvironmentTool):
    """可视化量子态（幅值和相位）"""
    
    name = "visualize_quantum_state"
    description = "可视化量子态（幅值和相位）"
    arguments = {
        "state_components": {"type": "array", "description": "量子态分量"},
        "basis_labels": {"type": "array", "description": "基矢标签列表，可选"},
        "title": {"type": "string", "description": "图表标题，默认为'Quantum State'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_quantum_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_components" not in args or args["state_components"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_components")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import visualize_quantum_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_components", "basis_labels", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_quantum_state(**func_kwargs)
            
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


@Toolbox.register(name="construct_hamiltonian")
class ConstructHamiltonianTool(EnvironmentTool):
    """构建自旋在磁场中的哈密顿量"""
    
    name = "construct_hamiltonian"
    description = "构建自旋在磁场中的哈密顿量"
    arguments = {
        "field_direction": {"type": "string", "description": "磁场方向 ('x', 'y', 'z')"},
        "field_strength": {"type": "number", "description": "磁场强度 B"},
        "gamma": {"type": "number", "description": "旋磁比"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_hamiltonian 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "field_direction" not in args or args["field_direction"] is None:
                return Observation(self.name, "错误: 缺少必需参数 field_direction")
            if "field_strength" not in args or args["field_strength"] is None:
                return Observation(self.name, "错误: 缺少必需参数 field_strength")
            if "gamma" not in args or args["gamma"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import construct_hamiltonian
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["field_direction", "field_strength", "gamma"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_hamiltonian(**func_kwargs)
            
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


@Toolbox.register(name="evolve_spin_state")
class EvolveSpinStateTool(EnvironmentTool):
    """在时间依赖哈密顿量下演化自旋态。"""
    
    name = "evolve_spin_state"
    description = "在时间依赖哈密顿量下演化自旋态。"
    arguments = {
        "psi_initial": {"type": "array", "description": "初始态 [c_up, c_down]"},
        "t_span": {"type": "array", "description": "时间范围 [t_start, t_end]（秒）"},
        "B0": {"type": "number", "description": "磁场强度（特斯拉）"},
        "theta": {"type": "number", "description": "倾斜角（弧度）"},
        "omega_rot": {"type": "number", "description": "旋转频率（弧度/秒）"},
        "n_points": {"type": "integer", "description": "时间点数量，默认为1000"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 evolve_spin_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "psi_initial" not in args or args["psi_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_initial")
            if "t_span" not in args or args["t_span"] is None:
                return Observation(self.name, "错误: 缺少必需参数 t_span")
            if "B0" not in args or args["B0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 B0")
            if "theta" not in args or args["theta"] is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            if "omega_rot" not in args or args["omega_rot"] is None:
                return Observation(self.name, "错误: 缺少必需参数 omega_rot")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_234 import evolve_spin_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["psi_initial", "t_span", "B0", "theta", "omega_rot", "n_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = evolve_spin_state(**func_kwargs)
            
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


@Toolbox.register(name="rotating_frame_transformation")
class RotatingFrameTransformationTool(EnvironmentTool):
    """分析旋转参考系中的自旋动力学（旋转波近似）。"""
    
    name = "rotating_frame_transformation"
    description = "分析旋转参考系中的自旋动力学（旋转波近似）。"
    arguments = {
        "B0": {"type": "number", "description": "磁场强度（特斯拉）"},
        "theta": {"type": "number", "description": "倾斜角（弧度）"},
        "omega_rot": {"type": "number", "description": "旋转频率（弧度/秒）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 rotating_frame_transformation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "B0" not in args or args["B0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 B0")
            if "theta" not in args or args["theta"] is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            if "omega_rot" not in args or args["omega_rot"] is None:
                return Observation(self.name, "错误: 缺少必需参数 omega_rot")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_234 import rotating_frame_transformation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["B0", "theta", "omega_rot"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = rotating_frame_transformation(**func_kwargs)
            
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


@Toolbox.register(name="adiabatic_spin_down_probability")
class AdiabaticSpinDownProbabilityTool(EnvironmentTool):
    """使用微扰理论计算绝热极限下的自旋向下概率。"""
    
    name = "adiabatic_spin_down_probability"
    description = "使用微扰理论计算绝热极限下的自旋向下概率。"
    arguments = {
        "B0": {"type": "number", "description": "磁场强度（特斯拉）"},
        "theta": {"type": "number", "description": "倾斜角（弧度）"},
        "omega_rot": {"type": "number", "description": "旋转频率（弧度/秒）"},
        "time_duration": {"type": "number", "description": "演化时间（秒）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 adiabatic_spin_down_probability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "B0" not in args or args["B0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 B0")
            if "theta" not in args or args["theta"] is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            if "omega_rot" not in args or args["omega_rot"] is None:
                return Observation(self.name, "错误: 缺少必需参数 omega_rot")
            if "time_duration" not in args or args["time_duration"] is None:
                return Observation(self.name, "错误: 缺少必需参数 time_duration")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_234 import adiabatic_spin_down_probability
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["B0", "theta", "omega_rot", "time_duration"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = adiabatic_spin_down_probability(**func_kwargs)
            
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


@Toolbox.register(name="plot_bloch_sphere_trajectory")
class PlotBlochSphereTrajectoryTool(EnvironmentTool):
    """在布洛赫球上可视化自旋态轨迹。"""
    
    name = "plot_bloch_sphere_trajectory"
    description = "在布洛赫球上可视化自旋态轨迹。"
    arguments = {
        "evolution_filepath": {"type": "string", "description": "演化数据JSON文件路径"},
        "save_path": {"type": "string", "description": "输出图像路径，默认自动生成"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_bloch_sphere_trajectory 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "evolution_filepath" not in args or args["evolution_filepath"] is None:
                return Observation(self.name, "错误: 缺少必需参数 evolution_filepath")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_234 import plot_bloch_sphere_trajectory
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["evolution_filepath", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_bloch_sphere_trajectory(**func_kwargs)
            
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


@Toolbox.register(name="plot_probability_evolution")
class PlotProbabilityEvolutionTool(EnvironmentTool):
    """绘制自旋向上和自旋向下概率随时间的演化。"""
    
    name = "plot_probability_evolution"
    description = "绘制自旋向上和自旋向下概率随时间的演化。"
    arguments = {
        "evolution_filepath": {"type": "string", "description": "演化数据JSON文件路径"},
        "save_path": {"type": "string", "description": "输出图像路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_probability_evolution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "evolution_filepath" not in args or args["evolution_filepath"] is None:
                return Observation(self.name, "错误: 缺少必需参数 evolution_filepath")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_234 import plot_probability_evolution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["evolution_filepath", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_probability_evolution(**func_kwargs)
            
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


@Toolbox.register(name="plot_rotating_frame_analysis")
class PlotRotatingFrameAnalysisTool(EnvironmentTool):
    """可视化旋转参考系中的有效场。"""
    
    name = "plot_rotating_frame_analysis"
    description = "可视化旋转参考系中的有效场。"
    arguments = {
        "B0": {"type": "number", "description": "磁场强度（特斯拉）"},
        "theta": {"type": "number", "description": "倾斜角（弧度）"},
        "omega_rot": {"type": "number", "description": "旋转频率（弧度/秒）"},
        "save_path": {"type": "string", "description": "输出图像路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_rotating_frame_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "B0" not in args or args["B0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 B0")
            if "theta" not in args or args["theta"] is None:
                return Observation(self.name, "错误: 缺少必需参数 theta")
            if "omega_rot" not in args or args["omega_rot"] is None:
                return Observation(self.name, "错误: 缺少必需参数 omega_rot")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_234 import plot_rotating_frame_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["B0", "theta", "omega_rot", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_rotating_frame_analysis(**func_kwargs)
            
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


@Toolbox.register(name="get_physical_constants")
class GetPhysicalConstantsTool(EnvironmentTool):
    """从scipy.constants检索基本物理常数。"""
    
    name = "get_physical_constants"
    description = "从scipy.constants检索基本物理常数。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_physical_constants 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import get_physical_constants
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_physical_constants(**func_kwargs)
            
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


@Toolbox.register(name="calculate_electron_g_factor")
class CalculateElectronGFactorTool(EnvironmentTool):
    """计算给定自旋的电子g因子。"""
    
    name = "calculate_electron_g_factor"
    description = "计算给定自旋的电子g因子。"
    arguments = {
        "spin": {"type": "number", "description": "电子自旋量子数（例如0.5、1.5等）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_electron_g_factor 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spin" not in args or args["spin"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spin")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_electron_g_factor
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spin"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_electron_g_factor(**func_kwargs)
            
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


@Toolbox.register(name="calculate_proton_g_factor")
class CalculateProtonGFactorTool(EnvironmentTool):
    """获取质子g因子（氢的核g因子）。"""
    
    name = "calculate_proton_g_factor"
    description = "获取质子g因子（氢的核g因子）。"
    arguments = {

    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_proton_g_factor 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_proton_g_factor
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = []
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_proton_g_factor(**func_kwargs)
            
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


@Toolbox.register(name="calculate_wavefunction_at_nucleus")
class CalculateWavefunctionAtNucleusTool(EnvironmentTool):
    """计算氢原子在原子核处的|ψ(0)|²。"""
    
    name = "calculate_wavefunction_at_nucleus"
    description = "计算氢原子在原子核处的|ψ(0)|²。"
    arguments = {
        "n": {"type": "integer", "description": "主量子数（必须为正整数）"},
        "l": {"type": "integer", "description": "轨道角动量量子数（0 ≤ l < n）"},
        "a_0": {"type": "number", "description": "玻尔半径，单位：米"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_wavefunction_at_nucleus 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n" not in args or args["n"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n")
            if "l" not in args or args["l"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l")
            if "a_0" not in args or args["a_0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 a_0")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_wavefunction_at_nucleus
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n", "l", "a_0"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_wavefunction_at_nucleus(**func_kwargs)
            
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


@Toolbox.register(name="calculate_total_angular_momentum_states")
class CalculateTotalAngularMomentumStatesTool(EnvironmentTool):
    """计算可能的总角动量量子数F。"""
    
    name = "calculate_total_angular_momentum_states"
    description = "计算可能的总角动量量子数F。"
    arguments = {
        "I": {"type": "number", "description": "核自旋量子数"},
        "J": {"type": "number", "description": "电子角动量量子数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_total_angular_momentum_states 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "I" not in args or args["I"] is None:
                return Observation(self.name, "错误: 缺少必需参数 I")
            if "J" not in args or args["J"] is None:
                return Observation(self.name, "错误: 缺少必需参数 J")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_total_angular_momentum_states
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["I", "J"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_total_angular_momentum_states(**func_kwargs)
            
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


@Toolbox.register(name="calculate_hyperfine_constant_A")
class CalculateHyperfineConstantATool(EnvironmentTool):
    """计算氢的超精细结构常数A。"""
    
    name = "calculate_hyperfine_constant_A"
    description = "计算氢的超精细结构常数A。"
    arguments = {
        "g_I": {"type": "number", "description": "核g因子"},
        "g_J": {"type": "number", "description": "电子g因子"},
        "mu_N": {"type": "number", "description": "核磁子，单位：J/T"},
        "mu_B": {"type": "number", "description": "玻尔磁子，单位：J/T"},
        "psi_squared": {"type": "number", "description": "|ψ(0)|²原子核处的电子密度，单位：m^-3"},
        "mu_0": {"type": "number", "description": "真空磁导率，单位：N/A²"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_hyperfine_constant_A 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "g_I" not in args or args["g_I"] is None:
                return Observation(self.name, "错误: 缺少必需参数 g_I")
            if "g_J" not in args or args["g_J"] is None:
                return Observation(self.name, "错误: 缺少必需参数 g_J")
            if "mu_N" not in args or args["mu_N"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_N")
            if "mu_B" not in args or args["mu_B"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_B")
            if "psi_squared" not in args or args["psi_squared"] is None:
                return Observation(self.name, "错误: 缺少必需参数 psi_squared")
            if "mu_0" not in args or args["mu_0"] is None:
                return Observation(self.name, "错误: 缺少必需参数 mu_0")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_hyperfine_constant_A
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["g_I", "g_J", "mu_N", "mu_B", "psi_squared", "mu_0"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_hyperfine_constant_A(**func_kwargs)
            
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


@Toolbox.register(name="calculate_hyperfine_energy_shift")
class CalculateHyperfineEnergyShiftTool(EnvironmentTool):
    """计算给定F态的超精细能量位移。"""
    
    name = "calculate_hyperfine_energy_shift"
    description = "计算给定F态的超精细能量位移。"
    arguments = {
        "A_hz": {"type": "number", "description": "超精细常数，单位：Hz"},
        "F": {"type": "number", "description": "总角动量量子数"},
        "I": {"type": "number", "description": "核自旋量子数"},
        "J": {"type": "number", "description": "电子角动量量子数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_hyperfine_energy_shift 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "A_hz" not in args or args["A_hz"] is None:
                return Observation(self.name, "错误: 缺少必需参数 A_hz")
            if "F" not in args or args["F"] is None:
                return Observation(self.name, "错误: 缺少必需参数 F")
            if "I" not in args or args["I"] is None:
                return Observation(self.name, "错误: 缺少必需参数 I")
            if "J" not in args or args["J"] is None:
                return Observation(self.name, "错误: 缺少必需参数 J")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_hyperfine_energy_shift
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["A_hz", "F", "I", "J"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_hyperfine_energy_shift(**func_kwargs)
            
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


@Toolbox.register(name="calculate_hyperfine_transition_frequency")
class CalculateHyperfineTransitionFrequencyTool(EnvironmentTool):
    """计算类氢原子的超精细跃迁频率。"""
    
    name = "calculate_hyperfine_transition_frequency"
    description = "计算类氢原子的超精细跃迁频率。"
    arguments = {
        "electron_spin": {"type": "number", "description": "电子自旋量子数（例如0.5、1.5）"},
        "nuclear_spin": {"type": "number", "description": "核自旋量子数（质子通常为0.5）"},
        "n": {"type": "integer", "description": "主量子数，默认为1（基态）"},
        "l": {"type": "integer", "description": "轨道角动量量子数，默认为0（s轨道）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_hyperfine_transition_frequency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "electron_spin" not in args or args["electron_spin"] is None:
                return Observation(self.name, "错误: 缺少必需参数 electron_spin")
            if "nuclear_spin" not in args or args["nuclear_spin"] is None:
                return Observation(self.name, "错误: 缺少必需参数 nuclear_spin")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import calculate_hyperfine_transition_frequency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["electron_spin", "nuclear_spin", "n", "l"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_hyperfine_transition_frequency(**func_kwargs)
            
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


@Toolbox.register(name="visualize_hyperfine_levels")
class VisualizeHyperfineLevelsTool(EnvironmentTool):
    """创建超精细结构的能级图。"""
    
    name = "visualize_hyperfine_levels"
    description = "创建超精细结构的能级图。"
    arguments = {
        "electron_spin": {"type": "number", "description": "电子自旋量子数"},
        "nuclear_spin": {"type": "number", "description": "核自旋量子数"},
        "output_path": {"type": "string", "description": "保存图形的路径，默认为./tool_images/hyperfine_levels.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_hyperfine_levels 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "electron_spin" not in args or args["electron_spin"] is None:
                return Observation(self.name, "错误: 缺少必需参数 electron_spin")
            if "nuclear_spin" not in args or args["nuclear_spin"] is None:
                return Observation(self.name, "错误: 缺少必需参数 nuclear_spin")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import visualize_hyperfine_levels
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["electron_spin", "nuclear_spin", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_hyperfine_levels(**func_kwargs)
            
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


@Toolbox.register(name="visualize_spin_comparison")
class VisualizeSpinComparisonTool(EnvironmentTool):
    """创建不同电子自旋的超精细频率比较图。"""
    
    name = "visualize_spin_comparison"
    description = "创建不同电子自旋的超精细频率比较图。"
    arguments = {
        "electron_spins": {"type": "array", "description": "电子自旋值列表"},
        "nuclear_spin": {"type": "number", "description": "核自旋量子数，默认为0.5"},
        "output_path": {"type": "string", "description": "保存图形的路径，默认为./tool_images/spin_comparison.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_spin_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "electron_spins" not in args or args["electron_spins"] is None:
                return Observation(self.name, "错误: 缺少必需参数 electron_spins")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.atomic_physics_toolkit_claude_288 import visualize_spin_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["electron_spins", "nuclear_spin", "output_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_spin_comparison(**func_kwargs)
            
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


@Toolbox.register(name="calculate_energy_uncertainty")
class CalculateEnergyUncertaintyTool(EnvironmentTool):
    """根据量子态寿命计算能量不确定度（自然线宽）"""
    
    name = "calculate_energy_uncertainty"
    description = "根据量子态寿命计算能量不确定度（自然线宽）"
    arguments = {
        "lifetime": {"type": "number", "description": "量子态寿命，单位：秒"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_energy_uncertainty 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "lifetime" not in args or args["lifetime"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_3 import calculate_energy_uncertainty
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["lifetime"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_energy_uncertainty(**func_kwargs)
            
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


@Toolbox.register(name="check_resolution_criterion")
class CheckResolutionCriterionTool(EnvironmentTool):
    """检查能级分辨判据"""
    
    name = "check_resolution_criterion"
    description = "检查能级分辨判据"
    arguments = {
        "energy_diff": {"type": "number", "description": "能级差，单位：eV"},
        "linewidth1": {"type": "number", "description": "第一个能级的线宽，单位：eV"},
        "linewidth2": {"type": "number", "description": "第二个能级的线宽，单位：eV"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 check_resolution_criterion 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "energy_diff" not in args or args["energy_diff"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energy_diff")
            if "linewidth1" not in args or args["linewidth1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 linewidth1")
            if "linewidth2" not in args or args["linewidth2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 linewidth2")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_3 import check_resolution_criterion
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["energy_diff", "linewidth1", "linewidth2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = check_resolution_criterion(**func_kwargs)
            
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


@Toolbox.register(name="analyze_two_state_resolution")
class AnalyzeTwoStateResolutionTool(EnvironmentTool):
    """分析两个量子态的能级分辨性"""
    
    name = "analyze_two_state_resolution"
    description = "分析两个量子态的能级分辨性"
    arguments = {
        "lifetime1": {"type": "number", "description": "第一个量子态寿命，单位：秒"},
        "lifetime2": {"type": "number", "description": "第二个量子态寿命，单位：秒"},
        "energy_diff_eV": {"type": "number", "description": "能级差，单位：eV"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_two_state_resolution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "lifetime1" not in args or args["lifetime1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime1")
            if "lifetime2" not in args or args["lifetime2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime2")
            if "energy_diff_eV" not in args or args["energy_diff_eV"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energy_diff_eV")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_3 import analyze_two_state_resolution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["lifetime1", "lifetime2", "energy_diff_eV"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_two_state_resolution(**func_kwargs)
            
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


@Toolbox.register(name="calculate_minimum_resolvable_energy")
class CalculateMinimumResolvableEnergyTool(EnvironmentTool):
    """计算两个量子态可分辨的最小能级差"""
    
    name = "calculate_minimum_resolvable_energy"
    description = "计算两个量子态可分辨的最小能级差"
    arguments = {
        "lifetime1": {"type": "number", "description": "第一个量子态寿命，单位：秒"},
        "lifetime2": {"type": "number", "description": "第二个量子态寿命，单位：秒"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_minimum_resolvable_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "lifetime1" not in args or args["lifetime1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime1")
            if "lifetime2" not in args or args["lifetime2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime2")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_3 import calculate_minimum_resolvable_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["lifetime1", "lifetime2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_minimum_resolvable_energy(**func_kwargs)
            
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


@Toolbox.register(name="scan_energy_differences")
class ScanEnergyDifferencesTool(EnvironmentTool):
    """扫描一系列能级差，判断哪些可以被分辨"""
    
    name = "scan_energy_differences"
    description = "扫描一系列能级差，判断哪些可以被分辨"
    arguments = {
        "lifetime1": {"type": "number", "description": "第一个量子态寿命，单位：秒"},
        "lifetime2": {"type": "number", "description": "第二个量子态寿命，单位：秒"},
        "energy_range_eV": {"type": "array", "description": "待测试的能级差列表，单位：eV"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 scan_energy_differences 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "lifetime1" not in args or args["lifetime1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime1")
            if "lifetime2" not in args or args["lifetime2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime2")
            if "energy_range_eV" not in args or args["energy_range_eV"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energy_range_eV")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_3 import scan_energy_differences
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["lifetime1", "lifetime2", "energy_range_eV"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = scan_energy_differences(**func_kwargs)
            
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


@Toolbox.register(name="plot_lorentzian_lineshape")
class PlotLorentzianLineshapeTool(EnvironmentTool):
    """绘制洛伦兹线型，展示能级的频谱分布"""
    
    name = "plot_lorentzian_lineshape"
    description = "绘制洛伦兹线型，展示能级的频谱分布"
    arguments = {
        "lifetime1": {"type": "number", "description": "第一个量子态寿命，单位：秒"},
        "lifetime2": {"type": "number", "description": "第二个量子态寿命，单位：秒"},
        "energy_diff_eV": {"type": "number", "description": "能级差，单位：eV"},
        "E1": {"type": "number", "description": "第一个能级的能量，单位：eV"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_lorentzian_lineshape 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "lifetime1" not in args or args["lifetime1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime1")
            if "lifetime2" not in args or args["lifetime2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 lifetime2")
            if "energy_diff_eV" not in args or args["energy_diff_eV"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energy_diff_eV")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_3 import plot_lorentzian_lineshape
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["lifetime1", "lifetime2", "energy_diff_eV", "E1"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_lorentzian_lineshape(**func_kwargs)
            
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


@Toolbox.register(name="create_quantum_state")
class CreateQuantumStateTool(EnvironmentTool):
    """创建量子态向量"""
    
    name = "create_quantum_state"
    description = "创建量子态向量"
    arguments = {
        "state_type": {"type": "string", "description": "量子态类型，可选值：'0', '1', '+', '-', 'custom'"},
        "alpha": {"type": "number", "description": "自定义态的|0⟩系数（仅state_type='custom'时使用）"},
        "beta": {"type": "number", "description": "自定义态的|1⟩系数（仅state_type='custom'时使用）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_quantum_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_type" not in args or args["state_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_type")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import create_quantum_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_type", "alpha", "beta"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_quantum_state(**func_kwargs)
            
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


@Toolbox.register(name="create_hermitian_operator")
class CreateHermitianOperatorTool(EnvironmentTool):
    """创建并验证厄米算符矩阵"""
    
    name = "create_hermitian_operator"
    description = "创建并验证厄米算符矩阵"
    arguments = {
        "matrix_elements": {"type": "array", "description": "矩阵元素的嵌套列表 [[row1], [row2], ...]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_hermitian_operator 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_elements" not in args or args["matrix_elements"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_elements")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import create_hermitian_operator
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_elements"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_hermitian_operator(**func_kwargs)
            
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


@Toolbox.register(name="analyze_operator_spectrum")
class AnalyzeOperatorSpectrumTool(EnvironmentTool):
    """完整分析算符的谱结构（本征值、简并度、本征向量）"""
    
    name = "analyze_operator_spectrum"
    description = "完整分析算符的谱结构（本征值、简并度、本征向量）"
    arguments = {
        "matrix_elements": {"type": "array", "description": "算符矩阵元素"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_operator_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_elements" not in args or args["matrix_elements"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_elements")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import analyze_operator_spectrum
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_elements"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_operator_spectrum(**func_kwargs)
            
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


@Toolbox.register(name="compute_degenerate_subspace_probabilities")
class ComputeDegenerateSubspaceProbabilitiesTool(EnvironmentTool):
    """计算量子态在所有简并子空间的测量概率"""
    
    name = "compute_degenerate_subspace_probabilities"
    description = "计算量子态在所有简并子空间的测量概率"
    arguments = {
        "state_components": {"type": "array", "description": "量子态分量"},
        "matrix_elements": {"type": "array", "description": "算符矩阵元素"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_degenerate_subspace_probabilities 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_components" not in args or args["state_components"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_components")
            if "matrix_elements" not in args or args["matrix_elements"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_elements")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import compute_degenerate_subspace_probabilities
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_components", "matrix_elements"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_degenerate_subspace_probabilities(**func_kwargs)
            
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


@Toolbox.register(name="find_degenerate_eigenvalues")
class FindDegenerateEigenvaluesTool(EnvironmentTool):
    """查找所有简并本征值及其简并度"""
    
    name = "find_degenerate_eigenvalues"
    description = "查找所有简并本征值及其简并度"
    arguments = {
        "matrix_elements": {"type": "array", "description": "算符矩阵元素"},
        "min_degeneracy": {"type": "integer", "description": "最小简并度阈值，默认为2"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_degenerate_eigenvalues 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_elements" not in args or args["matrix_elements"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_elements")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import find_degenerate_eigenvalues
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_elements", "min_degeneracy"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = find_degenerate_eigenvalues(**func_kwargs)
            
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


@Toolbox.register(name="visualize_energy_spectrum")
class VisualizeEnergySpectrumTool(EnvironmentTool):
    """可视化能级谱（包括简并度标注）"""
    
    name = "visualize_energy_spectrum"
    description = "可视化能级谱（包括简并度标注）"
    arguments = {
        "spectrum_data": {"type": "object", "description": "analyze_operator_spectrum的返回结果"},
        "title": {"type": "string", "description": "图表标题，默认为'Energy Spectrum'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_energy_spectrum 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spectrum_data" not in args or args["spectrum_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spectrum_data")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import visualize_energy_spectrum
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spectrum_data", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_energy_spectrum(**func_kwargs)
            
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


@Toolbox.register(name="visualize_measurement_probabilities")
class VisualizeMeasurementProbabilitiesTool(EnvironmentTool):
    """可视化测量概率分布"""
    
    name = "visualize_measurement_probabilities"
    description = "可视化测量概率分布"
    arguments = {
        "probability_data": {"type": "object", "description": "compute_degenerate_subspace_probabilities的返回结果"},
        "title": {"type": "string", "description": "图表标题，默认为'Measurement Probabilities'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_measurement_probabilities 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "probability_data" not in args or args["probability_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 probability_data")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_153 import visualize_measurement_probabilities
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["probability_data", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_measurement_probabilities(**func_kwargs)
            
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


@Toolbox.register(name="validate_quantum_state")
class ValidateQuantumStateTool(EnvironmentTool):
    """验证量子态的物理有效性"""
    
    name = "validate_quantum_state"
    description = "验证量子态的物理有效性"
    arguments = {
        "n": {"type": "integer", "description": "主量子数 (n ≥ 1)"},
        "l": {"type": "integer", "description": "角量子数 (0 ≤ l < n)"},
        "m": {"type": "integer", "description": "磁量子数 (-l ≤ m ≤ l)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 validate_quantum_state 操作"""
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
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import validate_quantum_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n", "l", "m"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = validate_quantum_state(**func_kwargs)
            
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


@Toolbox.register(name="check_dipole_selection_rule")
class CheckDipoleSelectionRuleTool(EnvironmentTool):
    """检查偶极跃迁选择定则"""
    
    name = "check_dipole_selection_rule"
    description = "检查偶极跃迁选择定则"
    arguments = {
        "n1": {"type": "integer", "description": "初态主量子数"},
        "l1": {"type": "integer", "description": "初态角量子数"},
        "m1": {"type": "integer", "description": "初态磁量子数"},
        "n2": {"type": "integer", "description": "末态主量子数"},
        "l2": {"type": "integer", "description": "末态角量子数"},
        "m2": {"type": "integer", "description": "末态磁量子数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 check_dipole_selection_rule 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n1" not in args or args["n1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n1")
            if "l1" not in args or args["l1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l1")
            if "m1" not in args or args["m1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m1")
            if "n2" not in args or args["n2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n2")
            if "l2" not in args or args["l2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l2")
            if "m2" not in args or args["m2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m2")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import check_dipole_selection_rule
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n1", "l1", "m1", "n2", "l2", "m2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = check_dipole_selection_rule(**func_kwargs)
            
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


@Toolbox.register(name="find_all_intermediate_states")
class FindAllIntermediateStatesTool(EnvironmentTool):
    """寻找所有可能的中间态(单步偶极跃迁)"""
    
    name = "find_all_intermediate_states"
    description = "寻找所有可能的中间态(单步偶极跃迁)"
    arguments = {
        "n_initial": {"type": "integer", "description": "初态主量子数"},
        "l_initial": {"type": "integer", "description": "初态角量子数"},
        "m_initial": {"type": "integer", "description": "初态磁量子数"},
        "n_final": {"type": "integer", "description": "末态主量子数"},
        "l_final": {"type": "integer", "description": "末态角量子数"},
        "m_final": {"type": "integer", "description": "末态磁量子数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 find_all_intermediate_states 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_initial" not in args or args["n_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_initial")
            if "l_initial" not in args or args["l_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_initial")
            if "m_initial" not in args or args["m_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_initial")
            if "n_final" not in args or args["n_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_final")
            if "l_final" not in args or args["l_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_final")
            if "m_final" not in args or args["m_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_final")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import find_all_intermediate_states
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_initial", "l_initial", "m_initial", "n_final", "l_final", "m_final"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = find_all_intermediate_states(**func_kwargs)
            
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


@Toolbox.register(name="analyze_all_transition_paths")
class AnalyzeAllTransitionPathsTool(EnvironmentTool):
    """分析所有可能的两步跃迁路径及其概率"""
    
    name = "analyze_all_transition_paths"
    description = "分析所有可能的两步跃迁路径及其概率"
    arguments = {
        "n_initial": {"type": "integer", "description": "初态主量子数"},
        "l_initial": {"type": "integer", "description": "初态角量子数"},
        "m_initial": {"type": "integer", "description": "初态磁量子数"},
        "n_final": {"type": "integer", "description": "末态主量子数"},
        "l_final": {"type": "integer", "description": "末态角量子数"},
        "m_final": {"type": "integer", "description": "末态磁量子数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_all_transition_paths 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_initial" not in args or args["n_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_initial")
            if "l_initial" not in args or args["l_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_initial")
            if "m_initial" not in args or args["m_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_initial")
            if "n_final" not in args or args["n_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_final")
            if "l_final" not in args or args["l_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_final")
            if "m_final" not in args or args["m_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_final")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import analyze_all_transition_paths
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_initial", "l_initial", "m_initial", "n_final", "l_final", "m_final"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_all_transition_paths(**func_kwargs)
            
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


@Toolbox.register(name="visualize_energy_levels_and_transitions")
class VisualizeEnergyLevelsAndTransitionsTool(EnvironmentTool):
    """可视化氢原子能级图和跃迁路径"""
    
    name = "visualize_energy_levels_and_transitions"
    description = "可视化氢原子能级图和跃迁路径"
    arguments = {
        "n_initial": {"type": "integer", "description": "初态主量子数"},
        "l_initial": {"type": "integer", "description": "初态角量子数"},
        "m_initial": {"type": "integer", "description": "初态磁量子数"},
        "n_final": {"type": "integer", "description": "末态主量子数"},
        "l_final": {"type": "integer", "description": "末态角量子数"},
        "m_final": {"type": "integer", "description": "末态磁量子数"},
        "paths": {"type": "array", "description": "跃迁路径列表(来自analyze_all_transition_paths)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_energy_levels_and_transitions 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_initial" not in args or args["n_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_initial")
            if "l_initial" not in args or args["l_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_initial")
            if "m_initial" not in args or args["m_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_initial")
            if "n_final" not in args or args["n_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_final")
            if "l_final" not in args or args["l_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_final")
            if "m_final" not in args or args["m_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_final")
            if "paths" not in args or args["paths"] is None:
                return Observation(self.name, "错误: 缺少必需参数 paths")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import visualize_energy_levels_and_transitions
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_initial", "l_initial", "m_initial", "n_final", "l_final", "m_final", "paths"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_energy_levels_and_transitions(**func_kwargs)
            
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


@Toolbox.register(name="visualize_transition_probabilities")
class VisualizeTransitionProbabilitiesTool(EnvironmentTool):
    """可视化跃迁概率分布(柱状图)"""
    
    name = "visualize_transition_probabilities"
    description = "可视化跃迁概率分布(柱状图)"
    arguments = {
        "paths": {"type": "array", "description": "跃迁路径列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_transition_probabilities 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "paths" not in args or args["paths"] is None:
                return Observation(self.name, "错误: 缺少必需参数 paths")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import visualize_transition_probabilities
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["paths"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_transition_probabilities(**func_kwargs)
            
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


@Toolbox.register(name="create_transition_summary_report")
class CreateTransitionSummaryReportTool(EnvironmentTool):
    """生成跃迁分析的完整报告(文本文件)"""
    
    name = "create_transition_summary_report"
    description = "生成跃迁分析的完整报告(文本文件)"
    arguments = {
        "n_initial": {"type": "integer", "description": "初态主量子数"},
        "l_initial": {"type": "integer", "description": "初态角量子数"},
        "m_initial": {"type": "integer", "description": "初态磁量子数"},
        "n_final": {"type": "integer", "description": "末态主量子数"},
        "l_final": {"type": "integer", "description": "末态角量子数"},
        "m_final": {"type": "integer", "description": "末态磁量子数"},
        "paths": {"type": "array", "description": "跃迁路径列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_transition_summary_report 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "n_initial" not in args or args["n_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_initial")
            if "l_initial" not in args or args["l_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_initial")
            if "m_initial" not in args or args["m_initial"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_initial")
            if "n_final" not in args or args["n_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_final")
            if "l_final" not in args or args["l_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 l_final")
            if "m_final" not in args or args["m_final"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_final")
            if "paths" not in args or args["paths"] is None:
                return Observation(self.name, "错误: 缺少必需参数 paths")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.hydrogen_transition_toolkit_claude_101 import create_transition_summary_report
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["n_initial", "l_initial", "m_initial", "n_final", "l_final", "m_final", "paths"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_transition_summary_report(**func_kwargs)
            
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


@Toolbox.register(name="construct_state_vector")
class ConstructStateVectorTool(EnvironmentTool):
    """从基态系数构造量子态矢量。"""
    
    name = "construct_state_vector"
    description = "从基态系数构造量子态矢量。"
    arguments = {
        "coefficients": {"type": "object", "description": "将基态标签映射到 [实部, 虚部] 的字典，例如 {'00': [0.408, 0], '01': [0.408, 0], ...}"},
        "n_qubits": {"type": "integer", "description": "量子比特数量"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_state_vector 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            if "n_qubits" not in args or args["n_qubits"] is None:
                return Observation(self.name, "错误: 缺少必需参数 n_qubits")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_entanglement_toolkit_claude_147 import construct_state_vector
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["coefficients", "n_qubits"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_state_vector(**func_kwargs)
            
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


@Toolbox.register(name="comprehensive_entanglement_analysis")
class ComprehensiveEntanglementAnalysisTool(EnvironmentTool):
    """使用多种准则执行全面的纠缠分析。"""
    
    name = "comprehensive_entanglement_analysis"
    description = "使用多种准则执行全面的纠缠分析。"
    arguments = {
        "state_vector": {"type": "array", "description": "量子态列表"},
        "dims": {"type": "array", "description": "维度 [dim_A, dim_B]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 comprehensive_entanglement_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_vector" not in args or args["state_vector"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_vector")
            if "dims" not in args or args["dims"] is None:
                return Observation(self.name, "错误: 缺少必需参数 dims")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_entanglement_toolkit_claude_147 import comprehensive_entanglement_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_vector", "dims"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = comprehensive_entanglement_analysis(**func_kwargs)
            
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


@Toolbox.register(name="visualize_density_matrix")
class VisualizeDensityMatrixTool(EnvironmentTool):
    """可视化密度矩阵（实部和虚部）"""
    
    name = "visualize_density_matrix"
    description = "可视化密度矩阵（实部和虚部）"
    arguments = {
        "density_matrix": {"type": "array", "description": "密度矩阵，二维复数数组"},
        "title": {"type": "string", "description": "图表标题，默认为'Density Matrix'"},
        "save_path": {"type": "string", "description": "保存路径（可选），默认为'./tool_images/density_matrix.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_density_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "density_matrix" not in args or args["density_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 density_matrix")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import visualize_density_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["density_matrix", "title", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_density_matrix(**func_kwargs)
            
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


@Toolbox.register(name="visualize_entanglement_summary")
class VisualizeEntanglementSummaryTool(EnvironmentTool):
    """创建纠缠分析结果的综合可视化。"""
    
    name = "visualize_entanglement_summary"
    description = "创建纠缠分析结果的综合可视化。"
    arguments = {
        "analysis_result": {"type": "object", "description": "来自 comprehensive_entanglement_analysis 的输出"},
        "save_path": {"type": "string", "description": "保存图形的可选路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_entanglement_summary 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "analysis_result" not in args or args["analysis_result"] is None:
                return Observation(self.name, "错误: 缺少必需参数 analysis_result")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_entanglement_toolkit_claude_147 import visualize_entanglement_summary
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["analysis_result", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_entanglement_summary(**func_kwargs)
            
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


@Toolbox.register(name="radial_probability_density")
class RadialProbabilityDensityTool(EnvironmentTool):
    """计算径向概率密度 P(r) = |ψ(r)|² * 4πr²。对于球对称波函数，概率密度包含雅可比因子 4πr²。"""
    
    name = "radial_probability_density"
    description = "计算径向概率密度 P(r) = |ψ(r)|² * 4πr²。对于球对称波函数，概率密度包含雅可比因子 4πr²。"
    arguments = {
        "r": {"type": "number", "description": "距离中心的径向距离"},
        "wave_function_type": {"type": "string", "description": "波函数类型: '1/r', 'exponential', 'gaussian', 'constant'"},
        "normalization_const": {"type": "number", "description": "归一化常数 A，默认为1.0"},
        "params": {"type": "object", "description": "波函数的额外参数（如衰减常数等）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 radial_probability_density 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "r" not in args or args["r"] is None:
                return Observation(self.name, "错误: 缺少必需参数 r")
            if "wave_function_type" not in args or args["wave_function_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_type")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import radial_probability_density
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["r", "wave_function_type", "normalization_const", "params"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = radial_probability_density(**func_kwargs)
            
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


@Toolbox.register(name="integrate_probability")
class IntegrateProbabilityTool(EnvironmentTool):
    """计算在径向区间 [r_min, r_max] 内找到粒子的概率。P = ∫[r_min to r_max] |ψ(r)|² * 4πr² dr。"""
    
    name = "integrate_probability"
    description = "计算在径向区间 [r_min, r_max] 内找到粒子的概率。P = ∫[r_min to r_max] |ψ(r)|² * 4πr² dr。"
    arguments = {
        "r_min": {"type": "number", "description": "积分下限"},
        "r_max": {"type": "number", "description": "积分上限"},
        "wave_function_type": {"type": "string", "description": "波函数类型"},
        "normalization_const": {"type": "number", "description": "归一化常数，默认为1.0"},
        "params": {"type": "object", "description": "波函数参数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 integrate_probability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "r_min" not in args or args["r_min"] is None:
                return Observation(self.name, "错误: 缺少必需参数 r_min")
            if "r_max" not in args or args["r_max"] is None:
                return Observation(self.name, "错误: 缺少必需参数 r_max")
            if "wave_function_type" not in args or args["wave_function_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_type")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import integrate_probability
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["r_min", "r_max", "wave_function_type", "normalization_const", "params"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = integrate_probability(**func_kwargs)
            
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


@Toolbox.register(name="symbolic_wave_function_analysis")
class SymbolicWaveFunctionAnalysisTool(EnvironmentTool):
    """使用符号计算分析波函数的性质。"""
    
    name = "symbolic_wave_function_analysis"
    description = "使用符号计算分析波函数的性质。"
    arguments = {
        "wave_function_expr": {"type": "string", "description": "波函数的符号表达式，如 'A/r', 'A*exp(-alpha*r)'"},
        "r_symbol": {"type": "string", "description": "径向变量符号，默认为'r'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 symbolic_wave_function_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wave_function_expr" not in args or args["wave_function_expr"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_expr")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import symbolic_wave_function_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wave_function_expr", "r_symbol"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = symbolic_wave_function_analysis(**func_kwargs)
            
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


@Toolbox.register(name="deduce_wave_function_from_probability_conditions")
class DeduceWaveFunctionFromProbabilityConditionsTool(EnvironmentTool):
    """根据概率条件推导波函数形式。给定在不同区间内概率相等的条件，推导满足条件的波函数类型。"""
    
    name = "deduce_wave_function_from_probability_conditions"
    description = "根据概率条件推导波函数形式。给定在不同区间内概率相等的条件，推导满足条件的波函数类型。"
    arguments = {
        "d1": {"type": "number", "description": "内球半径"},
        "outer_inner_ratio": {"type": "number", "description": "外球与内球半径比"},
        "probability_equal_intervals": {"type": "array", "description": "概率相等的区间列表，每个元组为 (r_min_factor, r_max_factor)，例如 [(1, 2), (2, 3)] 表示 [d1, 2*d1] 和 [2*d1, 3*d1]"},
        "candidate_wave_functions": {"type": "array", "description": "候选波函数类型列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 deduce_wave_function_from_probability_conditions 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "d1" not in args or args["d1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d1")
            if "outer_inner_ratio" not in args or args["outer_inner_ratio"] is None:
                return Observation(self.name, "错误: 缺少必需参数 outer_inner_ratio")
            if "probability_equal_intervals" not in args or args["probability_equal_intervals"] is None:
                return Observation(self.name, "错误: 缺少必需参数 probability_equal_intervals")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import deduce_wave_function_from_probability_conditions
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["d1", "outer_inner_ratio", "probability_equal_intervals", "candidate_wave_functions"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = deduce_wave_function_from_probability_conditions(**func_kwargs)
            
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


@Toolbox.register(name="verify_wave_function_properties")
class VerifyWaveFunctionPropertiesTool(EnvironmentTool):
    """验证波函数的各种性质。"""
    
    name = "verify_wave_function_properties"
    description = "验证波函数的各种性质。"
    arguments = {
        "wave_function_type": {"type": "string", "description": "波函数类型"},
        "d1": {"type": "number", "description": "内球半径"},
        "d_outer": {"type": "number", "description": "外球半径"},
        "test_intervals": {"type": "array", "description": "测试区间列表，每个元组为 (r_min, r_max)"},
        "normalization_const": {"type": "number", "description": "归一化常数（如果为None则自动计算）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 verify_wave_function_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wave_function_type" not in args or args["wave_function_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_type")
            if "d1" not in args or args["d1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d1")
            if "d_outer" not in args or args["d_outer"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d_outer")
            if "test_intervals" not in args or args["test_intervals"] is None:
                return Observation(self.name, "错误: 缺少必需参数 test_intervals")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import verify_wave_function_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wave_function_type", "d1", "d_outer", "test_intervals", "normalization_const"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = verify_wave_function_properties(**func_kwargs)
            
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


@Toolbox.register(name="compare_wave_function_candidates")
class CompareWaveFunctionCandidatesTool(EnvironmentTool):
    """比较多个候选波函数，找出最符合条件的。"""
    
    name = "compare_wave_function_candidates"
    description = "比较多个候选波函数，找出最符合条件的。"
    arguments = {
        "d1": {"type": "number", "description": "内球半径"},
        "outer_inner_ratio": {"type": "number", "description": "外球与内球半径比"},
        "equal_prob_intervals": {"type": "array", "description": "应该具有相等概率的区间，每个元组为 (r_min_factor, r_max_factor)"},
        "candidates": {"type": "array", "description": "候选波函数列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_wave_function_candidates 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "d1" not in args or args["d1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d1")
            if "outer_inner_ratio" not in args or args["outer_inner_ratio"] is None:
                return Observation(self.name, "错误: 缺少必需参数 outer_inner_ratio")
            if "equal_prob_intervals" not in args or args["equal_prob_intervals"] is None:
                return Observation(self.name, "错误: 缺少必需参数 equal_prob_intervals")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import compare_wave_function_candidates
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["d1", "outer_inner_ratio", "equal_prob_intervals", "candidates"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_wave_function_candidates(**func_kwargs)
            
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


@Toolbox.register(name="plot_wave_function_and_probability")
class PlotWaveFunctionAndProbabilityTool(EnvironmentTool):
    """绘制波函数和概率密度分布。"""
    
    name = "plot_wave_function_and_probability"
    description = "绘制波函数和概率密度分布。"
    arguments = {
        "wave_function_type": {"type": "string", "description": "波函数类型"},
        "d1": {"type": "number", "description": "内球半径"},
        "d_outer": {"type": "number", "description": "外球半径"},
        "normalization_const": {"type": "number", "description": "归一化常数"},
        "highlight_intervals": {"type": "array", "description": "需要高亮显示的区间，每个元组为 (r_min, r_max)"},
        "num_points": {"type": "integer", "description": "绘图点数，默认为1000"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_wave_function_and_probability 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wave_function_type" not in args or args["wave_function_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_type")
            if "d1" not in args or args["d1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d1")
            if "d_outer" not in args or args["d_outer"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d_outer")
            if "normalization_const" not in args or args["normalization_const"] is None:
                return Observation(self.name, "错误: 缺少必需参数 normalization_const")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import plot_wave_function_and_probability
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wave_function_type", "d1", "d_outer", "normalization_const", "highlight_intervals", "num_points"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_wave_function_and_probability(**func_kwargs)
            
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


@Toolbox.register(name="plot_probability_comparison")
class PlotProbabilityComparisonTool(EnvironmentTool):
    """比较不同波函数在各区间的概率分布。"""
    
    name = "plot_probability_comparison"
    description = "比较不同波函数在各区间的概率分布。"
    arguments = {
        "d1": {"type": "number", "description": "内球半径"},
        "d_outer": {"type": "number", "description": "外球半径"},
        "wave_function_types": {"type": "array", "description": "要比较的波函数类型列表"},
        "test_intervals": {"type": "array", "description": "测试区间，每个元组为 (r_min, r_max)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_probability_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "d1" not in args or args["d1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d1")
            if "d_outer" not in args or args["d_outer"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d_outer")
            if "wave_function_types" not in args or args["wave_function_types"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_types")
            if "test_intervals" not in args or args["test_intervals"] is None:
                return Observation(self.name, "错误: 缺少必需参数 test_intervals")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import plot_probability_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["d1", "d_outer", "wave_function_types", "test_intervals"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_probability_comparison(**func_kwargs)
            
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


@Toolbox.register(name="visualize_spherical_probability_distribution")
class VisualizeSphericalProbabilityDistributionTool(EnvironmentTool):
    """可视化球壳概率分布（3D效果的2D投影）。"""
    
    name = "visualize_spherical_probability_distribution"
    description = "可视化球壳概率分布（3D效果的2D投影）。"
    arguments = {
        "wave_function_type": {"type": "string", "description": "波函数类型"},
        "d1": {"type": "number", "description": "内球半径"},
        "d_outer": {"type": "number", "description": "外球半径"},
        "normalization_const": {"type": "number", "description": "归一化常数"},
        "num_shells": {"type": "integer", "description": "球壳数量，默认为50"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_spherical_probability_distribution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "wave_function_type" not in args or args["wave_function_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 wave_function_type")
            if "d1" not in args or args["d1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d1")
            if "d_outer" not in args or args["d_outer"] is None:
                return Observation(self.name, "错误: 缺少必需参数 d_outer")
            if "normalization_const" not in args or args["normalization_const"] is None:
                return Observation(self.name, "错误: 缺少必需参数 normalization_const")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_107 import visualize_spherical_probability_distribution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["wave_function_type", "d1", "d_outer", "normalization_const", "num_shells"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_spherical_probability_distribution(**func_kwargs)
            
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


@Toolbox.register(name="analyze_pair_production_threshold")
class AnalyzePairProductionThresholdTool(EnvironmentTool):
    """分析光子对产生反应的阈值条件（γγ → particle-antiparticle），完成从CMB能量到阈值γ射线能量的完整计算链"""
    
    name = "analyze_pair_production_threshold"
    description = "分析光子对产生反应的阈值条件（γγ → particle-antiparticle），完成从CMB能量到阈值γ射线能量的完整计算链"
    arguments = {
        "cmb_energy": {"type": "number", "description": "CMB光子能量 (eV)，默认10^-3 eV"},
        "particle_type": {"type": "string", "description": "产生的粒子类型，可选'electron', 'muon', 'pion'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_pair_production_threshold 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.particle_physics_toolkit_0 import analyze_pair_production_threshold
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cmb_energy", "particle_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_pair_production_threshold(**func_kwargs)
            
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


@Toolbox.register(name="analyze_gzk_cutoff")
class AnalyzeGzkCutoffTool(EnvironmentTool):
    """分析GZK截断效应（Greisen-Zatsepin-Kuzmin cutoff），完整计算链：CMB温度 → 光子能量 → 阈值γ能量 → 反应截面 → 平均自由程"""
    
    name = "analyze_gzk_cutoff"
    description = "分析GZK截断效应（Greisen-Zatsepin-Kuzmin cutoff），完整计算链：CMB温度 → 光子能量 → 阈值γ能量 → 反应截面 → 平均自由程"
    arguments = {
        "cmb_temperature": {"type": "number", "description": "CMB温度 (K)"},
        "particle_type": {"type": "string", "description": "产生的粒子类型"},
        "energy_type": {"type": "string", "description": "CMB能量类型 ('average', 'peak', 'rms')"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_gzk_cutoff 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.particle_physics_toolkit_0 import analyze_gzk_cutoff
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cmb_temperature", "particle_type", "energy_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_gzk_cutoff(**func_kwargs)
            
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


@Toolbox.register(name="scan_threshold_vs_cmb_energy")
class ScanThresholdVsCmbEnergyTool(EnvironmentTool):
    """扫描不同CMB能量下的阈值能量变化，用于研究背景辐射能量对高能粒子传播的影响"""
    
    name = "scan_threshold_vs_cmb_energy"
    description = "扫描不同CMB能量下的阈值能量变化，用于研究背景辐射能量对高能粒子传播的影响"
    arguments = {
        "cmb_energy_range": {"type": "array", "description": "CMB能量列表 (eV)"},
        "particle_type": {"type": "string", "description": "粒子类型"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 scan_threshold_vs_cmb_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "cmb_energy_range" not in args or args["cmb_energy_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 cmb_energy_range")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.particle_physics_toolkit_0 import scan_threshold_vs_cmb_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cmb_energy_range", "particle_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = scan_threshold_vs_cmb_energy(**func_kwargs)
            
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


@Toolbox.register(name="compare_particle_thresholds")
class CompareParticleThresholdsTool(EnvironmentTool):
    """比较不同粒子的对产生阈值"""
    
    name = "compare_particle_thresholds"
    description = "比较不同粒子的对产生阈值"
    arguments = {
        "cmb_energy": {"type": "number", "description": "CMB光子能量 (eV)"},
        "particle_types": {"type": "array", "description": "粒子类型列表，默认['electron', 'muon', 'pion']"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compare_particle_thresholds 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.particle_physics_toolkit_0 import compare_particle_thresholds
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cmb_energy", "particle_types"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compare_particle_thresholds(**func_kwargs)
            
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


@Toolbox.register(name="visualize_threshold_energy_scan")
class VisualizeThresholdEnergyScanTool(EnvironmentTool):
    """可视化阈值能量随CMB能量的变化"""
    
    name = "visualize_threshold_energy_scan"
    description = "可视化阈值能量随CMB能量的变化"
    arguments = {
        "cmb_energy_range": {"type": "array", "description": "CMB能量范围 (eV)"},
        "particle_types": {"type": "array", "description": "粒子类型列表"},
        "save_dir": {"type": "string", "description": "保存目录"},
        "filename": {"type": "string", "description": "文件名（不含扩展名）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_threshold_energy_scan 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "cmb_energy_range" not in args or args["cmb_energy_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 cmb_energy_range")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.particle_physics_toolkit_0 import visualize_threshold_energy_scan
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["cmb_energy_range", "particle_types", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_threshold_energy_scan(**func_kwargs)
            
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


@Toolbox.register(name="visualize_mean_free_path_vs_energy")
class VisualizeMeanFreePathVsEnergyTool(EnvironmentTool):
    """可视化平均自由程随γ射线能量的变化"""
    
    name = "visualize_mean_free_path_vs_energy"
    description = "可视化平均自由程随γ射线能量的变化"
    arguments = {
        "energy_range_GeV": {"type": "array", "description": "γ射线能量范围 (GeV)"},
        "cmb_temperature": {"type": "number", "description": "CMB温度 (K)"},
        "save_dir": {"type": "string", "description": "保存目录"},
        "filename": {"type": "string", "description": "文件名"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_mean_free_path_vs_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "energy_range_GeV" not in args or args["energy_range_GeV"] is None:
                return Observation(self.name, "错误: 缺少必需参数 energy_range_GeV")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.particle_physics_toolkit_0 import visualize_mean_free_path_vs_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["energy_range_GeV", "cmb_temperature", "save_dir", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_mean_free_path_vs_energy(**func_kwargs)
            
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


@Toolbox.register(name="parse_matrix_string")
class ParseMatrixStringTool(EnvironmentTool):
    """解析矩阵字符串为列表格式。支持带括号的格式，如 "(1,2;3,4)"；支持虚数单位 i，如 "(i, -1, 2i; 1, 0, 1; 2i, -1, -i)"。"""
    
    name = "parse_matrix_string"
    description = (
        '解析矩阵字符串为列表格式。支持带括号的格式，如 "(1,2;3,4)"；'
        '支持虚数单位 i，如 "(i, -1, 2i; 1, 0, 1; 2i, -1, -i)"'
    )
    arguments = {
        "matrix_str": {
            "type": "string",
            "description": (
                '矩阵字符串，行用分号分隔，如 "1,2;3,4" 或 '
                '"(i, -1, 2i; 1, 0, 1; 2i, -1, -i)"，支持带括号的格式和虚数单位 i'
            ),
        }
    }
    
    def use(self, environment, action) -> Observation:
        """执行 parse_matrix_string 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_str" not in args or args["matrix_str"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_str")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import parse_matrix_string
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_str"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = parse_matrix_string(**func_kwargs)
            
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


@Toolbox.register(name="is_unitary")
class IsUnitaryTool(EnvironmentTool):
    """检查矩阵是否为酉矩阵（Unitary）。酉矩阵满足: U * U^† = I"""
    
    name = "is_unitary"
    description = "检查矩阵是否为酉矩阵（Unitary）。酉矩阵满足: U * U^† = I"
    arguments = {
        "matrix_list": {"type": "array", "description": "矩阵的列表表示"},
        "tolerance": {"type": "number", "description": "数值容差，默认为1e-10"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 is_unitary 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_list" not in args or args["matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import is_unitary
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_list", "tolerance"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = is_unitary(**func_kwargs)
            
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


@Toolbox.register(name="is_density_matrix")
class IsDensityMatrixTool(EnvironmentTool):
    """检查矩阵是否为有效的密度矩阵（量子态）。密度矩阵必须满足：1. 厄米性 (Hermitian) 2. 半正定性 (Positive semi-definite) 3. 迹为1 (Trace = 1)"""
    
    name = "is_density_matrix"
    description = "检查矩阵是否为有效的密度矩阵（量子态）。密度矩阵必须满足：1. 厄米性 (Hermitian) 2. 半正定性 (Positive semi-definite) 3. 迹为1 (Trace = 1)"
    arguments = {
        "matrix_list": {"type": "array", "description": "矩阵的列表表示"},
        "tolerance": {"type": "number", "description": "数值容差，默认为1e-10"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 is_density_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_list" not in args or args["matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import is_density_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_list", "tolerance"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = is_density_matrix(**func_kwargs)
            
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


@Toolbox.register(name="similarity_transformation")
class SimilarityTransformationTool(EnvironmentTool):
    """相似变换: U * A * U^†。在量子力学中，酉变换保持密度矩阵的性质"""
    
    name = "similarity_transformation"
    description = "相似变换: U * A * U^†。在量子力学中，酉变换保持密度矩阵的性质"
    arguments = {
        "matrix_list": {"type": "array", "description": "要变换的矩阵 A"},
        "transform_matrix_list": {"type": "array", "description": "变换矩阵 U"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 similarity_transformation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_list" not in args or args["matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_list")
            if "transform_matrix_list" not in args or args["transform_matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 transform_matrix_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import similarity_transformation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_list", "transform_matrix_list"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = similarity_transformation(**func_kwargs)
            
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


@Toolbox.register(name="exponential_similarity_transformation")
class ExponentialSimilarityTransformationTool(EnvironmentTool):
    """指数相似变换: e^X * A * e^(-X)"""
    
    name = "exponential_similarity_transformation"
    description = "指数相似变换: e^X * A * e^(-X)"
    arguments = {
        "matrix_list": {"type": "array", "description": "要变换的矩阵 A"},
        "exponent_matrix_list": {"type": "array", "description": "指数矩阵 X"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 exponential_similarity_transformation 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_list" not in args or args["matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_list")
            if "exponent_matrix_list" not in args or args["exponent_matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 exponent_matrix_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import exponential_similarity_transformation
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_list", "exponent_matrix_list"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = exponential_similarity_transformation(**func_kwargs)
            
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


@Toolbox.register(name="analyze_matrix_properties")
class AnalyzeMatrixPropertiesTool(EnvironmentTool):
    """全面分析矩阵的量子力学性质"""
    
    name = "analyze_matrix_properties"
    description = "全面分析矩阵的量子力学性质"
    arguments = {
        "matrix_list": {"type": "array", "description": "矩阵的列表表示"},
        "matrix_name": {"type": "string", "description": '矩阵名称，默认为 "Matrix"'},
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_matrix_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_list" not in args or args["matrix_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import analyze_matrix_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_list", "matrix_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_matrix_properties(**func_kwargs)
            
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


@Toolbox.register(name="visualize_matrix_properties")
class VisualizeMatrixPropertiesTool(EnvironmentTool):
    """可视化矩阵性质"""
    
    name = "visualize_matrix_properties"
    description = "可视化矩阵性质"
    arguments = {
        "properties_dict": {"type": "object", "description": "矩阵性质字典"},
        "matrix_name": {"type": "string", "description": "矩阵名称"},
        "save_path": {"type": "string", "description": "保存路径，默认为'./tool_images/matrix_properties.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_matrix_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "properties_dict" not in args or args["properties_dict"] is None:
                return Observation(self.name, "错误: 缺少必需参数 properties_dict")
            if "matrix_name" not in args or args["matrix_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_name")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import visualize_matrix_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["properties_dict", "matrix_name", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_matrix_properties(**func_kwargs)
            
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


@Toolbox.register(name="visualize_eigenvalue_comparison")
class VisualizeEigenvalueComparisonTool(EnvironmentTool):
    """比较多个矩阵的特征值"""
    
    name = "visualize_eigenvalue_comparison"
    description = "比较多个矩阵的特征值"
    arguments = {
        "matrices_dict": {"type": "object", "description": "矩阵字典，格式为 {matrix_name: properties_dict}"},
        "save_path": {"type": "string", "description": "保存路径，默认为'./tool_images/eigenvalue_comparison.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_eigenvalue_comparison 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrices_dict" not in args or args["matrices_dict"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrices_dict")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_146 import visualize_eigenvalue_comparison
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrices_dict", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_eigenvalue_comparison(**func_kwargs)
            
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


@Toolbox.register(name="create_density_matrix")
class CreateDensityMatrixTool(EnvironmentTool):
    """从态向量创建密度矩阵 ρ = |ψ⟩⟨ψ|"""
    
    name = "create_density_matrix"
    description = "从态向量创建密度矩阵 ρ = |ψ⟩⟨ψ|"
    arguments = {
        "state_vector": {"type": "array", "description": "量子态向量 [[c0_real, c0_imag], [c1_real, c1_imag]]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_density_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_vector" not in args or args["state_vector"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_vector")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import create_density_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_vector"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_density_matrix(**func_kwargs)
            
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


@Toolbox.register(name="verify_kraus_completeness")
class VerifyKrausCompletenessTool(EnvironmentTool):
    """验证Kraus算符的完备性关系: Σ A_i† A_i = I"""
    
    name = "verify_kraus_completeness"
    description = "验证Kraus算符的完备性关系: Σ A_i† A_i = I"
    arguments = {
        "kraus_operators": {"type": "array", "description": "Kraus算符列表 [[[a00, a01], [a10, a11]], ...]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 verify_kraus_completeness 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "kraus_operators" not in args or args["kraus_operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kraus_operators")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import verify_kraus_completeness
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["kraus_operators"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = verify_kraus_completeness(**func_kwargs)
            
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


@Toolbox.register(name="construct_bit_flip_channel")
class ConstructBitFlipChannelTool(EnvironmentTool):
    """构建比特翻转信道的Kraus算符"""
    
    name = "construct_bit_flip_channel"
    description = "构建比特翻转信道的Kraus算符"
    arguments = {
        "flip_probability": {"type": "number", "description": "比特翻转概率 p ∈ [0, 1]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 construct_bit_flip_channel 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "flip_probability" not in args or args["flip_probability"] is None:
                return Observation(self.name, "错误: 缺少必需参数 flip_probability")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import construct_bit_flip_channel
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["flip_probability"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = construct_bit_flip_channel(**func_kwargs)
            
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


@Toolbox.register(name="apply_quantum_channel")
class ApplyQuantumChannelTool(EnvironmentTool):
    """应用完整的量子信道: E(ρ) = Σ A_i ρ A_i†"""
    
    name = "apply_quantum_channel"
    description = "应用完整的量子信道: E(ρ) = Σ A_i ρ A_i†"
    arguments = {
        "density_matrix": {"type": "array", "description": "输入密度矩阵"},
        "kraus_operators": {"type": "array", "description": "Kraus算符列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 apply_quantum_channel 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "density_matrix" not in args or args["density_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 density_matrix")
            if "kraus_operators" not in args or args["kraus_operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kraus_operators")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import apply_quantum_channel
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["density_matrix", "kraus_operators"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = apply_quantum_channel(**func_kwargs)
            
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


@Toolbox.register(name="compute_fidelity")
class ComputeFidelityTool(EnvironmentTool):
    """计算两个量子态的保真度 F(ρ1, ρ2) = Tr(√(√ρ1 ρ2 √ρ1))²"""
    
    name = "compute_fidelity"
    description = "计算两个量子态的保真度 F(ρ1, ρ2) = Tr(√(√ρ1 ρ2 √ρ1))²"
    arguments = {
        "rho1": {"type": "array", "description": "第一个密度矩阵"},
        "rho2": {"type": "array", "description": "第二个密度矩阵"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_fidelity 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "rho1" not in args or args["rho1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 rho1")
            if "rho2" not in args or args["rho2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 rho2")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import compute_fidelity
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["rho1", "rho2"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_fidelity(**func_kwargs)
            
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


@Toolbox.register(name="analyze_channel_properties")
class AnalyzeChannelPropertiesTool(EnvironmentTool):
    """分析量子信道的性质"""
    
    name = "analyze_channel_properties"
    description = "分析量子信道的性质"
    arguments = {
        "kraus_operators": {"type": "array", "description": "Kraus算符列表"},
        "test_states": {"type": "array", "description": "测试态列表，例如 ['0', '1', '+', '-']"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_channel_properties 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "kraus_operators" not in args or args["kraus_operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kraus_operators")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import analyze_channel_properties
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["kraus_operators", "test_states"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_channel_properties(**func_kwargs)
            
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


@Toolbox.register(name="visualize_bloch_sphere_evolution")
class VisualizeBlochSphereEvolutionTool(EnvironmentTool):
    """可视化量子态在Bloch球面上的演化"""
    
    name = "visualize_bloch_sphere_evolution"
    description = "可视化量子态在Bloch球面上的演化"
    arguments = {
        "initial_state": {"type": "string", "description": "初始态类型，可选值：'0', '1', '+', '-'"},
        "kraus_operators": {"type": "array", "description": "Kraus算符列表"},
        "num_steps": {"type": "integer", "description": "演化步数，默认为10"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_bloch_sphere_evolution 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "initial_state" not in args or args["initial_state"] is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_state")
            if "kraus_operators" not in args or args["kraus_operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kraus_operators")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import visualize_bloch_sphere_evolution
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["initial_state", "kraus_operators", "num_steps"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_bloch_sphere_evolution(**func_kwargs)
            
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


@Toolbox.register(name="visualize_kraus_operators")
class VisualizeKrausOperatorsTool(EnvironmentTool):
    """可视化Kraus算符的矩阵表示"""
    
    name = "visualize_kraus_operators"
    description = "可视化Kraus算符的矩阵表示"
    arguments = {
        "kraus_operators": {"type": "array", "description": "Kraus算符列表"},
        "operator_names": {"type": "array", "description": "算符名称列表，例如 ['A_0', 'A_1', ...]"},
        "title": {"type": "string", "description": "图表标题，默认为'Kraus Operators'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_kraus_operators 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "kraus_operators" not in args or args["kraus_operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kraus_operators")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import visualize_kraus_operators
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["kraus_operators", "operator_names", "title"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_kraus_operators(**func_kwargs)
            
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


@Toolbox.register(name="visualize_channel_effect")
class VisualizeChannelEffectTool(EnvironmentTool):
    """可视化量子信道对不同输入态的影响"""
    
    name = "visualize_channel_effect"
    description = "可视化量子信道对不同输入态的影响"
    arguments = {
        "kraus_operators": {"type": "array", "description": "Kraus算符列表"},
        "test_states": {"type": "array", "description": "测试态列表，例如 ['0', '1', '+', '-']"},
        "channel_name": {"type": "string", "description": "信道名称，默认为'Quantum Channel'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_channel_effect 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "kraus_operators" not in args or args["kraus_operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 kraus_operators")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import visualize_channel_effect
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["kraus_operators", "test_states", "channel_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_channel_effect(**func_kwargs)
            
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


@Toolbox.register(name="save_analysis_report")
class SaveAnalysisReportTool(EnvironmentTool):
    """保存量子信道分析报告"""
    
    name = "save_analysis_report"
    description = "保存量子信道分析报告"
    arguments = {
        "analysis_data": {"type": "object", "description": "分析数据字典"},
        "filename": {"type": "string", "description": "输出文件名，默认为'quantum_channel_report.json'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 save_analysis_report 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "analysis_data" not in args or args["analysis_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 analysis_data")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_operations_toolkit_claude_229 import save_analysis_report
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["analysis_data", "filename"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = save_analysis_report(**func_kwargs)
            
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


@Toolbox.register(name="create_spin_state")
class CreateSpinStateTool(EnvironmentTool):
    """创建沿特定方向的自旋本征态"""
    
    name = "create_spin_state"
    description = "创建沿特定方向的自旋本征态"
    arguments = {
        "direction": {"type": "string", "description": "自旋方向，可选 'up_z', 'down_z', 'up_x', 'down_x', 'up_y', 'down_y'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_spin_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "direction" not in args or args["direction"] is None:
                return Observation(self.name, "错误: 缺少必需参数 direction")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import create_spin_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["direction"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_spin_state(**func_kwargs)
            
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


@Toolbox.register(name="get_pauli_matrix")
class GetPauliMatrixTool(EnvironmentTool):
    """获取泡利矩阵"""
    
    name = "get_pauli_matrix"
    description = "获取泡利矩阵"
    arguments = {
        "matrix_type": {"type": "string", "description": "泡利矩阵类型，可选值：'x', 'y', 'z', 'identity', 'I'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_pauli_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "matrix_type" not in args or args["matrix_type"] is None:
                return Observation(self.name, "错误: 缺少必需参数 matrix_type")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import get_pauli_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["matrix_type"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_pauli_matrix(**func_kwargs)
            
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


@Toolbox.register(name="compute_expectation_value")
class ComputeExpectationValueTool(EnvironmentTool):
    """计算算符在给定密度矩阵下的期望值"""
    
    name = "compute_expectation_value"
    description = "计算算符在给定密度矩阵下的期望值"
    arguments = {
        "density_matrix": {"type": "array", "description": "密度矩阵，二维复数数组 [[...], [...]]"},
        "operator": {"type": "array", "description": "算符矩阵，二维复数数组 [[...], [...]]"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_expectation_value 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "density_matrix" not in args or args["density_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 density_matrix")
            if "operator" not in args or args["operator"] is None:
                return Observation(self.name, "错误: 缺少必需参数 operator")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import compute_expectation_value
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["density_matrix", "operator"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_expectation_value(**func_kwargs)
            
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


@Toolbox.register(name="create_mixed_state_density_matrix")
class CreateMixedStateDensityMatrixTool(EnvironmentTool):
    """创建混合态的密度矩阵"""
    
    name = "create_mixed_state_density_matrix"
    description = "创建混合态的密度矩阵"
    arguments = {
        "states": {"type": "array", "description": "状态向量列表，每个元素是包含'state'字段的字典对象，如 {'state': [c1, c2]}"},
        "probabilities": {"type": "array", "description": "对应的概率列表，所有概率之和必须为1"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_mixed_state_density_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "states" not in args or args["states"] is None:
                return Observation(self.name, "错误: 缺少必需参数 states")
            if "probabilities" not in args or args["probabilities"] is None:
                return Observation(self.name, "错误: 缺少必需参数 probabilities")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import create_mixed_state_density_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["states", "probabilities"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_mixed_state_density_matrix(**func_kwargs)
            
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


@Toolbox.register(name="create_linear_combination_operator")
class CreateLinearCombinationOperatorTool(EnvironmentTool):
    """创建算符的线性组合"""
    
    name = "create_linear_combination_operator"
    description = "创建算符的线性组合"
    arguments = {
        "operators": {"type": "array", "description": "算符列表，每个元素是包含'matrix'字段的字典对象，如 {'matrix': [[...], [...]]}"},
        "coefficients": {"type": "array", "description": "对应的系数列表"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_linear_combination_operator 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "operators" not in args or args["operators"] is None:
                return Observation(self.name, "错误: 缺少必需参数 operators")
            if "coefficients" not in args or args["coefficients"] is None:
                return Observation(self.name, "错误: 缺少必需参数 coefficients")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import create_linear_combination_operator
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["operators", "coefficients"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_linear_combination_operator(**func_kwargs)
            
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


@Toolbox.register(name="analyze_quantum_state")
class AnalyzeQuantumStateTool(EnvironmentTool):
    """分析量子态的各种性质"""
    
    name = "analyze_quantum_state"
    description = "分析量子态的各种性质"
    arguments = {
        "state_vector": {"type": "array", "description": "状态向量，复数列表形式"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_quantum_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_vector" not in args or args["state_vector"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_vector")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import analyze_quantum_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_vector"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_quantum_state(**func_kwargs)
            
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


@Toolbox.register(name="visualize_bloch_sphere")
class VisualizeBlochSphereTool(EnvironmentTool):
    """在布洛赫球上可视化自旋-1/2态"""
    
    name = "visualize_bloch_sphere"
    description = "在布洛赫球上可视化自旋-1/2态"
    arguments = {
        "state_vector": {"type": "array", "description": "状态向量 [c_up, c_down]，仅支持二维自旋态"},
        "title": {"type": "string", "description": "图表标题，默认为'Bloch Sphere Representation'"},
        "save_path": {"type": "string", "description": "保存路径（可选），默认为'./tool_images/bloch_sphere.png'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_bloch_sphere 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "state_vector" not in args or args["state_vector"] is None:
                return Observation(self.name, "错误: 缺少必需参数 state_vector")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_mechanics_toolkit_claude_142 import visualize_bloch_sphere
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["state_vector", "title", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_bloch_sphere(**func_kwargs)
            
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


@Toolbox.register(name="calculate_spin_states")
class CalculateSpinStatesTool(EnvironmentTool):
    """计算给定自旋量子数的可能自旋态数量"""
    
    name = "calculate_spin_states"
    description = "计算给定自旋量子数的可能自旋态数量"
    arguments = {
        "spin_quantum_number": {"type": "number", "description": "自旋量子数 (如1/2, 1, 3/2等)"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_spin_states 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "spin_quantum_number" not in args or args["spin_quantum_number"] is None:
                return Observation(self.name, "错误: 缺少必需参数 spin_quantum_number")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.nmr_spin_system_toolkit_claude_64 import calculate_spin_states
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["spin_quantum_number"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_spin_states(**func_kwargs)
            
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


@Toolbox.register(name="calculate_zeeman_energy")
class CalculateZeemanEnergyTool(EnvironmentTool):
    """计算Zeeman能级（考虑化学位移）"""
    
    name = "calculate_zeeman_energy"
    description = "计算Zeeman能级（考虑化学位移）"
    arguments = {
        "magnetic_field": {"type": "number", "description": "磁场强度 (Tesla)"},
        "gyromagnetic_ratio": {"type": "number", "description": "旋磁比 (rad/(s·T))"},
        "m_quantum_number": {"type": "number", "description": "磁量子数"},
        "chemical_shift": {"type": "number", "description": "化学位移 (ppm)，默认为0.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_zeeman_energy 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "magnetic_field" not in args or args["magnetic_field"] is None:
                return Observation(self.name, "错误: 缺少必需参数 magnetic_field")
            if "gyromagnetic_ratio" not in args or args["gyromagnetic_ratio"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gyromagnetic_ratio")
            if "m_quantum_number" not in args or args["m_quantum_number"] is None:
                return Observation(self.name, "错误: 缺少必需参数 m_quantum_number")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.nmr_spin_system_toolkit_claude_64 import calculate_zeeman_energy
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["magnetic_field", "gyromagnetic_ratio", "m_quantum_number", "chemical_shift"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_zeeman_energy(**func_kwargs)
            
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


@Toolbox.register(name="calculate_dipolar_coupling")
class CalculateDipolarCouplingTool(EnvironmentTool):
    """计算偶极耦合常数"""
    
    name = "calculate_dipolar_coupling"
    description = "计算偶极耦合常数"
    arguments = {
        "distance": {"type": "number", "description": "核间距离 (angstroms)"},
        "gyromagnetic_ratio_1": {"type": "number", "description": "核1的旋磁比 (rad/(s·T))"},
        "gyromagnetic_ratio_2": {"type": "number", "description": "核2的旋磁比 (rad/(s·T))"},
        "theta": {"type": "number", "description": "核间矢量与磁场夹角 (度)，默认为0.0"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 calculate_dipolar_coupling 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "distance" not in args or args["distance"] is None:
                return Observation(self.name, "错误: 缺少必需参数 distance")
            if "gyromagnetic_ratio_1" not in args or args["gyromagnetic_ratio_1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gyromagnetic_ratio_1")
            if "gyromagnetic_ratio_2" not in args or args["gyromagnetic_ratio_2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gyromagnetic_ratio_2")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.nmr_spin_system_toolkit_claude_64 import calculate_dipolar_coupling
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["distance", "gyromagnetic_ratio_1", "gyromagnetic_ratio_2", "theta"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = calculate_dipolar_coupling(**func_kwargs)
            
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


@Toolbox.register(name="generate_basis_states")
class GenerateBasisStatesTool(EnvironmentTool):
    """生成多自旋系统的基态"""
    
    name = "generate_basis_states"
    description = "生成多自旋系统的基态"
    arguments = {
        "num_spins": {"type": "integer", "description": "自旋核数量"},
        "spin_value": {"type": "number", "description": "每个核的自旋量子数，默认为0.5"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 generate_basis_states 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "num_spins" not in args or args["num_spins"] is None:
                return Observation(self.name, "错误: 缺少必需参数 num_spins")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.nmr_spin_system_toolkit_claude_64 import generate_basis_states
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["num_spins", "spin_value"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = generate_basis_states(**func_kwargs)
            
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


@Toolbox.register(name="analyze_two_spin_system")
class AnalyzeTwoSpinSystemTool(EnvironmentTool):
    """完整分析双自旋1/2系统"""
    
    name = "analyze_two_spin_system"
    description = "完整分析双自旋1/2系统"
    arguments = {
        "magnetic_field": {"type": "number", "description": "磁场强度 (T)"},
        "chemical_shift_1": {"type": "number", "description": "核1化学位移 (ppm)"},
        "chemical_shift_2": {"type": "number", "description": "核2化学位移 (ppm)"},
        "distance": {"type": "number", "description": "核间距离 (angstroms)"},
        "include_dipolar": {"type": "boolean", "description": "是否包含偶极耦合，默认为true"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_two_spin_system 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "magnetic_field" not in args or args["magnetic_field"] is None:
                return Observation(self.name, "错误: 缺少必需参数 magnetic_field")
            if "chemical_shift_1" not in args or args["chemical_shift_1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 chemical_shift_1")
            if "chemical_shift_2" not in args or args["chemical_shift_2"] is None:
                return Observation(self.name, "错误: 缺少必需参数 chemical_shift_2")
            if "distance" not in args or args["distance"] is None:
                return Observation(self.name, "错误: 缺少必需参数 distance")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.nmr_spin_system_toolkit_claude_64 import analyze_two_spin_system
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["magnetic_field", "chemical_shift_1", "chemical_shift_2", "distance", "include_dipolar"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_two_spin_system(**func_kwargs)
            
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


@Toolbox.register(name="visualize_spin_system_analysis")
class VisualizeSpinSystemAnalysisTool(EnvironmentTool):
    """可视化完整的自旋系统分析结果"""
    
    name = "visualize_spin_system_analysis"
    description = "可视化完整的自旋系统分析结果"
    arguments = {
        "analysis_filepath": {"type": "string", "description": "分析结果JSON文件路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_spin_system_analysis 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "analysis_filepath" not in args or args["analysis_filepath"] is None:
                return Observation(self.name, "错误: 缺少必需参数 analysis_filepath")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.nmr_spin_system_toolkit_claude_64 import visualize_spin_system_analysis
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["analysis_filepath"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_spin_system_analysis(**func_kwargs)
            
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


@Toolbox.register(name="solve_eigensystem")
class SolveEigensystemTool(EnvironmentTool):
    """求解哈密顿量的本征值和本征态"""
    
    name = "solve_eigensystem"
    description = "求解哈密顿量的本征值和本征态"
    arguments = {
        "hamiltonian_matrix": {"type": "array", "description": "2x2哈密顿量矩阵（list格式）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 solve_eigensystem 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "hamiltonian_matrix" not in args or args["hamiltonian_matrix"] is None:
                return Observation(self.name, "错误: 缺少必需参数 hamiltonian_matrix")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import solve_eigensystem
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["hamiltonian_matrix"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = solve_eigensystem(**func_kwargs)
            
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


@Toolbox.register(name="compute_eigenstate_derivative")
class ComputeEigenstateDerivativeTool(EnvironmentTool):
    """计算本征态对参数γ的导数（数值微分法）"""
    
    name = "compute_eigenstate_derivative"
    description = "计算本征态对参数γ的导数（数值微分法）"
    arguments = {
        "gamma": {"type": "number", "description": "参数γ值"},
        "delta_gamma": {"type": "number", "description": "微分步长，默认为1e-6"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_eigenstate_derivative 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gamma" not in args or args["gamma"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import compute_eigenstate_derivative
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gamma", "delta_gamma"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_eigenstate_derivative(**func_kwargs)
            
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


@Toolbox.register(name="compute_quantum_fisher_information")
class ComputeQuantumFisherInformationTool(EnvironmentTool):
    """计算单个本征态的量子Fisher信息"""
    
    name = "compute_quantum_fisher_information"
    description = "计算单个本征态的量子Fisher信息"
    arguments = {
        "eigenstate": {"type": "array", "description": "本征态 |ψ⟩"},
        "derivative_eigenstate": {"type": "array", "description": "导数态 |∂ψ/∂γ⟩"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_quantum_fisher_information 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "eigenstate" not in args or args["eigenstate"] is None:
                return Observation(self.name, "错误: 缺少必需参数 eigenstate")
            if "derivative_eigenstate" not in args or args["derivative_eigenstate"] is None:
                return Observation(self.name, "错误: 缺少必需参数 derivative_eigenstate")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import compute_quantum_fisher_information
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["eigenstate", "derivative_eigenstate"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_quantum_fisher_information(**func_kwargs)
            
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


@Toolbox.register(name="analytical_variance_formula")
class AnalyticalVarianceFormulaTool(EnvironmentTool):
    """使用解析公式计算方差上限"""
    
    name = "analytical_variance_formula"
    description = "使用解析公式计算方差上限"
    arguments = {
        "gamma": {"type": "number", "description": "参数γ"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analytical_variance_formula 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gamma" not in args or args["gamma"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import analytical_variance_formula
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gamma"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analytical_variance_formula(**func_kwargs)
            
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


@Toolbox.register(name="scan_variance_vs_gamma")
class ScanVarianceVsGammaTool(EnvironmentTool):
    """扫描不同γ值下的方差上限"""
    
    name = "scan_variance_vs_gamma"
    description = "扫描不同γ值下的方差上限"
    arguments = {
        "gamma_range": {"type": "array", "description": "γ值列表"},
        "method": {"type": "string", "description": "计算方法，'numerical' 或 'analytical'，默认为'numerical'"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 scan_variance_vs_gamma 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gamma_range" not in args or args["gamma_range"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma_range")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import scan_variance_vs_gamma
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gamma_range", "method"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = scan_variance_vs_gamma(**func_kwargs)
            
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


@Toolbox.register(name="plot_variance_vs_gamma")
class PlotVarianceVsGammaTool(EnvironmentTool):
    """绘制方差上限随γ变化的曲线"""
    
    name = "plot_variance_vs_gamma"
    description = "绘制方差上限随γ变化的曲线"
    arguments = {
        "gamma_values": {"type": "array", "description": "γ值列表"},
        "variance_limits": {"type": "array", "description": "对应的方差上限列表"},
        "title": {"type": "string", "description": "图表标题，默认为'Variance Upper Limit vs Parameter γ'"},
        "save_path": {"type": "string", "description": "保存路径（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_variance_vs_gamma 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gamma_values" not in args or args["gamma_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma_values")
            if "variance_limits" not in args or args["variance_limits"] is None:
                return Observation(self.name, "错误: 缺少必需参数 variance_limits")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import plot_variance_vs_gamma
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gamma_values", "variance_limits", "title", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_variance_vs_gamma(**func_kwargs)
            
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


@Toolbox.register(name="plot_quantum_fisher_info")
class PlotQuantumFisherInfoTool(EnvironmentTool):
    """绘制量子Fisher信息随γ变化的曲线"""
    
    name = "plot_quantum_fisher_info"
    description = "绘制量子Fisher信息随γ变化的曲线"
    arguments = {
        "gamma_values": {"type": "array", "description": "γ值列表"},
        "save_path": {"type": "string", "description": "保存路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_quantum_fisher_info 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gamma_values" not in args or args["gamma_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma_values")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import plot_quantum_fisher_info
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gamma_values", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_quantum_fisher_info(**func_kwargs)
            
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


@Toolbox.register(name="plot_eigenvalues_and_gap")
class PlotEigenvaluesAndGapTool(EnvironmentTool):
    """绘制本征值和能隙随γ变化的曲线"""
    
    name = "plot_eigenvalues_and_gap"
    description = "绘制本征值和能隙随γ变化的曲线"
    arguments = {
        "gamma_values": {"type": "array", "description": "γ值列表"},
        "save_path": {"type": "string", "description": "保存路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 plot_eigenvalues_and_gap 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "gamma_values" not in args or args["gamma_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 gamma_values")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_metrology_toolkit_claude_291 import plot_eigenvalues_and_gap
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["gamma_values", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = plot_eigenvalues_and_gap(**func_kwargs)
            
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


@Toolbox.register(name="validate_spin_state")
class ValidateSpinStateTool(EnvironmentTool):
    """验证向量是否满足n电子系统自旋态的所有条件"""
    
    name = "validate_spin_state"
    description = "验证向量是否满足n电子系统自旋态的所有条件"
    arguments = {
        "vector_data": {"type": "array", "description": "向量数据"},
        "vector_name": {"type": "string", "description": "向量名称（用于标识）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 validate_spin_state 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "vector_data" not in args or args["vector_data"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vector_data")
            if "vector_name" not in args or args["vector_name"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vector_name")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_toolkit_claude_139 import validate_spin_state
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["vector_data", "vector_name"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = validate_spin_state(**func_kwargs)
            
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


@Toolbox.register(name="analyze_vector_set")
class AnalyzeVectorSetTool(EnvironmentTool):
    """分析一组向量，判断哪些可以作为n电子系统的自旋态"""
    
    name = "analyze_vector_set"
    description = "分析一组向量，判断哪些可以作为n电子系统的自旋态"
    arguments = {
        "vectors_dict": {"type": "object", "description": "向量字典 {name: vector_data}，其中name为字符串，vector_data为向量数据数组"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 analyze_vector_set 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "vectors_dict" not in args or args["vectors_dict"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vectors_dict")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_toolkit_claude_139 import analyze_vector_set
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["vectors_dict"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = analyze_vector_set(**func_kwargs)
            
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


@Toolbox.register(name="compute_orthogonality_matrix")
class ComputeOrthogonalityMatrixTool(EnvironmentTool):
    """计算向量集合的正交性矩阵"""
    
    name = "compute_orthogonality_matrix"
    description = "计算向量集合的正交性矩阵"
    arguments = {
        "vectors_dict": {"type": "object", "description": "向量字典 {name: vector_data}，其中name为字符串，vector_data为向量数据数组"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_orthogonality_matrix 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "vectors_dict" not in args or args["vectors_dict"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vectors_dict")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_toolkit_claude_139 import compute_orthogonality_matrix
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["vectors_dict"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_orthogonality_matrix(**func_kwargs)
            
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


@Toolbox.register(name="visualize_validation_results")
class VisualizeValidationResultsTool(EnvironmentTool):
    """可视化自旋态验证结果"""
    
    name = "visualize_validation_results"
    description = "可视化自旋态验证结果"
    arguments = {
        "analysis_result": {"type": "object", "description": "analyze_vector_set的返回结果"},
        "save_path": {"type": "string", "description": "图像保存路径，默认为./tool_images/spin_state_validation.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_validation_results 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "analysis_result" not in args or args["analysis_result"] is None:
                return Observation(self.name, "错误: 缺少必需参数 analysis_result")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_toolkit_claude_139 import visualize_validation_results
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["analysis_result", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_validation_results(**func_kwargs)
            
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


@Toolbox.register(name="create_dimension_analysis_plot")
class CreateDimensionAnalysisPlotTool(EnvironmentTool):
    """创建维度分析图，展示各向量维度与n电子系统的关系"""
    
    name = "create_dimension_analysis_plot"
    description = "创建维度分析图，展示各向量维度与n电子系统的关系"
    arguments = {
        "vectors_dict": {"type": "object", "description": "向量字典 {name: vector_data}"},
        "save_path": {"type": "string", "description": "图像保存路径，默认为./tool_images/dimension_analysis.png"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 create_dimension_analysis_plot 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "vectors_dict" not in args or args["vectors_dict"] is None:
                return Observation(self.name, "错误: 缺少必需参数 vectors_dict")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_toolkit_claude_139 import create_dimension_analysis_plot
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["vectors_dict", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = create_dimension_analysis_plot(**func_kwargs)
            
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


@Toolbox.register(name="get_spin_operators")
class GetSpinOperatorsTool(EnvironmentTool):
    """获取自旋算符 S = (ℏ/2)σ"""
    
    name = "get_spin_operators"
    description = "获取自旋算符 S = (ℏ/2)σ"
    arguments = {
        "hbar": {"type": "number", "description": "约化普朗克常数，默认为1.0（自然单位制）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 get_spin_operators 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            pass
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import get_spin_operators
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["hbar"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = get_spin_operators(**func_kwargs)
            
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


@Toolbox.register(name="compute_time_series")
class ComputeTimeSeriesTool(EnvironmentTool):
    """计算可观测量随时间的演化序列"""
    
    name = "compute_time_series"
    description = "计算可观测量随时间的演化序列"
    arguments = {
        "initial_state_list": {"type": "array", "description": "初态（2x1列表）"},
        "hamiltonian_list": {"type": "array", "description": "哈密顿量（2x2列表）"},
        "time_points": {"type": "array", "description": "时间点列表"},
        "observable_list": {"type": "array", "description": "可观测量算符（2x2列表）"},
        "hbar": {"type": "number", "description": "约化普朗克常数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 compute_time_series 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "initial_state_list" not in args or args["initial_state_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_state_list")
            if "hamiltonian_list" not in args or args["hamiltonian_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 hamiltonian_list")
            if "time_points" not in args or args["time_points"] is None:
                return Observation(self.name, "错误: 缺少必需参数 time_points")
            if "observable_list" not in args or args["observable_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 observable_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import compute_time_series
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["initial_state_list", "hamiltonian_list", "time_points", "observable_list", "hbar"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = compute_time_series(**func_kwargs)
            
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


@Toolbox.register(name="extract_oscillation_frequency")
class ExtractOscillationFrequencyTool(EnvironmentTool):
    """从时间序列中提取振荡频率"""
    
    name = "extract_oscillation_frequency"
    description = "从时间序列中提取振荡频率"
    arguments = {
        "times": {"type": "array", "description": "时间点列表"},
        "values": {"type": "array", "description": "对应的数值列表"},
        "method": {"type": "string", "description": "提取方法 ('fft' 或 'fit')"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 extract_oscillation_frequency 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "times" not in args or args["times"] is None:
                return Observation(self.name, "错误: 缺少必需参数 times")
            if "values" not in args or args["values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 values")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import extract_oscillation_frequency
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["times", "values", "method"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = extract_oscillation_frequency(**func_kwargs)
            
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


@Toolbox.register(name="visualize_spin_precession")
class VisualizeSpinPrecessionTool(EnvironmentTool):
    """可视化自旋进动"""
    
    name = "visualize_spin_precession"
    description = "可视化自旋进动"
    arguments = {
        "times": {"type": "array", "description": "时间点列表"},
        "expectation_values": {"type": "array", "description": "期望值列表"},
        "frequency": {"type": "number", "description": "振荡频率"},
        "component": {"type": "string", "description": "自旋分量 ('x', 'y', 'z')"},
        "save_path": {"type": "string", "description": "图像保存路径（可选）"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_spin_precession 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "times" not in args or args["times"] is None:
                return Observation(self.name, "错误: 缺少必需参数 times")
            if "expectation_values" not in args or args["expectation_values"] is None:
                return Observation(self.name, "错误: 缺少必需参数 expectation_values")
            if "frequency" not in args or args["frequency"] is None:
                return Observation(self.name, "错误: 缺少必需参数 frequency")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import visualize_spin_precession
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["times", "expectation_values", "frequency", "component", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_spin_precession(**func_kwargs)
            
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


@Toolbox.register(name="visualize_bloch_sphere_trajectory")
class VisualizeBlochSphereTrajectoryTool(EnvironmentTool):
    """在Bloch球面上可视化自旋轨迹"""
    
    name = "visualize_bloch_sphere_trajectory"
    description = "在Bloch球面上可视化自旋轨迹"
    arguments = {
        "times": {"type": "array", "description": "时间点列表"},
        "initial_state_list": {"type": "array", "description": "初态"},
        "hamiltonian_list": {"type": "array", "description": "哈密顿量"},
        "save_path": {"type": "string", "description": "保存路径"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visualize_bloch_sphere_trajectory 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "times" not in args or args["times"] is None:
                return Observation(self.name, "错误: 缺少必需参数 times")
            if "initial_state_list" not in args or args["initial_state_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 initial_state_list")
            if "hamiltonian_list" not in args or args["hamiltonian_list"] is None:
                return Observation(self.name, "错误: 缺少必需参数 hamiltonian_list")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.quantum_spin_dynamics_toolkit_claude_131 import visualize_bloch_sphere_trajectory
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["times", "initial_state_list", "hamiltonian_list", "save_path"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visualize_bloch_sphere_trajectory(**func_kwargs)
            
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


@Toolbox.register(name="math_ionization_energy_relations")
class MathIonizationEnergyRelationsTool(EnvironmentTool):
    """计算电离能相关参数。基于第一电离能计算氢原子类系统的各种量子化学参数，包括第二电离能、总能量、电子间排斥能、有效核电荷和屏蔽常数。适用于氢原子类离子体系分析。"""
    
    name = "math_ionization_energy_relations"
    description = "计算电离能相关参数。基于第一电离能计算氢原子类系统的各种量子化学参数，包括第二电离能、总能量、电子间排斥能、有效核电荷和屏蔽常数。适用于氢原子类离子体系分析。"
    arguments = {
        "I1": {"type": "number", "description": "第一电离能，单位为eV。从基态原子中移去一个电子所需的能量"},
        "Z": {"type": "integer", "description": "原子序数，默认为2。核电荷数，决定氢原子类系统的性质"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 math_ionization_energy_relations 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "I1" not in args or args["I1"] is None:
                return Observation(self.name, "错误: 缺少必需参数 I1")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.math_ionization_energy_relations import math_ionization_energy_relations
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["I1", "Z"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = math_ionization_energy_relations(**func_kwargs)
            
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


@Toolbox.register(name="context_ionization_energy_relations")
class ContextIonizationEnergyRelationsTool(EnvironmentTool):
    """显示电离能分析结果。格式化输出电离能分析的计算结果，包括第一电离能、第二电离能、总能量、电子间排斥能、有效核电荷和屏蔽常数等量子化学参数。"""
    
    name = "context_ionization_energy_relations"
    description = "显示电离能分析结果。格式化输出电离能分析的计算结果，包括第一电离能、第二电离能、总能量、电子间排斥能、有效核电荷和屏蔽常数等量子化学参数。"
    arguments = {
        "results": {"type": "object", "description": "电离能分析结果字典，包含I1、I2、E_total、V_ee、Z_eff、sigma等参数"},
        "Z": {"type": "integer", "description": "原子序数，默认为2。用于显示分析的元素信息"},
        "verbose": {"type": "boolean", "description": "是否显示详细输出，默认为True。控制是否打印分析结果到控制台"},
        "precision": {"type": "integer", "description": "数值显示精度，默认为4位小数。控制输出数值的小数位数"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 context_ionization_energy_relations 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "results" not in args or args["results"] is None:
                return Observation(self.name, "错误: 缺少必需参数 results")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.math_ionization_energy_relations import context_ionization_energy_relations
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["results", "Z", "verbose", "precision"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = context_ionization_energy_relations(**func_kwargs)
            
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


@Toolbox.register(name="visual_func")
class VisualFuncTool(EnvironmentTool):
    """生成电离能分析的可视化图表。创建包含两个子图的综合分析图表：1. 能级图显示原子的能量状态和电离能；2. 屏蔽效应示意图展示核电荷屏蔽和有效核电荷的概念。"""
    
    name = "visual_func"
    description = "生成电离能分析的可视化图表。创建包含两个子图的综合分析图表：1. 能级图显示原子的能量状态和电离能；2. 屏蔽效应示意图展示核电荷屏蔽和有效核电荷的概念。"
    arguments = {
        "results": {"type": "object", "description": "电离能分析结果字典，包含I1、I2、E_total、Z、Z_eff、sigma等参数"},
        "save_path": {"type": "string", "description": "图表保存路径，默认为None。如果提供路径，图表将保存为文件"},
        "figsize": {"type": "array", "description": "图表尺寸，默认为(14, 6)。控制图表的宽度和高度"},
        "dpi": {"type": "integer", "description": "图像分辨率，默认为150。控制保存图像的质量"}
    }
    
    def use(self, environment, action) -> Observation:
        """执行 visual_func 操作"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            if "results" not in args or args["results"] is None:
                return Observation(self.name, "错误: 缺少必需参数 results")
            
            # 导入并调用原始函数
            from toolkits.physics.atomic_and_molecular_physics.math_ionization_energy_relations import visual_func
            
            # 构建参数字典，只传递非 None 的值
            func_kwargs = {}
            param_names = ["results", "save_path", "figsize", "dpi"]
            for param_name in param_names:
                param_val = args.get(param_name)
                if param_val is not None:
                    func_kwargs[param_name] = param_val
            
            # 调用函数
            result = visual_func(**func_kwargs)
            
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

