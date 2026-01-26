#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分配系数计算器函数暴露模块
将 PartitionCoefficientCalculator 类的主要方法以顶层函数形式暴露，便于直接调用
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np

# 统一从当前文件所在目录导入 PartitionCoefficientCalculator
# 这样无论是被作为脚本直接运行，还是通过 importlib 从仓库根目录加载，都可以工作
try:
    from PartitionCoefficientCalculator import PartitionCoefficientCalculator  # 同目录直导入
except ImportError:
    import os
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    if CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from PartitionCoefficientCalculator import PartitionCoefficientCalculator

# 创建一个默认的计算器实例（单例）
_default_calculator: Union[PartitionCoefficientCalculator, None] = None


def _get_calculator(temperature: float = 298.15) -> PartitionCoefficientCalculator:
    """获取或创建分配系数计算器实例"""
    global _default_calculator
    if _default_calculator is None or _default_calculator.temperature != temperature:
        _default_calculator = PartitionCoefficientCalculator(temperature=temperature)
    return _default_calculator


# ========================
# 1. 电离与logP/分配系数计算
# ========================

def calculate_ionization_fraction(pH: float, pKa: Union[float, List[float], Tuple[float, ...]],
                                  temperature: float = 298.15) -> Union[float, List[float]]:
    """
    计算给定 pH 条件下的电离度

    Args:
        pH: 水相 pH 值
        pKa: 酸性基团的 pKa，或由多个 pKa 组成的列表/元组
        temperature: 温度(K)

    Returns:
        单一基团返回 float；多基团返回同长度的 List[float]
    """
    calculator = _get_calculator(temperature)
    return calculator.calculate_ionization_fraction(pH, pKa)


def calculate_apparent_logP(logP_neutral: float, pH: float, pKa: float, *, acidic: bool = True,
                            temperature: float = 298.15) -> float:
    """
    计算考虑电离的表观 logP

    Args:
        logP_neutral: 中性分子的 logP
        pH: 水相 pH
        pKa: 酸性(或碱性)基团的 pKa
        acidic: 是否为酸性基团（True 表示酸性，False 表示碱性）
        temperature: 温度(K)

    Returns:
        表观 logP 值
    """
    calculator = _get_calculator(temperature)
    return calculator.calculate_apparent_logP(logP_neutral, pH, pKa, acidic=acidic)


def calculate_distribution_coefficient(logP_app: float, temperature: float = 298.15) -> float:
    """
    由表观 logP 计算分配系数 K

    Args:
        logP_app: 表观 logP
        temperature: 温度(K)，此处仅为接口统一保留

    Returns:
        分配系数 K = 10**logP_app
    """
    calculator = _get_calculator(temperature)
    return calculator.calculate_distribution_coefficient(logP_app)


def predict_logP_from_structure(smiles: str, temperature: float = 298.15) -> float:
    """
    基于 RDKit 从 SMILES 预测中性 logP（Crippen logP）
    """
    calculator = _get_calculator(temperature)
    return calculator.predict_logP_from_structure(smiles)


# ========================
# 2. 机制分析与可视化
# ========================

def analyze_partition_mechanism(compound_name: str, smiles: str, logP_app: float,
                                pH: float, pKa: float, temperature: float = 298.15) -> Dict[str, Any]:
    """
    分析在两相体系中的分配偏好与分子机制
    """
    calculator = _get_calculator(temperature)
    return calculator.analyze_partition_mechanism(compound_name, smiles, logP_app, pH, pKa)


def visualize_pH_logP_profile(logP_neutral: float, pKa: float,
                              pH_range: Tuple[float, float] = (0.0, 14.0), *, acidic: bool = True,
                              compound_name: str = "", temperature: float = 298.15):
    """
    生成 pH-logP 曲线的 Matplotlib Figure（不直接展示）
    """
    calculator = _get_calculator(temperature)
    return calculator.visualize_pH_logP_profile(logP_neutral, pKa, pH_range=pH_range,
                                                acidic=acidic, compound_name=compound_name)


# ========================
# 3. 便捷求解入口
# ========================

def solve_partition_problem(compound: str, smiles: str, logP_neutral: float,
                            pKa: float, pH: float, *, acidic: bool = True,
                            temperature: float = 298.15) -> Dict[str, Any]:
    """
    综合求解单一酸/碱基团体系的分配问题。
    
    设计目标：
    - 当上游传入的 logP_neutral / pKa / pH 不完整或类型错误时，不抛异常；
      而是返回结构化的 issues 与提示，引导 LLM 先调用其他工具补全参数。
    
    正常返回:
        {
          'ok': True,
          'compound': str,
          'logP_app': float,
          'K_ow': float,
          'analysis': Dict[str, Any]
        }
    失败/缺参数时返回:
        {
          'ok': False,
          'compound': str,
          'smiles': str,
          'logP_neutral': ...,
          'pKa': ...,
          'pH': ...,
          'issues': [...],
          'message': '...'
        }
    """
    issues = []

    # 基础输入验证（避免在 calculate_apparent_logP 中崩溃）
    if not isinstance(logP_neutral, (int, float)):
        issues.append("logP_neutral_missing_or_invalid")
    if not isinstance(pKa, (int, float)):
        issues.append("pKa_missing_or_invalid")
    if not isinstance(pH, (int, float)):
        issues.append("pH_missing_or_invalid")

    if issues:
        return {
            "ok": False,
            "compound": compound,
            "smiles": smiles,
            "logP_neutral": logP_neutral,
            "pKa": pKa,
            "pH": pH,
            "issues": issues,
            "message": (
                "solve_partition_problem 需要数值类型的 logP_neutral、pKa 和 pH 才能进行分配系数与机理分析；"
                "请先使用其他工具（如 predict_logP_from_structure / 文献或题目已给的 pKa）补全这些字段后再调用。"
            ),
        }

    # 参数合法时按原流程计算
    logP_app = calculate_apparent_logP(logP_neutral, pH, pKa, acidic=acidic, temperature=temperature)
    K_ow = calculate_distribution_coefficient(logP_app, temperature=temperature)
    analysis = analyze_partition_mechanism(compound, smiles, logP_app, pH, pKa, temperature=temperature)

    return {
        "ok": True,
        "compound": compound,
        "logP_app": logP_app,
        "K_ow": K_ow,
        "analysis": analysis,
    }
