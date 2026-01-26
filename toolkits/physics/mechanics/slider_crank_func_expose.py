#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滑块-曲柄机构函数暴露模块
参考 thin_film_func_expose.py 的风格，将 SliderCrankMechanism 类的主要能力以顶层函数暴露。
"""

from typing import Dict, Any, Tuple
import numpy as np

try:
    # 脚本直接运行（同目录）
    from constrained_systems import SliderCrankMechanism
except ImportError:
    try:
        # 包内相对导入（包方式使用）
        from .constrained_systems import SliderCrankMechanism
    except ImportError:
        # 绝对导入（从项目根目录）
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from tools.variational_calculus.mechanics.constrained_systems import SliderCrankMechanism

_default_system = None


def _get_system(l1: float = 1.0, l2: float = 2.0,
                m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> SliderCrankMechanism:
    """获取或创建滑块-曲柄机构系统实例（单例，参数变化会重建）。"""
    global _default_system
    if (
        _default_system is None or
        _default_system.l1 != l1 or _default_system.l2 != l2 or
        _default_system.m1 != m1 or _default_system.m2 != m2 or _default_system.m3 != m3
    ):
        _default_system = SliderCrankMechanism(l1=l1, l2=l2, m1=m1, m2=m2, m3=m3)
    return _default_system


# ========================
# 1) 元信息与矩阵
# ========================

def get_physical_interpretation(l1: float = 1.0, l2: float = 2.0,
                                m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> Dict[str, str]:
    system = _get_system(l1, l2, m1, m2, m3)
    return system.get_physical_interpretation()


def get_lagrangian(l1: float = 1.0, l2: float = 2.0,
                   m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> str:
    system = _get_system(l1, l2, m1, m2, m3)
    return system.get_lagrangian()


def get_constraints(l1: float = 1.0, l2: float = 2.0,
                    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> list:
    system = _get_system(l1, l2, m1, m2, m3)
    return system.get_constraints()


def get_mass_matrix(l1: float = 1.0, l2: float = 2.0,
                    m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> np.ndarray:
    system = _get_system(l1, l2, m1, m2, m3)
    return system.get_mass_matrix()


def get_constraint_jacobian(l1: float = 1.0, l2: float = 2.0,
                            m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> np.ndarray:
    system = _get_system(l1, l2, m1, m2, m3)
    return system.get_constraint_jacobian()


def solve_dae(l1: float = 1.0, l2: float = 2.0,
              m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> Dict[str, Any]:
    system = _get_system(l1, l2, m1, m2, m3)
    return system.solve_dae()


# ========================
# 2) 运动求解
# ========================

def solve_motion(initial_conditions: Dict[str, float],
                 t_span: Tuple[float, float] = (0.0, 10.0),
                 num_points: int = 100,
                 l1: float = 1.0, l2: float = 2.0,
                 m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> Dict[str, Any]:
    """便捷求解入口：返回包含时间、广义坐标、速度、拉格朗日乘子的解。"""
    system = _get_system(l1, l2, m1, m2, m3)
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    return system.solve_motion(initial_conditions, t_span=t_span, t_eval=t_eval)


def solve_complete(initial_conditions: Dict[str, float],
                   t_span: Tuple[float, float] = (0.0, 10.0),
                   num_points: int = 100,
                   l1: float = 1.0, l2: float = 2.0,
                   m1: float = 1.0, m2: float = 1.0, m3: float = 1.0) -> Dict[str, Any]:
    """综合求解：返回 DAE 结构、运动解与物理解释。"""
    dae = solve_dae(l1, l2, m1, m2, m3)
    motion = solve_motion(initial_conditions, t_span, num_points, l1, l2, m1, m2, m3)
    info = get_physical_interpretation(l1, l2, m1, m2, m3)
    return {
        'dae': dae,
        'motion': motion,
        'info': info,
    }


