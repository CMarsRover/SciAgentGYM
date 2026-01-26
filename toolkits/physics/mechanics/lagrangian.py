#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拉格朗日函数类
"""

import numpy as np
import sympy as sp
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass

@dataclass
class Lagrangian:
    """
    拉格朗日函数类
    
    用于表示和分析拉格朗日函数
    """
    
    expression: str
    variables: Optional[List[str]] = None
    parameters: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.variables is None:
            self.variables = ['t', 'x', 'x_dot']
        if self.parameters is None:
            self.parameters = {}
        
        self.symbols = {}
        self._setup_symbols()
        self._parse_expression()
    
    def _setup_symbols(self):
        """设置符号变量"""
        for var in self.variables:
            self.symbols[var] = sp.Symbol(var)
        
        # 添加参数符号
        for param, value in self.parameters.items():
            self.symbols[param] = sp.Symbol(param)
    
    def _parse_expression(self):
        """解析表达式"""
        try:
            self.sympy_expr = sp.sympify(self.expression)
        except Exception as e:
            raise ValueError(f"无法解析拉格朗日函数表达式: {e}")
    
    def set_parameters(self, parameters: Dict[str, float]):
        """设置参数值"""
        self.parameters.update(parameters)
        self._setup_symbols()
        self._parse_expression()
    
    def get_kinetic_energy(self) -> sp.Expr:
        """提取动能项"""
        # 假设动能项包含速度的平方项
        kinetic_terms = []
        for term in sp.expand(self.sympy_expr).args:
            if hasattr(term, 'free_symbols') and sp.Symbol('x_dot') in term.free_symbols:
                kinetic_terms.append(term)
        
        return sp.Add(*kinetic_terms) if kinetic_terms else sp.Integer(0)
    
    def get_potential_energy(self) -> sp.Expr:
        """提取势能项"""
        # 假设势能项不包含速度
        potential_terms = []
        for term in sp.expand(self.sympy_expr).args:
            if hasattr(term, 'free_symbols') and sp.Symbol('x_dot') not in term.free_symbols:
                potential_terms.append(term)
        
        return sp.Add(*potential_terms) if potential_terms else sp.Integer(0)
    
    def compute_partial_derivatives(self) -> Dict[str, sp.Expr]:
        """计算偏导数"""
        derivatives = {}
        
        for var in self.variables:
            if var in self.symbols:
                derivatives[f'dL_d{var}'] = sp.diff(self.sympy_expr, self.symbols[var])
        
        return derivatives
    
    def compute_euler_lagrange_equation(self) -> sp.Expr:
        """计算欧拉-拉格朗日方程"""
        t = self.symbols.get('t', sp.Symbol('t'))
        x = self.symbols.get('x', sp.Symbol('x'))
        x_dot = self.symbols.get('x_dot', sp.Symbol('x_dot'))
        
        # 计算偏导数
        dL_dx = sp.diff(self.sympy_expr, x)
        dL_dxdot = sp.diff(self.sympy_expr, x_dot)
        
        # 计算时间导数
        dL_dxdot_dt = sp.diff(dL_dxdot, t)
        
        # 欧拉-拉格朗日方程
        return dL_dxdot_dt - dL_dx
    
    def to_numerical_function(self) -> Callable:
        """转换为数值函数"""
        # 创建数值函数
        def lagrangian_func(t, x, x_dot, **params):
            # 合并参数
            all_params = {**self.parameters, **params}
            
            # 替换符号为数值
            expr = self.sympy_expr
            for param, value in all_params.items():
                expr = expr.subs(sp.Symbol(param), value)
            
            # 转换为数值函数
            func = sp.lambdify([sp.Symbol('t'), sp.Symbol('x'), sp.Symbol('x_dot')], expr)
            return func(t, x, x_dot)
        
        return lagrangian_func
    
    def evaluate(self, 
                t: float, 
                x: float, 
                x_dot: float, 
                **params) -> float:
        """计算拉格朗日函数值"""
        func = self.to_numerical_function()
        return func(t, x, x_dot, **params)
    
    def get_conserved_quantities(self) -> Dict[str, sp.Expr]:
        """计算守恒量"""
        conserved = {}
        
        # 能量守恒（如果拉格朗日函数不显含时间）
        if sp.Symbol('t') not in self.sympy_expr.free_symbols:
            x_dot = self.symbols.get('x_dot', sp.Symbol('x_dot'))
            dL_dxdot = sp.diff(self.sympy_expr, x_dot)
            energy = x_dot * dL_dxdot - self.sympy_expr
            conserved['energy'] = energy
        
        # 动量守恒（如果拉格朗日函数不显含位置）
        if sp.Symbol('x') not in self.sympy_expr.free_symbols:
            x_dot = self.symbols.get('x_dot', sp.Symbol('x_dot'))
            momentum = sp.diff(self.sympy_expr, x_dot)
            conserved['momentum'] = momentum
        
        return conserved
    
    def is_autonomous(self) -> bool:
        """检查是否为自治系统（不显含时间）"""
        return sp.Symbol('t') not in self.sympy_expr.free_symbols
    
    def is_translation_invariant(self) -> bool:
        """检查是否具有平移不变性（不显含位置）"""
        return sp.Symbol('x') not in self.sympy_expr.free_symbols
    
    def get_info(self) -> Dict[str, Any]:
        """获取拉格朗日函数信息"""
        return {
            'expression': self.expression,
            'variables': self.variables,
            'parameters': self.parameters,
            'kinetic_energy': str(self.get_kinetic_energy()),
            'potential_energy': str(self.get_potential_energy()),
            'euler_lagrange_equation': str(self.compute_euler_lagrange_equation()),
            'conserved_quantities': {k: str(v) for k, v in self.get_conserved_quantities().items()},
            'is_autonomous': self.is_autonomous(),
            'is_translation_invariant': self.is_translation_invariant()
        }
    
    def __str__(self) -> str:
        return f"Lagrangian({self.expression})"
    
    def __repr__(self) -> str:
        return self.__str__()


class CommonLagrangians:
    """常用拉格朗日函数集合"""
    
    @staticmethod
    def harmonic_oscillator(m: float = 1.0, k: float = 1.0) -> Lagrangian:
        """谐振子拉格朗日函数"""
        return Lagrangian(
            expression="0.5 * m * x_dot^2 - 0.5 * k * x^2",
            parameters={'m': m, 'k': k}
        )
    
    @staticmethod
    def pendulum(l: float = 1.0, g: float = 9.81) -> Lagrangian:
        """单摆拉格朗日函数"""
        return Lagrangian(
            expression="0.5 * l^2 * theta_dot^2 + g * l * cos(theta)",
            variables=['t', 'theta', 'theta_dot'],
            parameters={'l': l, 'g': g}
        )
    
    @staticmethod
    def free_particle(m: float = 1.0) -> Lagrangian:
        """自由粒子拉格朗日函数"""
        return Lagrangian(
            expression="0.5 * m * x_dot^2",
            parameters={'m': m}
        )
    
    @staticmethod
    def charged_particle_in_field(m: float = 1.0, q: float = 1.0, E: float = 1.0) -> Lagrangian:
        """电场中带电粒子拉格朗日函数"""
        return Lagrangian(
            expression="0.5 * m * x_dot^2 + q * E * x",
            parameters={'m': m, 'q': q, 'E': E}
        )
