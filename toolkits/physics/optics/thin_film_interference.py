#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版薄膜干涉计算器
结合两个文件的优点，提供完整的薄膜干涉分析功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import fsolve
import sympy as sp
from typing import Tuple, Optional, Dict, List, Union
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedThinFilmInterferenceCalculator:
    """
    增强版薄膜干涉计算器
    
    功能特点：
    1. 支持垂直入射和斜入射
    2. 精确的半波损失计算
    3. 完整的干涉分析（相长/相消/部分）
    4. 多种可视化功能
    5. 颜色预测和光谱分析
    6. 符号计算和数值优化
    7. 完善的错误处理
    """
    
    def __init__(self, n_air: float = 1.0, n_film: float = 1.5, n_substrate: float = 1.0):
        """
        初始化薄膜干涉计算器
        
        Args:
            n_air: 入射介质折射率（通常为空气=1.0）
            n_film: 薄膜折射率
            n_substrate: 基底折射率
        """
        self.n_air = n_air
        self.n_film = n_film
        self.n_substrate = n_substrate
        
        # 验证物理合理性
        if n_film <= 0 or n_substrate <= 0 or n_air <= 0:
            raise ValueError("折射率必须为正数")
    
    # ========================
    # 1. 核心物理计算模块
    # ========================
    
    def _has_half_wave_loss_reflection(self, n_i: float, n_t: float) -> bool:
        """
        判断光从介质i到介质t反射时是否有半波损失
        
        Args:
            n_i: 入射介质折射率
            n_t: 透射介质折射率
            
        Returns:
            bool: 是否有半波损失（仅当n_i < n_t时发生）
        """
        return n_i < n_t
    
    def _calculate_refraction_angle(self, theta: float) -> float:
        """
        计算折射角
        
        Args:
            theta: 入射角（弧度）
            
        Returns:
            float: 折射角（弧度）
        """
        if np.sin(theta) >= self.n_air / self.n_film:
            raise ValueError("发生全反射，无透射角")
        return np.arcsin(self.n_air / self.n_film * np.sin(theta))
    
    def optical_path_difference(self, d: float, theta: float, wavelength: float) -> float:
        """
        计算光程差（支持斜入射）
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            wavelength: 波长
            
        Returns:
            float: 光程差
        """
        try:
            theta_t = self._calculate_refraction_angle(theta)
            opd = 2 * self.n_film * d * np.cos(theta_t)
            
            # 半波损失：仅当n_film > n_substrate时，净增λ/2
            if self.n_film > self.n_substrate:
                opd += wavelength / 2
                
            return opd
        except ValueError:
            return np.nan
    
    def interference_condition(self, d: float, theta: float, wavelength: float) -> Dict[str, Union[str, float]]:
        """
        计算干涉条件（增强版）
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            wavelength: 波长
            
        Returns:
            dict: 包含光程差、相位差和干涉类型的字典
        """
        opd = self.optical_path_difference(d, theta, wavelength)
        
        if np.isnan(opd):
            return {
                "opd": np.nan,
                "phase_diff": np.nan,
                "interference": "total_reflection"
            }
        
        # 计算相位差
        phase_diff = 2 * np.pi * opd / wavelength
        
        # 判断干涉类型
        if abs(opd % wavelength) < 1e-10:
            interference = "constructive"
        elif abs((opd - wavelength / 2) % wavelength - wavelength / 2) < 1e-10:
            interference = "destructive"
        else:
            interference = "partial"
        
        return {
            "opd": opd,
            "phase_diff": phase_diff,
            "interference": interference
        }
    
    # ========================
    # 2. 厚度计算模块
    # ========================
    
    def min_thickness_for_constructive_reflection(self, wavelength: float, theta: float = 0) -> float:
        """
        计算反射光相长干涉的最小厚度（考虑半波损失）
        
        Args:
            wavelength: 波长
            theta: 入射角（弧度，默认0为垂直入射）
            
        Returns:
            float: 最小厚度
        """
        # 判断净半波损失
        upper_phase_change = self._has_half_wave_loss_reflection(self.n_air, self.n_film)
        lower_phase_change = self._has_half_wave_loss_reflection(self.n_film, self.n_substrate)
        net_half_wave_loss = upper_phase_change != lower_phase_change
        
        if theta == 0:  # 垂直入射
            if net_half_wave_loss:
                d_min = (0.5 * wavelength) / (2 * self.n_film)
            else:
                d_min = (1 * wavelength) / (2 * self.n_film)
        else:  # 斜入射
            try:
                theta_t = self._calculate_refraction_angle(theta)
                if net_half_wave_loss:
                    d_min = (0.5 * wavelength) / (2 * self.n_film * np.cos(theta_t))
                else:
                    d_min = (1 * wavelength) / (2 * self.n_film * np.cos(theta_t))
            except ValueError:
                raise ValueError("斜入射角度过大，发生全反射")
        
        return d_min
    
    def min_thickness_for_constructive_transmission(self, wavelength: float, theta: float = 0) -> float:
        """
        计算透射光相长干涉的最小厚度
        
        Args:
            wavelength: 波长
            theta: 入射角（弧度，默认0为垂直入射）
            
        Returns:
            float: 最小厚度
        """
        # 透射光相长条件与反射光相消条件相同
        upper_phase_change = self._has_half_wave_loss_reflection(self.n_air, self.n_film)
        lower_phase_change = self._has_half_wave_loss_reflection(self.n_film, self.n_substrate)
        net_half_wave_loss = upper_phase_change != lower_phase_change
        
        if theta == 0:  # 垂直入射
            if net_half_wave_loss:
                d_min = (1 * wavelength) / (2 * self.n_film)
            else:
                d_min = (0.5 * wavelength) / (2 * self.n_film)
        else:  # 斜入射
            try:
                theta_t = self._calculate_refraction_angle(theta)
                if net_half_wave_loss:
                    d_min = (1 * wavelength) / (2 * self.n_film * np.cos(theta_t))
                else:
                    d_min = (0.5 * wavelength) / (2 * self.n_film * np.cos(theta_t))
            except ValueError:
                raise ValueError("斜入射角度过大，发生全反射")
        
        return d_min
    
    def min_antireflection_thickness(self, wavelength: float) -> float:
        """
        计算最小增透厚度（λ/4n）
        
        Args:
            wavelength: 波长
            
        Returns:
            float: 最小增透厚度
        """
        return wavelength / (4 * self.n_film)
    
    def wavelength_for_constructive(self, d: float, m: int = 1, theta: float = 0) -> float:
        """
        计算相长干涉的波长
        
        Args:
            d: 薄膜厚度
            m: 干涉级次
            theta: 入射角（弧度）
            
        Returns:
            float: 相长干涉波长
        """
        if theta == 0:
            return 2 * self.n_film * d / m
        else:
            try:
                theta_t = self._calculate_refraction_angle(theta)
                return 2 * self.n_film * d * np.cos(theta_t) / m
            except ValueError:
                raise ValueError("斜入射角度过大，发生全反射")
    
    # ========================
    # 3. 干涉判断模块
    # ========================
    
    def is_constructive(self, d: float, theta: float, wavelength: float) -> bool:
        """
        判断是否为相长干涉
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            wavelength: 波长
            
        Returns:
            bool: 是否为相长干涉
        """
        opd = self.optical_path_difference(d, theta, wavelength)
        if np.isnan(opd):
            return False
        return abs((opd % wavelength) - 0) < 1e-5 or abs((opd % wavelength) - wavelength) < 1e-5
    
    def is_destructive(self, d: float, theta: float, wavelength: float) -> bool:
        """
        判断是否为相消干涉
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            wavelength: 波长
            
        Returns:
            bool: 是否为相消干涉
        """
        opd = self.optical_path_difference(d, theta, wavelength)
        if np.isnan(opd):
            return False
        return abs((opd % wavelength) - wavelength / 2) < 1e-5
    
    # ========================
    # 4. 光谱分析模块
    # ========================
    
    def reflectance_spectrum(self, d: float, theta: float, wavelengths: np.ndarray) -> np.ndarray:
        """
        计算反射光谱
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            wavelengths: 波长数组
            
        Returns:
            np.ndarray: 反射率数组
        """
        R_base = 0.04  # 假设基底反射率
        R = []
        
        for wl in wavelengths:
            opd = self.optical_path_difference(d, theta, wl)
            if np.isnan(opd):
                R.append(1.0)  # 全反射
                continue
            
            delta = 2 * np.pi * opd / wl
            # 干涉项：R_total = R1 + R2 + 2√(R1 R2) cos(δ)
            interference_term = 2 * np.sqrt(R_base * 0.04) * np.cos(delta)
            R_total = R_base + 0.04 + interference_term
            R_total = np.clip(R_total, 0, 1)
            R.append(R_total)
        
        return np.array(R)
    
    def predict_color(self, d: float, theta: float = 0) -> Dict[str, Union[float, List[int], np.ndarray]]:
        """
        预测薄膜颜色
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            
        Returns:
            dict: 包含峰值波长、RGB值和反射光谱的字典
        """
        wavelengths = np.linspace(400e-9, 700e-9, 100)
        R = self.reflectance_spectrum(d, theta, wavelengths)
        
        # 找到最大响应波长
        peak_idx = np.argmax(R)
        peak_wl = wavelengths[peak_idx] * 1e9  # 转换为nm
        
        # 转换为RGB
        rgb = self._wavelength_to_rgb(peak_wl)
        
        return {
            "peak_wavelength_nm": peak_wl,
            "rgb": rgb,
            "reflectance": R
        }
    
    def _wavelength_to_rgb(self, wavelength: float) -> List[int]:
        """
        将波长转换为RGB值
        
        Args:
            wavelength: 波长（nm）
            
        Returns:
            List[int]: RGB值列表
        """
        gamma = 0.8
        
        def adjust(color):
            return int(255 * (color ** gamma))
        
        if 380 <= wavelength < 440:
            r, g, b = -(wavelength - 440) / (440 - 380), 0, 1
        elif 440 <= wavelength < 490:
            r, g, b = 0, (wavelength - 440) / (490 - 440), 1
        elif 490 <= wavelength < 510:
            r, g, b = 0, 1, -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580:
            r, g, b = (wavelength - 510) / (580 - 510), 1, 0
        elif 580 <= wavelength < 645:
            r, g, b = 1, -(wavelength - 645) / (645 - 580), 0
        else:
            r = g = b = 0
        
        return [adjust(r), adjust(g), adjust(b)]
    
    # ========================
    # 5. 符号计算模块
    # ========================
    
    def solve_thickness_symbolically(self, wavelength: float, interference: str = "destructive") -> float:
        """
        使用符号计算求解厚度
        
        Args:
            wavelength: 波长
            interference: 干涉类型（"constructive"或"destructive"）
            
        Returns:
            float: 厚度值
        """
        d = sp.symbols('d')
        
        if interference == "destructive":
            eq = sp.Eq(2 * self.n_film * d, wavelength / 2)
        else:
            eq = sp.Eq(2 * self.n_film * d, wavelength)
        
        sol = sp.solve(eq, d)
        return float(sol[0].evalf())
    
    def find_optimal_thickness(self, target_wavelength: float, theta: float = 0) -> float:
        """
        数值优化求解最优厚度
        
        Args:
            target_wavelength: 目标波长
            theta: 入射角（弧度）
            
        Returns:
            float: 最优厚度
        """
        def objective(d):
            opd = self.optical_path_difference(d, theta, target_wavelength)
            if np.isnan(opd):
                return 1e6
            return abs(opd - target_wavelength / 2)
        
        initial_guess = target_wavelength / (4 * self.n_film)
        result = fsolve(objective, initial_guess)
        return result[0]
    
    # ========================
    # 6. 综合求解模块
    # ========================
    
    def solve_complete_problem(self, wavelength: float, theta: float = 0) -> Dict[str, Union[float, str, Dict]]:
        """
        综合求解薄膜干涉问题
        
        Args:
            wavelength: 波长
            theta: 入射角（弧度）
            
        Returns:
            dict: 完整的分析结果
        """
        try:
            # 计算各种厚度
            d_reflect_max = self.min_thickness_for_constructive_reflection(wavelength, theta)
            d_transmit_max = self.min_thickness_for_constructive_transmission(wavelength, theta)
            d_ar = self.min_antireflection_thickness(wavelength)
            
            # 计算干涉条件
            interference_condition = self.interference_condition(d_ar, theta, wavelength)
            
            # 预测颜色
            color_info = self.predict_color(d_ar, theta)
            
            return {
                "wavelength": wavelength,
                "theta_degrees": np.degrees(theta),
                "n_air": self.n_air,
                "n_film": self.n_film,
                "n_substrate": self.n_substrate,
                "thicknesses": {
                    "reflection_max": d_reflect_max,
                    "transmission_max": d_transmit_max,
                    "antireflection": d_ar
                },
                "interference_condition": interference_condition,
                "color_prediction": color_info,
                "status": "success"
            }
            
        except ValueError as e:
            return {
                "wavelength": wavelength,
                "theta_degrees": np.degrees(theta),
                "error": str(e),
                "status": "error"
            }
    
    # ========================
    # 7. 可视化模块
    # ========================
    
    def plot_intensity_vs_thickness(self, wavelength: float, theta: float = 0, 
                                  d_max: Optional[float] = None, num_points: int = 500):
        """
        绘制强度随厚度的变化
        
        Args:
            wavelength: 波长
            theta: 入射角（弧度）
            d_max: 最大厚度
            num_points: 计算点数
        """
        if d_max is None:
            d_max = 2 * wavelength / self.n_film
        
        d = np.linspace(0, d_max, num_points)
        
        # 计算相位差
        delta = 4 * np.pi * self.n_film * d / wavelength
        
        # 净相位偏移
        upper_phase_change = self._has_half_wave_loss_reflection(self.n_air, self.n_film)
        lower_phase_change = self._has_half_wave_loss_reflection(self.n_film, self.n_substrate)
        net_half_loss = upper_phase_change != lower_phase_change
        phase_offset = np.pi if net_half_loss else 0
        
        # 总相位差
        total_phase = delta + phase_offset
        
        # 强度计算
        I_reflect = np.cos(total_phase / 2) ** 2
        I_transmit = 1 - I_reflect
        
        # 绘图
        plt.figure(figsize=(12, 8))
        plt.plot(d, I_reflect, label="反射光强度", color='red', linewidth=2)
        plt.plot(d, I_transmit, label="透射光强度", color='blue', linewidth=2)
        
        # 标记关键厚度
        d_ar = self.min_antireflection_thickness(wavelength)
        plt.axvline(x=d_ar, color='green', linestyle='--', alpha=0.7, label=f'增透厚度: {d_ar:.2f}')
        
        plt.xlabel("薄膜厚度 (与波长相同单位)")
        plt.ylabel("相对强度")
        plt.title(f"薄膜干涉强度分析\n"
                 f"λ={wavelength}, θ={np.degrees(theta):.1f}°, "
                 f"n₁={self.n_air}, n₂={self.n_film}, n₃={self.n_substrate}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_reflectance_spectrum(self, d: float, theta: float = 0, 
                                title: str = "反射光谱"):
        """
        绘制反射光谱
        
        Args:
            d: 薄膜厚度
            theta: 入射角（弧度）
            title: 图表标题
        """
        wavelengths = np.linspace(400, 700, 300) * 1e-9
        R = self.reflectance_spectrum(d, theta, wavelengths)
        
        plt.figure(figsize=(12, 6))
        
        # 添加颜色背景
        colors = [np.array(self._wavelength_to_rgb(wl * 1e9)) / 255 for wl in wavelengths]
        for i in range(len(wavelengths) - 1):
            plt.axvline(x=wavelengths[i] * 1e9, color=colors[i], linewidth=2, alpha=0.3)
        
        plt.plot(wavelengths * 1e9, R, color='black', linewidth=2, label='反射率')
        plt.xlabel("波长 (nm)")
        plt.ylabel("反射率")
        plt.title(f"{title}\n厚度: {d*1e9:.1f} nm, 入射角: {np.degrees(theta):.1f}°")
        plt.legend()
        plt.xlim(400, 700)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_thickness_wavelength_heatmap(self, d_range: Tuple[float, float], 
                                        wl_range: Tuple[float, float]):
        """
        绘制厚度-波长热图
        
        Args:
            d_range: 厚度范围 (min, max)
            wl_range: 波长范围 (min, max)
        """
        d_vals = np.linspace(d_range[0], d_range[1], 100)
        wl_vals = np.linspace(wl_range[0], wl_range[1], 100)
        D, WL = np.meshgrid(d_vals, wl_vals)
        I = np.zeros_like(D)
        
        for i in range(len(wl_vals)):
            for j in range(len(d_vals)):
                opd = self.optical_path_difference(D[i, j], 0, WL[i, j])
                if np.isnan(opd):
                    I[i, j] = 0
                else:
                    I[i, j] = 0.5 * (1 + np.cos(2 * np.pi * opd / WL[i, j]))
        
        plt.figure(figsize=(12, 8))
        plt.imshow(I, extent=[d_vals.min()*1e9, d_vals.max()*1e9, 
                             wl_vals.min()*1e9, wl_vals.max()*1e9],
                  origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label="干涉强度")
        plt.xlabel("厚度 (nm)")
        plt.ylabel("波长 (nm)")
        plt.title("薄膜干涉：厚度-波长关系图")
        plt.tight_layout()
        plt.show()


# ========================
# 使用示例和测试
# ========================

def demo_enhanced_calculator():
    """演示增强版计算器的功能"""
    print("=== 增强版薄膜干涉计算器演示 ===\n")
    
    # 创建计算器：MgF2增透膜，n=1.38，玻璃基底n=1.5
    calc = EnhancedThinFilmInterferenceCalculator(n_air=1.0, n_film=1.38, n_substrate=1.5)
    
    # 1. 基础计算
    print("1. 基础计算")
    wavelength = 550e-9  # 550 nm
    d_ar = calc.min_antireflection_thickness(wavelength)
    print(f"   最小增透厚度 (550nm): {d_ar*1e9:.2f} nm")
    
    # 2. 综合问题求解
    print("\n2. 综合问题求解")
    result = calc.solve_complete_problem(wavelength)
    if result["status"] == "success":
        print(f"   反射光最强厚度: {result['thicknesses']['reflection_max']*1e9:.2f} nm")
        print(f"   透射光最强厚度: {result['thicknesses']['transmission_max']*1e9:.2f} nm")
        print(f"   增透厚度: {result['thicknesses']['antireflection']*1e9:.2f} nm")
        print(f"   预测颜色峰值: {result['color_prediction']['peak_wavelength_nm']:.1f} nm")
    
    # 3. 斜入射分析
    print("\n3. 斜入射分析")
    theta_30 = np.radians(30)
    try:
        d_ar_30 = calc.min_antireflection_thickness(wavelength)
        print(f"   30°入射角增透厚度: {d_ar_30*1e9:.2f} nm")
    except ValueError as e:
        print(f"   30°入射角: {e}")
    
    # 4. 符号计算
    print("\n4. 符号计算")
    d_symbolic = calc.solve_thickness_symbolically(wavelength, "destructive")
    print(f"   符号求解增透厚度: {d_symbolic*1e9:.2f} nm")
    
    # 5. 干涉判断
    print("\n5. 干涉判断")
    d_test = 100e-9  # 100 nm
    is_constructive = calc.is_constructive(d_test, 0, wavelength)
    is_destructive = calc.is_destructive(d_test, 0, wavelength)
    print(f"   100nm厚度在550nm波长下:")
    print(f"   相长干涉: {is_constructive}")
    print(f"   相消干涉: {is_destructive}")
    
    return calc


if __name__ == "__main__":
    # 运行演示
    calculator = demo_enhanced_calculator()
    
    # 可视化演示
    print("\n=== 可视化演示 ===")
    wavelength = 550e-9
    
    # 1. 强度-厚度图
    print("1. 生成强度-厚度图...")
    calculator.plot_intensity_vs_thickness(wavelength, d_max=300e-9)
    
    # 2. 反射光谱
    print("2. 生成反射光谱...")
    d_ar = calculator.min_antireflection_thickness(wavelength)
    calculator.plot_reflectance_spectrum(d_ar, title="增透膜反射光谱")
    
    # 3. 厚度-波长热图
    print("3. 生成厚度-波长热图...")
    calculator.plot_thickness_wavelength_heatmap(
        d_range=(50e-9, 200e-9), 
        wl_range=(400e-9, 700e-9)
    )
    
    print("\n=== 演示完成 ===")
