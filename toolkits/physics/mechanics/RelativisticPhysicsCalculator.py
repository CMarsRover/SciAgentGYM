"""
RelativisticPhysicsCalculator - 一个通用的相对论物理计算工具
包含三大功能模块：
1. coding func: 核心计算函数
2. math func: 符号化数学推导
3. visual func: 可视化分析
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import c
from scipy.integrate import solve_ivp
from sympy import symbols, simplify, sqrt, latex, Eq
import warnings
warnings.filterwarnings("ignore")

# =============================================
# 类定义：RelativisticPhysicsCalculator
# =============================================
class RelativisticPhysicsCalculator:
    def __init__(self):
        self.v = symbols('v')  # 符号化速度
        self.c_sym = symbols('c')  # 符号化光速

    # --------------------------------------------------
    # 1. 狭义相对论：洛伦兹因子与基本效应
    # --------------------------------------------------

    def lorentz_factor_coding(self, v):
        """【coding func】数值计算洛伦兹因子 γ = 1/sqrt(1 - v²/c²)"""
        beta = (v / c).to('')
        if beta >= 1:
            raise ValueError("速度不能大于等于光速！")
        return 1 / np.sqrt(1 - beta**2)

    def lorentz_factor_math(self):
        """【math func】符号推导洛伦兹因子公式"""
        gamma = 1 / sqrt(1 - (self.v / self.c_sym)**2)
        return Eq(symbols('γ'), gamma), latex(Eq(symbols('γ'), gamma))

    def plot_lorentz_factor_visual(self, v_range=(0, 0.99), num_points=200):
        """【visual func】绘制洛伦兹因子随速度变化曲线"""
        v_vals = np.linspace(v_range[0], v_range[1], num_points) * c
        gamma_vals = [self.lorentz_factor_coding(v) for v in v_vals]

        plt.figure(figsize=(8, 5))
        plt.plot(v_vals / c, gamma_vals, 'b-', linewidth=2, label=r'$\gamma = 1/\sqrt{1 - v^2/c^2}$')
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Velocity $v/c$')
        plt.ylabel('Lorentz Factor $\\gamma$')
        plt.title('Lorentz Factor vs Velocity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(v_range[0], v_range[1])
        plt.ylim(1, max(gamma_vals) * 1.1)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 2. 时间膨胀与长度收缩
    # --------------------------------------------------

    def time_dilation_coding(self, proper_time, v):
        """【coding func】时间膨胀：坐标时 = γ × 固有时"""
        gamma = self.lorentz_factor_coding(v)
        return proper_time * gamma

    def length_contraction_coding(self, proper_length, v):
        """【coding func】长度收缩：观测长度 = 固有长度 / γ"""
        gamma = self.lorentz_factor_coding(v)
        return proper_length / gamma

    def relativistic_effects_math(self):
        """【math func】符号化表达时间膨胀与长度收缩"""
        tau, t, L0, L = symbols('τ t L_0 L')
        gamma = 1 / sqrt(1 - (self.v / self.c_sym)**2)
        time_dilation_eq = Eq(t, gamma * tau)
        length_contraction_eq = Eq(L, L0 / gamma)
        return time_dilation_eq, length_contraction_eq

    def plot_time_length_effects_visual(self, v_range=(0, 0.99), num_points=200):
        """【visual func】绘制时间膨胀与长度收缩对比图"""
        v_vals = np.linspace(v_range[0], v_range[1], num_points)
        gamma_vals = 1 / np.sqrt(1 - v_vals**2)
        time_dilated = gamma_vals
        length_contracted = 1 / gamma_vals

        plt.figure(figsize=(10, 6))
        plt.plot(v_vals, time_dilated, 'r-', label='Time Dilation $t = \\gamma τ$', linewidth=2)
        plt.plot(v_vals, length_contracted, 'b--', label='Length Contraction $L = L_0 / \\gamma$', linewidth=2)
        plt.xlabel('Velocity $v/c$')
        plt.ylabel('Relative Change')
        plt.title('Relativistic Effects: Time Dilation vs Length Contraction')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(v_range[0], v_range[1])
        plt.ylim(0, max(time_dilated) * 1.1)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 3. 相对论速度叠加
    # --------------------------------------------------

    def velocity_addition_coding(self, u, v):
        """【coding func】相对论速度叠加：w = (u + v) / (1 + u*v/c²)"""
        u_vel = u.to('m/s')
        v_vel = v.to('m/s')
        numerator = u_vel + v_vel
        denominator = 1 + (u_vel * v_vel) / c**2
        return (numerator / denominator).to('m/s')

    def velocity_addition_math(self):
        """【math func】符号推导速度叠加公式"""
        u, v, w = symbols('u v w')
        addition_formula = (u + v) / (1 + u*v/self.c_sym**2)
        return Eq(w, addition_formula), latex(Eq(w, addition_formula))

    def plot_velocity_addition_visual(self, v_rest=0.8*c, u_range=(-0.99, 0.99), num_points=200):
        """【visual func】经典 vs 相对论速度叠加对比"""
        u_vals = np.linspace(u_range[0], u_range[1], num_points)
        classical = v_rest.value / c.value + u_vals  # 经典叠加
        relativistic = np.array([
            self.velocity_addition_coding(u * c, v_rest).value / c.value
            for u in u_vals
        ])

        plt.figure(figsize=(9, 6))
        plt.plot(u_vals, classical, 'g--', label='Classical Addition $u+v$', alpha=0.8)
        plt.plot(u_vals, relativistic, 'b-', label='Relativistic Addition $(u+v)/(1+uv/c^2)$', linewidth=2)
        plt.axhline(y=1, color='k', linestyle=':', alpha=0.5, label='Speed of Light $c$')
        plt.axhline(y=-1, color='k', linestyle=':', alpha=0.5)
        plt.xlabel('Velocity in S′ Frame $u/c$')
        plt.ylabel('Resultant Velocity in S Frame $w/c$')
        plt.title(f'Velocity Addition Comparison (S′ moving at {v_rest/c:.2f}c relative to S)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(u_range)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 4. 相对论能量与动量
    # --------------------------------------------------

    def energy_momentum_coding(self, m, v):
        """【coding func】计算相对论总能量与动量"""
        gamma = self.lorentz_factor_coding(v)
        E_total = gamma * m * c**2
        p = gamma * m * v
        E_rest = m * c**2
        E_kinetic = E_total - E_rest
        return {
            'total_energy': E_total.to('MeV'),
            'rest_energy': E_rest.to('MeV'),
            'kinetic_energy': E_kinetic.to('MeV'),
            'momentum': p
        }

    def energy_momentum_math(self):
        """【math func】符号化能量-动量关系"""
        m, E, p = symbols('m E p')
        E_total = symbols('E_total')
        rest_energy = self.c_sym**2 * m
        relativistic_energy = self.c_sym**2 * m / sqrt(1 - (self.v / self.c_sym)**2)
        invariant = Eq(E_total**2, p**2 * self.c_sym**2 + m**2 * self.c_sym**4)
        return Eq(E, relativistic_energy), invariant

    def plot_energy_components_visual(self, m, v_range=(0, 0.99), num_points=200):
        """【visual func】绘制能量随速度变化：静能、动能、总能"""
        v_vals = np.linspace(v_range[0], v_range[1], num_points) * c
        rest_E = (m * c**2).to('MeV').value
        total_E = np.array([self.energy_momentum_coding(m, v)['total_energy'].value for v in v_vals])
        kinetic_E = total_E - rest_E

        plt.figure(figsize=(9, 6))
        plt.plot(v_vals / c, [rest_E]*len(v_vals), 'k--', label='Rest Energy $mc^2$', linewidth=2)
        plt.plot(v_vals / c, kinetic_E, 'r-', label='Kinetic Energy $(\\gamma-1)mc^2$', linewidth=2)
        plt.plot(v_vals / c, total_E, 'b-', label='Total Energy $\\gamma mc^2$', linewidth=2)
        plt.xlabel('Velocity $v/c$')
        plt.ylabel('Energy (MeV)')
        plt.title(f'Relativistic Energy Components for Electron (Mass {m:.2e} kg)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(v_range[0], v_range[1])
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 5. 广义相对论：史瓦西测地线（简化示例）
    # --------------------------------------------------

    def schwarzschild_geodesic_rhs(self, t, y, M):
        """【coding func】史瓦西度规下赤道面测地线微分方程右侧（简化）"""
        # y = [t, r, phi, dt/dλ, dr/dλ, dphi/dλ]
        G = 6.67430e-11  # m³ kg⁻¹ s⁻²
        c_val = c.value
        r_s = 2 * G * M / c_val**2  # 史瓦西半径
        r = y[1]
        if r <= r_s:
            raise ValueError("粒子进入事件视界，数值不稳定")
        dt_dλ = y[3]
        dr_dλ = y[4]
        dphi_dλ = y[5]

        d2t_dλ2 = - (r_s / (r**2 * (1 - r_s/r))) * dt_dλ * dr_dλ
        d2r_dλ2 = - (r_s * c_val**2) / (2 * r**2) * dt_dλ**2 \
                  + (r_s / (2*(r - r_s))) * dr_dλ**2 \
                  + (r - r_s) * r * dphi_dλ**2
        d2phi_dλ2 = - (2 / r) * dr_dλ * dphi_dλ

        return [dt_dλ, dr_dλ, dphi_dλ, d2t_dλ2, d2r_dλ2, d2phi_dλ2]

    def solve_black_hole_orbit_coding(self, M, r0, v0_radial, v0_angular, lambda_max=1000, steps=10000):
        """数值求解黑洞附近粒子轨道（简化模型）"""
        # 初始条件：λ=0 时的位置和导数（仿射参数)
        y0 = [0, r0, 0, 1, v0_radial, v0_angular]
        lambdas = np.linspace(0, lambda_max, steps)
        try:
            sol = solve_ivp(
                lambda t, y: self.schwarzschild_geodesic_rhs(t, y, M),
                [0, lambda_max], y0, t_eval=lambdas, method='RK45', rtol=1e-8
            )
            return sol.t, sol.y
        except Exception as e:
            print("轨道求解失败:", str(e))
            return None, None

    def plot_black_hole_orbit_visual(self, M, r0, v0_radial, v0_angular):
        """【visual func】绘制黑洞周围粒子轨道"""
        t_vals, y = self.solve_black_hole_orbit_coding(M, r0, v0_radial, v0_angular)
        if y is None:
            return

        r = y[1]
        phi = y[2]
        x = r * np.cos(phi)
        y_coord = r * np.sin(phi)

        G = 6.67430e-11
        c_val = c.value
        r_s = 2 * G * M / c_val**2

        plt.figure(figsize=(8, 8))
        plt.plot(x, y_coord, 'b-', linewidth=1.5, label='Particle Trajectory')
        circle = plt.Circle((0, 0), r_s, color='k', alpha=0.8, label='Event Horizon')
        plt.gca().add_patch(circle)
        plt.scatter([0], [0], color='red', s=50, label='Black Hole Center')
        plt.axis('equal')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title(f'Particle Orbit Around Schwarzschild Black Hole (M={M:.2e} kg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------
    # 6. 相对论质量：纵向质量与横向质量
    # --------------------------------------------------

    def longitudinal_mass_coding(self, m, v):
        """【coding func】计算纵向质量：m_long = γ³m"""
        gamma = self.lorentz_factor_coding(v)
        return gamma**3 * m

    def transverse_mass_coding(self, m, v):
        """【coding func】计算横向质量：m_trans = γm"""
        gamma = self.lorentz_factor_coding(v)
        return gamma * m

    def relativistic_mass_math(self):
        """【math func】符号化表达纵向质量和横向质量"""
        m, m_long, m_trans = symbols('m m_long m_trans')
        gamma = 1 / sqrt(1 - (self.v / self.c_sym)**2)
        longitudinal_mass_eq = Eq(m_long, gamma**3 * m)
        transverse_mass_eq = Eq(m_trans, gamma * m)
        return longitudinal_mass_eq, transverse_mass_eq

    def plot_relativistic_mass_visual(self, m, v_range=(0, 0.99), num_points=200):
        """【visual func】绘制纵向质量和横向质量随速度变化"""
        v_vals = np.linspace(v_range[0], v_range[1], num_points)
        gamma_vals = 1 / np.sqrt(1 - v_vals**2)
        m_long_vals = gamma_vals**3 * m.value
        m_trans_vals = gamma_vals * m.value
        m_rest = m.value

        plt.figure(figsize=(10, 6))
        plt.plot(v_vals, [m_rest]*len(v_vals), 'k--', label='Rest Mass $m$', linewidth=2)
        plt.plot(v_vals, m_trans_vals, 'b-', label='Transverse Mass $m_{trans} = \\gamma m$', linewidth=2)
        plt.plot(v_vals, m_long_vals, 'r-', label='Longitudinal Mass $m_{long} = \\gamma^3 m$', linewidth=2)
        plt.xlabel('Velocity $v/c$')
        plt.ylabel('Mass (kg)')
        plt.title(f'Relativistic Mass vs Velocity (Rest Mass {m:.2e} kg)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(v_range[0], v_range[1])
        plt.yscale('log')  # 使用对数坐标更好地显示差异
        plt.tight_layout()
        plt.show()

    def calculate_force_acceleration_ratio_coding(self, m, v, force_direction='longitudinal'):
        """【coding func】计算力与加速度的比值（纵向或横向）"""
        if force_direction == 'longitudinal':
            # 纵向力：F = m_long * a
            m_eff = self.longitudinal_mass_coding(m, v)
        elif force_direction == 'transverse':
            # 横向力：F = m_trans * a
            m_eff = self.transverse_mass_coding(m, v)
        else:
            raise ValueError("force_direction 必须是 'longitudinal' 或 'transverse'")
        
        return m_eff

    def relativistic_mass_comparison_coding(self, m, v):
        """【coding func】比较不同质量定义"""
        m_rest = m
        m_trans = self.transverse_mass_coding(m, v)
        m_long = self.longitudinal_mass_coding(m, v)
        gamma = self.lorentz_factor_coding(v)
        
        return {
            'rest_mass': m_rest,
            'transverse_mass': m_trans,
            'longitudinal_mass': m_long,
            'lorentz_factor': gamma,
            'mass_ratio_longitudinal': m_long / m_rest,
            'mass_ratio_transverse': m_trans / m_rest
        }


if __name__ == "__main__":
    calc = RelativisticPhysicsCalculator()

    # 示例1：洛伦兹因子
    print("=== 洛伦兹因子符号表达式 ===")
    eq, latex_str = calc.lorentz_factor_math()
    print(eq)

    print("\n=== 时间膨胀与长度收缩公式 ===")
    time_eq, length_eq = calc.relativistic_effects_math()
    print("时间膨胀:", time_eq)
    print("长度收缩:", length_eq)

    # 示例2：电子在0.9c时的能量
    electron_mass = 9.109e-31 * u.kg
    result = calc.energy_momentum_coding(electron_mass, 0.9 * c)
    print(f"\n=== 电子在0.9c时的能量 ===")
    for k, v in result.items():
        print(f"{k}: {v:.3f}")

    # 示例3：速度叠加
    u = 0.8 * c
    v = 0.7 * c
    w = calc.velocity_addition_coding(u, v)
    print(f"\n=== 速度叠加：{u/c:.1f}c + {v/c:.1f}c = {w/c:.3f}c")

    # 示例4：相对论质量（纵向质量和横向质量）
    print(f"\n=== 相对论质量计算 ===")
    mass_comparison = calc.relativistic_mass_comparison_coding(electron_mass, 0.9 * c)
    print(f"静质量: {mass_comparison['rest_mass']:.3e} kg")
    print(f"横向质量: {mass_comparison['transverse_mass']:.3e} kg")
    print(f"纵向质量: {mass_comparison['longitudinal_mass']:.3e} kg")
    print(f"洛伦兹因子: {mass_comparison['lorentz_factor']:.3f}")
    print(f"纵向质量比: {mass_comparison['mass_ratio_longitudinal']:.3f}")
    print(f"横向质量比: {mass_comparison['mass_ratio_transverse']:.3f}")

    # 示例5：力与加速度比值
    print(f"\n=== 力与加速度比值 ===")
    m_long_ratio = calc.calculate_force_acceleration_ratio_coding(electron_mass, 0.9 * c, 'longitudinal')
    m_trans_ratio = calc.calculate_force_acceleration_ratio_coding(electron_mass, 0.9 * c, 'transverse')
    print(f"纵向力/加速度比值: {m_long_ratio:.3e} kg")
    print(f"横向力/加速度比值: {m_trans_ratio:.3e} kg")
    calc.plot_lorentz_factor_visual()
    calc.plot_time_length_effects_visual()
    calc.plot_velocity_addition_visual()
    calc.plot_energy_components_visual(electron_mass)
    calc.plot_black_hole_orbit_visual(M=1.989e30, r0=10e3, v0_radial=0, v0_angular=1e-5)
    calc.plot_relativistic_mass_visual(electron_mass)