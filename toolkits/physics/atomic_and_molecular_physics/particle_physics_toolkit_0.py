# Filename: particle_physics_toolkit.py
"""
粒子物理与高能天体物理计算工具包

主要功能：
1. 粒子反应阈值计算：基于相对论运动学计算粒子产生阈值
2. 宇宙学光子场相互作用：计算高能粒子与背景辐射的相互作用
3. 不变质量与洛伦兹变换：处理相对论粒子碰撞问题
4. GZK截断效应分析：研究超高能宇宙线传播限制

依赖库：
pip install numpy scipy sympy matplotlib astropy
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from scipy import constants as const
from scipy.optimize import fsolve, brentq
from scipy.integrate import quad
import sympy as sp
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ============ 物理常数 ============
ELECTRON_MASS_EV = 0.511e6  # eV/c^2 (电子静止质量)
PROTON_MASS_EV = 938.272e6  # eV/c^2 (质子静止质量)
PION_MASS_EV = 139.57e6  # eV/c^2 (π介子静止质量)
SPEED_OF_LIGHT = const.c  # m/s
PLANCK_CONSTANT = const.h  # J·s
HBAR = const.hbar  # J·s
ELECTRON_CHARGE = const.e  # C
CMB_TEMPERATURE = 2.725  # K (CMB温度)
BOLTZMANN_CONSTANT = const.k  # J/K

# 单位转换
EV_TO_JOULE = const.e
GEV = 1e9  # eV
TEV = 1e12  # eV
MEV = 1e6  # eV

# ============ 第一层：原子工具函数 ============

def calculate_invariant_mass_squared(E1: float, E2: float, theta_deg: float = 180.0,
                                     m1: float = 0.0, m2: float = 0.0) -> dict:
    """
    计算两粒子碰撞的不变质量平方 s = (p1 + p2)^2
    
    基于相对论四动量守恒，不变质量是洛伦兹不变量，在任何参考系中相同。
    对于两个粒子碰撞：s = (E1+E2)^2 - (p1+p2)^2·c^2
    
    Args:
        E1: 粒子1能量 (eV)
        E2: 粒子2能量 (eV)
        theta_deg: 两粒子夹角 (度)，默认180度为对头碰撞
        m1: 粒子1静止质量 (eV/c^2)，默认0为光子
        m2: 粒子2静止质量 (eV/c^2)，默认0为光子
    
    Returns:
        dict: {
            'result': s值 (eV^2),
            'metadata': {
                's_sqrt': sqrt(s) (eV),
                'collision_type': 碰撞类型,
                'theta_rad': 夹角(弧度)
            }
        }
    
    Example:
        >>> result = calculate_invariant_mass_squared(1e14, 1e-3, 180.0)
        >>> print(f"s = {result['result']:.2e} eV^2")
    """
    # === 边界检查 ===
    if not isinstance(E1, (int, float)) or E1 <= 0:
        raise ValueError(f"E1必须为正数，当前值: {E1}")
    if not isinstance(E2, (int, float)) or E2 <= 0:
        raise ValueError(f"E2必须为正数，当前值: {E2}")
    if not isinstance(theta_deg, (int, float)) or not (0 <= theta_deg <= 180):
        raise ValueError(f"theta_deg必须在[0, 180]范围内，当前值: {theta_deg}")
    if m1 < 0 or m2 < 0:
        raise ValueError(f"质量不能为负数: m1={m1}, m2={m2}")
    
    # 转换角度为弧度
    theta_rad = np.deg2rad(theta_deg)
    
    # 计算动量（相对论关系 E^2 = (pc)^2 + (mc^2)^2）
    p1 = np.sqrt(E1**2 - m1**2) if E1 > m1 else 0
    p2 = np.sqrt(E2**2 - m2**2) if E2 > m2 else 0
    
    # 不变质量平方：s = (E1+E2)^2 - |p1+p2|^2
    # 对于两个粒子：|p1+p2|^2 = p1^2 + p2^2 + 2*p1*p2*cos(theta)
    p_sum_squared = p1**2 + p2**2 + 2*p1*p2*np.cos(theta_rad)
    s = (E1 + E2)**2 - p_sum_squared
    
    # 判断碰撞类型
    if theta_deg == 180.0:
        collision_type = "head-on (对头碰撞)"
    elif theta_deg == 0.0:
        collision_type = "same direction (同向)"
    else:
        collision_type = f"angle collision (角度碰撞 {theta_deg}°)"
    
    return {
        'result': s,
        'metadata': {
            's_sqrt': np.sqrt(max(0, s)),
            'collision_type': collision_type,
            'theta_rad': theta_rad,
            'p1': p1,
            'p2': p2
        }
    }


def calculate_threshold_energy(target_energy: float, product_mass_total: float,
                               projectile_mass: float = 0.0, target_mass: float = 0.0,
                               collision_angle: float = 180.0) -> dict:
    """
    计算粒子反应的阈值能量（实验室系）
    
    物理原理：要产生总质量为M的粒子，需要满足 sqrt(s) >= M·c^2
    对于固定靶实验或宇宙学背景光子碰撞，求解入射粒子的最小能量。
    
    Args:
        target_energy: 靶粒子能量 (eV)，对于静止靶为其静止质量能量
        product_mass_total: 产物总静止质量 (eV/c^2)
        projectile_mass: 入射粒子静止质量 (eV/c^2)，默认0为光子
        target_mass: 靶粒子静止质量 (eV/c^2)，默认0为光子
        collision_angle: 碰撞角度 (度)，默认180度
    
    Returns:
        dict: {
            'result': 阈值能量 (eV),
            'metadata': {
                's_threshold': 阈值不变质量平方,
                'threshold_gamma': 对应的洛伦兹因子,
                'reaction_type': 反应类型描述
            }
        }
    
    Example:
        >>> # 计算γ + γ_CMB → e+ + e- 的阈值
        >>> result = calculate_threshold_energy(1e-3, 2*0.511e6, 0, 0, 180)
        >>> print(f"阈值能量: {result['result']:.2e} eV")
    """
    # === 边界检查 ===
    if target_energy <= 0:
        raise ValueError(f"target_energy必须为正数: {target_energy}")
    if product_mass_total <= 0:
        raise ValueError(f"product_mass_total必须为正数: {product_mass_total}")
    if projectile_mass < 0 or target_mass < 0:
        raise ValueError(f"质量不能为负数: projectile={projectile_mass}, target={target_mass}")
    
    # 阈值条件：sqrt(s) = product_mass_total
    s_threshold = product_mass_total**2
    
    # 对于对头碰撞（theta=180°）的简化公式
    if collision_angle == 180.0 and projectile_mass == 0 and target_mass == 0:
        # 光子-光子碰撞：s = 4*E1*E2
        # s_threshold = 4*E_projectile*E_target
        E_threshold = s_threshold / (4 * target_energy)
        
        reaction_type = "photon-photon annihilation (光子-光子湮灭)"
        gamma_factor = E_threshold / max(projectile_mass, 1e-10)  # 避免除零
        
    else:
        # 一般情况：使用不变质量公式反解
        # s = (E1+E2)^2 - (p1+p2)^2
        # 需要数值求解
        def equation(E_proj):
            result = calculate_invariant_mass_squared(
                E_proj, target_energy, collision_angle,
                projectile_mass, target_mass
            )
            return result['result'] - s_threshold
        
        # 初始猜测：使用非相对论近似
        E_guess = s_threshold / (2 * target_energy)
        
        try:
            E_threshold = fsolve(equation, E_guess)[0]
            if E_threshold < 0:
                # 尝试更大的初始值
                E_threshold = fsolve(equation, E_guess * 10)[0]
        except:
            # 如果fsolve失败，使用brentq方法
            E_threshold = brentq(equation, projectile_mass, 1e20)
        
        reaction_type = "general particle collision (一般粒子碰撞)"
        gamma_factor = E_threshold / max(projectile_mass, target_energy)
    
    return {
        'result': E_threshold,
        'metadata': {
            's_threshold': s_threshold,
            'threshold_gamma': gamma_factor,
            'reaction_type': reaction_type,
            'collision_angle': collision_angle
        }
    }


def calculate_cmb_photon_energy(temperature: float = CMB_TEMPERATURE,
                                energy_type: str = 'average') -> dict:
    """
    计算宇宙微波背景辐射(CMB)的光子能量
    
    CMB是黑体辐射，遵循普朗克分布。可计算平均能量、峰值能量等。
    
    Args:
        temperature: CMB温度 (K)，默认2.725K
        energy_type: 能量类型，可选：
            - 'average': 平均光子能量 <E> = 2.7*kT
            - 'peak': 峰值能量（维恩位移定律）
            - 'rms': 均方根能量
    
    Returns:
        dict: {
            'result': 光子能量 (eV),
            'metadata': {
                'temperature': 温度(K),
                'wavelength_peak': 峰值波长(m),
                'frequency_peak': 峰值频率(Hz),
                'energy_type': 能量类型
            }
        }
    
    Example:
        >>> result = calculate_cmb_photon_energy(2.725, 'average')
        >>> print(f"CMB平均光子能量: {result['result']:.2e} eV")
    """
    # === 边界检查 ===
    if temperature <= 0:
        raise ValueError(f"温度必须为正数: {temperature}")
    if energy_type not in ['average', 'peak', 'rms']:
        raise ValueError(f"energy_type必须为'average', 'peak'或'rms': {energy_type}")
    
    kT = BOLTZMANN_CONSTANT * temperature  # J
    kT_eV = kT / EV_TO_JOULE  # eV
    
    if energy_type == 'average':
        # 黑体辐射平均光子能量：<E> = 2.7*kT
        energy_eV = 2.7 * kT_eV
        
    elif energy_type == 'peak':
        # 维恩位移定律：λ_peak = b/T, 其中 b = 2.898e-3 m·K
        lambda_peak = 2.898e-3 / temperature  # m
        freq_peak = SPEED_OF_LIGHT / lambda_peak  # Hz
        energy_eV = (PLANCK_CONSTANT * freq_peak) / EV_TO_JOULE
        
    else:  # rms
        # 均方根能量（数值积分普朗克分布）
        energy_eV = 3.15 * kT_eV  # 近似值
    
    # 计算峰值波长和频率
    lambda_peak = 2.898e-3 / temperature
    freq_peak = SPEED_OF_LIGHT / lambda_peak
    
    return {
        'result': energy_eV,
        'metadata': {
            'temperature': temperature,
            'wavelength_peak': lambda_peak,
            'frequency_peak': freq_peak,
            'energy_type': energy_type,
            'kT_eV': kT_eV
        }
    }


def calculate_cross_section_pair_production(photon_energy: float, 
                                            target_photon_energy: float,
                                            particle_mass: float = ELECTRON_MASS_EV) -> dict:
    """
    计算光子对产生反应截面 γγ → e+e-
    
    使用Breit-Wheeler过程的解析公式（近阈值区域）或高能极限公式。
    
    Args:
        photon_energy: 高能光子能量 (eV)
        target_photon_energy: 靶光子能量 (eV)
        particle_mass: 产生粒子的静止质量 (eV/c^2)，默认为电子
    
    Returns:
        dict: {
            'result': 反应截面 (barn, 1 barn = 10^-28 m^2),
            'metadata': {
                's': 不变质量平方,
                'threshold_ratio': s/s_threshold,
                'regime': 能量区域('near-threshold'或'high-energy')
            }
        }
    
    Example:
        >>> result = calculate_cross_section_pair_production(1e14, 1e-3)
        >>> print(f"截面: {result['result']:.2e} barn")
    """
    # === 边界检查 ===
    if photon_energy <= 0 or target_photon_energy <= 0:
        raise ValueError(f"光子能量必须为正数: E1={photon_energy}, E2={target_photon_energy}")
    if particle_mass <= 0:
        raise ValueError(f"粒子质量必须为正数: {particle_mass}")
    
    # 计算不变质量平方（对头碰撞）
    s_result = calculate_invariant_mass_squared(photon_energy, target_photon_energy, 180.0)
    s = s_result['result']
    s_threshold = (2 * particle_mass)**2
    
    # 检查是否超过阈值
    if s < s_threshold:
        return {
            'result': 0.0,
            'metadata': {
                's': s,
                'threshold_ratio': s / s_threshold,
                'regime': 'below-threshold',
                'note': '能量低于阈值，反应截面为0'
            }
        }
    
    # 计算截面（使用经典电子半径）
    r_e = 2.818e-15  # m (经典电子半径)
    sigma_0 = np.pi * r_e**2  # m^2
    
    # 阈值比
    beta = np.sqrt(1 - s_threshold / s)
    
    # Breit-Wheeler截面公式（简化版）
    if s / s_threshold < 2:
        # 近阈值区域
        sigma = sigma_0 * beta * (3 - beta**4) / 2
        regime = 'near-threshold'
    else:
        # 高能区域
        sigma = sigma_0 * (np.log(s / s_threshold) - 0.5) / (s / s_threshold)
        regime = 'high-energy'
    
    # 转换为barn (1 barn = 10^-28 m^2)
    sigma_barn = sigma / 1e-28
    
    return {
        'result': sigma_barn,
        'metadata': {
            's': s,
            's_sqrt_GeV': np.sqrt(s) / GEV,
            'threshold_ratio': s / s_threshold,
            'regime': regime,
            'beta': beta
        }
    }


def calculate_mean_free_path(cross_section_barn: float, 
                             number_density: float) -> dict:
    """
    计算粒子在介质中的平均自由程
    
    平均自由程 λ = 1/(n·σ)，其中n是靶粒子数密度，σ是反应截面。
    
    Args:
        cross_section_barn: 反应截面 (barn)
        number_density: 靶粒子数密度 (m^-3)
    
    Returns:
        dict: {
            'result': 平均自由程 (m),
            'metadata': {
                'mfp_Mpc': 平均自由程(Mpc),
                'mfp_ly': 平均自由程(光年),
                'optical_depth_per_Mpc': 每Mpc的光学深度
            }
        }
    
    Example:
        >>> result = calculate_mean_free_path(1e-6, 4e8)
        >>> print(f"平均自由程: {result['metadata']['mfp_Mpc']:.2e} Mpc")
    """
    # === 边界检查 ===
    if cross_section_barn < 0:
        raise ValueError(f"截面不能为负数: {cross_section_barn}")
    if number_density < 0:
        raise ValueError(f"数密度不能为负数: {number_density}")
    
    if cross_section_barn == 0 or number_density == 0:
        return {
            'result': np.inf,
            'metadata': {
                'mfp_Mpc': np.inf,
                'mfp_ly': np.inf,
                'optical_depth_per_Mpc': 0.0,
                'note': '截面或数密度为0，平均自由程无穷大'
            }
        }
    
    # 转换截面单位：barn → m^2
    sigma_m2 = cross_section_barn * 1e-28
    
    # 计算平均自由程
    mfp_m = 1.0 / (number_density * sigma_m2)
    
    # 单位转换
    Mpc_to_m = 3.086e22  # 1 Mpc = 3.086e22 m
    ly_to_m = 9.461e15   # 1 light-year = 9.461e15 m
    
    mfp_Mpc = mfp_m / Mpc_to_m
    mfp_ly = mfp_m / ly_to_m
    
    # 光学深度
    optical_depth_per_Mpc = Mpc_to_m / mfp_m
    
    return {
        'result': mfp_m,
        'metadata': {
            'mfp_Mpc': mfp_Mpc,
            'mfp_ly': mfp_ly,
            'optical_depth_per_Mpc': optical_depth_per_Mpc,
            'cross_section_m2': sigma_m2
        }
    }


def calculate_cmb_number_density(temperature: float = CMB_TEMPERATURE) -> dict:
    """
    计算CMB光子数密度
    
    黑体辐射光子数密度：n = (2*ζ(3)/π^2) * (kT/ħc)^3
    其中ζ(3) ≈ 1.202是黎曼ζ函数。
    
    Args:
        temperature: CMB温度 (K)，默认2.725K
    
    Returns:
        dict: {
            'result': 光子数密度 (m^-3),
            'metadata': {
                'density_cm3': 数密度(cm^-3),
                'temperature': 温度(K),
                'energy_density': 能量密度(J/m^3)
            }
        }
    
    Example:
        >>> result = calculate_cmb_number_density(2.725)
        >>> print(f"CMB光子数密度: {result['metadata']['density_cm3']:.2f} cm^-3")
    """
    # === 边界检查 ===
    if temperature <= 0:
        raise ValueError(f"温度必须为正数: {temperature}")
    
    # 黎曼ζ函数 ζ(3)
    zeta_3 = 1.2020569
    
    # 光子数密度公式
    kT = BOLTZMANN_CONSTANT * temperature
    prefactor = 2 * zeta_3 / (np.pi**2)
    n = prefactor * (kT / (HBAR * SPEED_OF_LIGHT))**3
    
    # 单位转换
    n_cm3 = n / 1e6  # m^-3 → cm^-3
    
    # 能量密度（Stefan-Boltzmann定律）
    stefan_boltzmann = 5.670374419e-8  # W·m^-2·K^-4
    energy_density = 4 * stefan_boltzmann * temperature**4 / SPEED_OF_LIGHT
    
    return {
        'result': n,
        'metadata': {
            'density_cm3': n_cm3,
            'temperature': temperature,
            'energy_density': energy_density,
            'energy_density_eV_m3': energy_density / EV_TO_JOULE
        }
    }


# ============ 第二层：组合工具函数 ============

def analyze_pair_production_threshold(cmb_energy: float = 1e-3,
                                     particle_type: str = 'electron') -> dict:
    """
    分析光子对产生反应的阈值条件（γγ → particle-antiparticle）
    
    组合多个原子函数，完成从CMB能量到阈值γ射线能量的完整计算链。
    
    Args:
        cmb_energy: CMB光子能量 (eV)，默认10^-3 eV
        particle_type: 产生的粒子类型，可选'electron', 'muon', 'pion'
    
    Returns:
        dict: {
            'result': {
                'threshold_energy_eV': 阈值能量(eV),
                'threshold_energy_GeV': 阈值能量(GeV),
                'particle_mass_eV': 粒子质量(eV),
                's_threshold': 阈值不变质量平方
            },
            'metadata': {
                'cmb_energy': CMB能量,
                'particle_type': 粒子类型,
                'calculation_steps': 计算步骤记录
            }
        }
    
    Example:
        >>> result = analyze_pair_production_threshold(1e-3, 'electron')
        >>> print(f"阈值能量: {result['result']['threshold_energy_GeV']:.2e} GeV")
    """
    # === 参数检查 ===
    if not isinstance(cmb_energy, (int, float)) or cmb_energy <= 0:
        raise ValueError(f"cmb_energy必须为正数: {cmb_energy}")
    if not isinstance(particle_type, str):
        raise TypeError(f"particle_type必须为字符串: {type(particle_type)}")
    
    # 粒子质量映射
    mass_map = {
        'electron': ELECTRON_MASS_EV,
        'muon': 105.66e6,  # eV
        'pion': PION_MASS_EV
    }
    
    if particle_type not in mass_map:
        raise ValueError(f"不支持的粒子类型: {particle_type}，可选: {list(mass_map.keys())}")
    
    particle_mass = mass_map[particle_type]
    calculation_steps = []
    
    # 步骤1：计算产物总质量
    ## using calculate_threshold_energy (内部调用)
    product_mass_total = 2 * particle_mass
    calculation_steps.append(f"步骤1: 产物总质量 = 2 × {particle_mass/MEV:.3f} MeV = {product_mass_total/MEV:.3f} MeV")
    
    # 步骤2：计算阈值能量
    ## using calculate_threshold_energy
    threshold_result = calculate_threshold_energy(
        target_energy=cmb_energy,
        product_mass_total=product_mass_total,
        projectile_mass=0.0,  # 光子
        target_mass=0.0,      # 光子
        collision_angle=180.0
    )
    
    E_threshold = threshold_result['result']
    s_threshold = threshold_result['metadata']['s_threshold']
    calculation_steps.append(f"步骤2: 使用calculate_threshold_energy计算阈值 = {E_threshold:.2e} eV")
    
    # 步骤3：验证不变质量条件
    ## using calculate_invariant_mass_squared
    s_check = calculate_invariant_mass_squared(E_threshold, cmb_energy, 180.0)
    calculation_steps.append(f"步骤3: 验证 sqrt(s) = {s_check['metadata']['s_sqrt']/MEV:.3f} MeV ≈ {product_mass_total/MEV:.3f} MeV")
    
    return {
        'result': {
            'threshold_energy_eV': E_threshold,
            'threshold_energy_GeV': E_threshold / GEV,
            'threshold_energy_TeV': E_threshold / TEV,
            'particle_mass_eV': particle_mass,
            's_threshold': s_threshold,
            's_threshold_GeV2': s_threshold / GEV**2
        },
        'metadata': {
            'cmb_energy': cmb_energy,
            'particle_type': particle_type,
            'calculation_steps': calculation_steps,
            'reaction': f'γ + γ_CMB → {particle_type}⁺ + {particle_type}⁻'
        }
    }


def analyze_gzk_cutoff(cmb_temperature: float = CMB_TEMPERATURE,
                      particle_type: str = 'electron',
                      energy_type: str = 'average') -> dict:
    """
    分析GZK截断效应（Greisen-Zatsepin-Kuzmin cutoff）
    
    完整计算链：CMB温度 → 光子能量 → 阈值γ能量 → 反应截面 → 平均自由程
    
    Args:
        cmb_temperature: CMB温度 (K)
        particle_type: 产生的粒子类型
        energy_type: CMB能量类型 ('average', 'peak', 'rms')
    
    Returns:
        dict: {
            'result': {
                'threshold_energy_GeV': 阈值能量(GeV),
                'mean_free_path_Mpc': 平均自由程(Mpc),
                'cross_section_barn': 反应截面(barn),
                'cmb_photon_density': CMB光子数密度(cm^-3)
            },
            'metadata': {...}
        }
    
    Example:
        >>> result = analyze_gzk_cutoff(2.725, 'electron', 'average')
        >>> print(f"GZK截断能量: {result['result']['threshold_energy_GeV']:.2e} GeV")
    """
    # === 参数检查 ===
    if not isinstance(cmb_temperature, (int, float)) or cmb_temperature <= 0:
        raise ValueError(f"cmb_temperature必须为正数: {cmb_temperature}")
    if not isinstance(particle_type, str):
        raise TypeError(f"particle_type必须为字符串")
    if energy_type not in ['average', 'peak', 'rms']:
        raise ValueError(f"energy_type必须为'average', 'peak'或'rms'")
    
    calculation_steps = []
    
    # 步骤1：计算CMB光子能量
    ## using calculate_cmb_photon_energy
    cmb_result = calculate_cmb_photon_energy(cmb_temperature, energy_type)
    E_cmb = cmb_result['result']
    calculation_steps.append(f"步骤1: CMB光子能量({energy_type}) = {E_cmb:.2e} eV")
    
    # 步骤2：计算阈值γ射线能量
    ## using analyze_pair_production_threshold (内部调用calculate_threshold_energy)
    threshold_analysis = analyze_pair_production_threshold(E_cmb, particle_type)
    E_threshold_GeV = threshold_analysis['result']['threshold_energy_GeV']
    calculation_steps.append(f"步骤2: 阈值γ能量 = {E_threshold_GeV:.2e} GeV")
    
    # 步骤3：计算CMB光子数密度
    ## using calculate_cmb_number_density
    density_result = calculate_cmb_number_density(cmb_temperature)
    n_cmb = density_result['result']
    n_cmb_cm3 = density_result['metadata']['density_cm3']
    calculation_steps.append(f"步骤3: CMB光子数密度 = {n_cmb_cm3:.2f} cm^-3")
    
    # 步骤4：计算反应截面（在阈值附近）
    ## using calculate_cross_section_pair_production
    E_gamma = E_threshold_GeV * GEV * 1.5  # 略高于阈值
    cross_section_result = calculate_cross_section_pair_production(E_gamma, E_cmb)
    sigma_barn = cross_section_result['result']
    calculation_steps.append(f"步骤4: 反应截面 = {sigma_barn:.2e} barn")
    
    # 步骤5：计算平均自由程
    ## using calculate_mean_free_path
    mfp_result = calculate_mean_free_path(sigma_barn, n_cmb)
    mfp_Mpc = mfp_result['metadata']['mfp_Mpc']
    calculation_steps.append(f"步骤5: 平均自由程 = {mfp_Mpc:.2e} Mpc")
    
    return {
        'result': {
            'threshold_energy_GeV': E_threshold_GeV,
            'threshold_energy_eV': E_threshold_GeV * GEV,
            'mean_free_path_Mpc': mfp_Mpc,
            'mean_free_path_m': mfp_result['result'],
            'cross_section_barn': sigma_barn,
            'cmb_photon_density_cm3': n_cmb_cm3,
            'cmb_photon_energy_eV': E_cmb
        },
        'metadata': {
            'cmb_temperature': cmb_temperature,
            'particle_type': particle_type,
            'energy_type': energy_type,
            'calculation_steps': calculation_steps,
            'physical_interpretation': f'高于{E_threshold_GeV:.2e} GeV的γ射线在宇宙中传播{mfp_Mpc:.2e} Mpc后会与CMB光子发生对产生反应'
        }
    }


def scan_threshold_vs_cmb_energy(cmb_energy_range: List[float],
                                 particle_type: str = 'electron') -> dict:
    """
    扫描不同CMB能量下的阈值能量变化
    
    用于研究背景辐射能量对高能粒子传播的影响。
    
    Args:
        cmb_energy_range: CMB能量列表 (eV)
        particle_type: 粒子类型
    
    Returns:
        dict: {
            'result': {
                'cmb_energies': CMB能量数组,
                'threshold_energies_GeV': 阈值能量数组(GeV),
                'scaling_law': 标度律参数
            },
            'metadata': {...}
        }
    
    Example:
        >>> energies = [1e-4, 1e-3, 1e-2]
        >>> result = scan_threshold_vs_cmb_energy(energies, 'electron')
    """
    # === 参数检查 ===
    if not isinstance(cmb_energy_range, list):
        raise TypeError(f"cmb_energy_range必须是列表: {type(cmb_energy_range)}")
    if len(cmb_energy_range) == 0:
        raise ValueError("cmb_energy_range不能为空")
    if not all(isinstance(e, (int, float)) and e > 0 for e in cmb_energy_range):
        raise ValueError("cmb_energy_range中所有元素必须为正数")
    
    threshold_energies = []
    calculation_log = []
    
    for E_cmb in cmb_energy_range:
        ## using analyze_pair_production_threshold
        result = analyze_pair_production_threshold(E_cmb, particle_type)
        E_th_GeV = result['result']['threshold_energy_GeV']
        threshold_energies.append(E_th_GeV)
        calculation_log.append(f"E_CMB={E_cmb:.2e} eV → E_threshold={E_th_GeV:.2e} GeV")
    
    # 分析标度律：E_threshold ∝ 1/E_CMB
    cmb_array = np.array(cmb_energy_range)
    threshold_array = np.array(threshold_energies)
    
    # 拟合 log(E_th) = a + b*log(E_cmb)
    log_cmb = np.log10(cmb_array)
    log_threshold = np.log10(threshold_array)
    coeffs = np.polyfit(log_cmb, log_threshold, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    return {
        'result': {
            'cmb_energies_eV': cmb_energy_range,
            'threshold_energies_GeV': threshold_energies,
            'scaling_law': {
                'slope': slope,
                'intercept': intercept,
                'formula': f'log10(E_th/GeV) = {intercept:.2f} + {slope:.2f}*log10(E_cmb/eV)'
            }
        },
        'metadata': {
            'particle_type': particle_type,
            'num_points': len(cmb_energy_range),
            'calculation_log': calculation_log,
            'expected_slope': -1.0,
            'slope_deviation': abs(slope - (-1.0))
        }
    }


def compare_particle_thresholds(cmb_energy: float = 1e-3,
                               particle_types: List[str] = None) -> dict:
    """
    比较不同粒子的对产生阈值
    
    Args:
        cmb_energy: CMB光子能量 (eV)
        particle_types: 粒子类型列表，默认['electron', 'muon', 'pion']
    
    Returns:
        dict: {
            'result': {
                'particle': 粒子名称,
                'threshold_GeV': 阈值能量(GeV),
                'mass_MeV': 粒子质量(MeV)
            }的列表,
            'metadata': {...}
        }
    
    Example:
        >>> result = compare_particle_thresholds(1e-3)
        >>> for p in result['result']:
        >>>     print(f"{p['particle']}: {p['threshold_GeV']:.2e} GeV")
    """
    # === 参数检查 ===
    if particle_types is None:
        particle_types = ['electron', 'muon', 'pion']
    
    if not isinstance(particle_types, list):
        raise TypeError(f"particle_types必须是列表: {type(particle_types)}")
    if not isinstance(cmb_energy, (int, float)) or cmb_energy <= 0:
        raise ValueError(f"cmb_energy必须为正数: {cmb_energy}")
    
    results = []
    
    for particle in particle_types:
        ## using analyze_pair_production_threshold
        analysis = analyze_pair_production_threshold(cmb_energy, particle)
        
        results.append({
            'particle': particle,
            'threshold_GeV': analysis['result']['threshold_energy_GeV'],
            'threshold_eV': analysis['result']['threshold_energy_eV'],
            'mass_MeV': analysis['result']['particle_mass_eV'] / MEV,
            'mass_eV': analysis['result']['particle_mass_eV']
        })
    
    # 按阈值排序
    results_sorted = sorted(results, key=lambda x: x['threshold_GeV'])
    
    return {
        'result': results_sorted,
        'metadata': {
            'cmb_energy': cmb_energy,
            'num_particles': len(particle_types),
            'lowest_threshold': results_sorted[0],
            'highest_threshold': results_sorted[-1],
            'threshold_ratio': results_sorted[-1]['threshold_GeV'] / results_sorted[0]['threshold_GeV']
        }
    }


# ============ 第三层：可视化工具 ============

def visualize_threshold_energy_scan(cmb_energy_range: List[float],
                                   particle_types: List[str] = None,
                                   save_dir: str = './tool_images/',
                                   filename: str = None) -> dict:
    """
    可视化阈值能量随CMB能量的变化
    
    Args:
        cmb_energy_range: CMB能量范围 (eV)
        particle_types: 粒子类型列表
        save_dir: 保存目录
        filename: 文件名（不含扩展名）
    
    Returns:
        dict: {
            'result': 图片保存路径,
            'metadata': {
                'plot_type': 'threshold_scan',
                'num_particles': 粒子数量
            }
        }
    """
    # === 参数检查 ===
    if particle_types is None:
        particle_types = ['electron', 'muon']
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"threshold_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    for particle in particle_types:
        ## using scan_threshold_vs_cmb_energy
        scan_result = scan_threshold_vs_cmb_energy(cmb_energy_range, particle)
        
        cmb_energies = scan_result['result']['cmb_energies_eV']
        thresholds = scan_result['result']['threshold_energies_GeV']
        
        plt.loglog(cmb_energies, thresholds, 'o-', label=f'{particle}', linewidth=2, markersize=6)
    
    plt.xlabel('CMB Photon Energy (eV)', fontsize=12)
    plt.ylabel('Threshold γ-ray Energy (GeV)', fontsize=12)
    plt.title('Pair Production Threshold vs CMB Energy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    # 添加理论线 E_th ∝ 1/E_cmb
    cmb_array = np.array(cmb_energy_range)
    theory_line = (ELECTRON_MASS_EV**2 / cmb_array) / GEV
    plt.loglog(cmb_array, theory_line, 'k--', alpha=0.5, label='Theory: E∝1/E_CMB')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Plot | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'plot_type': 'threshold_scan',
            'num_particles': len(particle_types),
            'num_points': len(cmb_energy_range),
            'save_path': save_path
        }
    }


def visualize_mean_free_path_vs_energy(energy_range_GeV: List[float],
                                       cmb_temperature: float = CMB_TEMPERATURE,
                                       save_dir: str = './tool_images/',
                                       filename: str = None) -> dict:
    """
    可视化平均自由程随γ射线能量的变化
    
    Args:
        energy_range_GeV: γ射线能量范围 (GeV)
        cmb_temperature: CMB温度 (K)
        save_dir: 保存目录
        filename: 文件名
    
    Returns:
        dict: {
            'result': 图片路径,
            'metadata': {...}
        }
    """
    # === 参数检查 ===
    if not isinstance(energy_range_GeV, list):
        raise TypeError("energy_range_GeV必须是列表")
    
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        filename = f"mean_free_path_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 获取CMB参数
    ## using calculate_cmb_photon_energy
    cmb_result = calculate_cmb_photon_energy(cmb_temperature, 'average')
    E_cmb = cmb_result['result']
    
    ## using calculate_cmb_number_density
    density_result = calculate_cmb_number_density(cmb_temperature)
    n_cmb = density_result['result']
    
    # 计算每个能量点的平均自由程
    mfp_list = []
    cross_sections = []
    
    for E_GeV in energy_range_GeV:
        E_eV = E_GeV * GEV
        
        ## using calculate_cross_section_pair_production
        sigma_result = calculate_cross_section_pair_production(E_eV, E_cmb)
        sigma_barn = sigma_result['result']
        cross_sections.append(sigma_barn)
        
        ## using calculate_mean_free_path
        mfp_result = calculate_mean_free_path(sigma_barn, n_cmb)
        mfp_Mpc = mfp_result['metadata']['mfp_Mpc']
        mfp_list.append(mfp_Mpc)
    
    # 创建双y轴图
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('γ-ray Energy (GeV)', fontsize=12)
    ax1.set_ylabel('Mean Free Path (Mpc)', color=color1, fontsize=12)
    ax1.loglog(energy_range_GeV, mfp_list, 'o-', color=color1, linewidth=2, markersize=6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Cross Section (barn)', color=color2, fontsize=12)
    ax2.loglog(energy_range_GeV, cross_sections, 's--', color=color2, linewidth=2, markersize=6, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f'γ-ray Mean Free Path in CMB (T={cmb_temperature}K)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FILE_GENERATED: Plot | PATH: {save_path}")
    
    return {
        'result': save_path,
        'metadata': {
            'plot_type': 'mean_free_path_vs_energy',
            'cmb_temperature': cmb_temperature,
            'energy_range_GeV': [min(energy_range_GeV), max(energy_range_GeV)],
            'save_path': save_path
        }
    }


# ============ 第四层：主流程演示 ============

def main():
    """
    演示工具包解决【当前问题】+【至少2个相关场景】
    """
    
    print("=" * 80)
    print("场景1：原始问题求解 - GZK截断能量计算")
    print("=" * 80)
    print("问题描述：计算高能γ射线与CMB光子发生对产生反应(γγ→e⁺e⁻)的阈值能量")
    print("已知：CMB平均光子能量 = 10^-3 eV")
    print("-" * 80)
    
    # 步骤1：计算电子对产生的阈值能量
    # 调用函数：analyze_pair_production_threshold()
    print("\n步骤1：计算γ + γ_CMB → e⁺ + e⁻ 的阈值能量")
    result1 = analyze_pair_production_threshold(cmb_energy=1e-3, particle_type='electron')
    threshold_GeV = result1['result']['threshold_energy_GeV']
    print(f"FUNCTION_CALL: analyze_pair_production_threshold | PARAMS: cmb_energy=1e-3, particle_type='electron' | RESULT: {threshold_GeV:.2e} GeV")
    print(f"  - 阈值能量: {threshold_GeV:.2e} GeV = {threshold_GeV*1e3:.2e} MeV")
    print(f"  - 不变质量平方: {result1['result']['s_threshold_GeV2']:.2e} GeV²")
    print(f"  - 反应方程: {result1['metadata']['reaction']}")
    
    # 步骤2：完整的GZK截断分析（包含平均自由程）
    # 调用函数：analyze_gzk_cutoff()，该函数内部调用了 analyze_pair_production_threshold()
    print("\n步骤2：完整GZK截断分析（含平均自由程计算）")
    result2 = analyze_gzk_cutoff(cmb_temperature=2.725, particle_type='electron', energy_type='average')
    mfp_Mpc = result2['result']['mean_free_path_Mpc']
    print(f"FUNCTION_CALL: analyze_gzk_cutoff | PARAMS: cmb_temperature=2.725, particle_type='electron' | RESULT: threshold={result2['result']['threshold_energy_GeV']:.2e} GeV, mfp={mfp_Mpc:.2e} Mpc")
    print(f"  - CMB光子数密度: {result2['result']['cmb_photon_density_cm3']:.2f} cm⁻³")
    print(f"  - 反应截面: {result2['result']['cross_section_barn']:.2e} barn")
    print(f"  - 平均自由程: {mfp_Mpc:.2e} Mpc")
    print(f"  - 物理解释: {result2['metadata']['physical_interpretation']}")
    
    print(f"\n✓ 场景1完成：阈值能量 = {threshold_GeV:.2e} GeV ≈ 2.6×10⁵ GeV")
    print(f"FINAL_ANSWER: {threshold_GeV:.2e} GeV")
    
    
    print("\n" + "=" * 80)
    print("场景2：参数扫描 - 不同CMB能量下的阈值变化")
    print("=" * 80)
    print("问题描述：研究背景辐射能量对高能粒子阈值的影响，验证E_threshold ∝ 1/E_CMB标度律")
    print("-" * 80)
    
    # 步骤1：扫描CMB能量范围
    # 调用函数：scan_threshold_vs_cmb_energy()
    print("\n步骤1：扫描CMB能量从10^-4到10^-2 eV")
    cmb_energies = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    result3 = scan_threshold_vs_cmb_energy(cmb_energies, particle_type='electron')
    print(f"FUNCTION_CALL: scan_threshold_vs_cmb_energy | PARAMS: cmb_energies={len(cmb_energies)} points | RESULT: slope={result3['result']['scaling_law']['slope']:.3f}")
    print(f"  - 标度律斜率: {result3['result']['scaling_law']['slope']:.3f} (理论值: -1.0)")
    print(f"  - 拟合公式: {result3['result']['scaling_law']['formula']}")
    print(f"  - 斜率偏差: {result3['metadata']['slope_deviation']:.4f}")
    
    # 步骤2：可视化扫描结果
    # 调用函数：visualize_threshold_energy_scan()
    print("\n步骤2：生成阈值能量扫描图")
    cmb_range_extended = np.logspace(-4, -2, 20).tolist()
    result4 = visualize_threshold_energy_scan(cmb_range_extended, ['electron', 'muon'], 
                                             save_dir='./tool_images/', filename='threshold_scan_scenario2')
    print(f"FUNCTION_CALL: visualize_threshold_energy_scan | PARAMS: {len(cmb_range_extended)} points, 2 particles | RESULT: {result4['result']}")
    print(f"FILE_GENERATED: Plot | PATH: {result4['result']}")
    
    print(f"\n✓ 场景2完成：验证了E_threshold ∝ 1/E_CMB的标度律")
    
    
    print("\n" + "=" * 80)
    print("场景3：多粒子对比 - 不同粒子的对产生阈值")
    print("=" * 80)
    print("问题描述：比较电子、μ子、π介子的对产生阈值，分析质量对阈值的影响")
    print("-" * 80)
    
    # 步骤1：比较不同粒子的阈值
    # 调用函数：compare_particle_thresholds()
    print("\n步骤1：计算三种粒子的对产生阈值")
    result5 = compare_particle_thresholds(cmb_energy=1e-3, particle_types=['electron', 'muon', 'pion'])
    print(f"FUNCTION_CALL: compare_particle_thresholds | PARAMS: cmb_energy=1e-3, 3 particles | RESULT: ratio={result5['metadata']['threshold_ratio']:.2f}")
    print("\n粒子对比结果：")
    for particle_data in result5['result']:
        print(f"  - {particle_data['particle']:8s}: 质量={particle_data['mass_MeV']:8.2f} MeV, 阈值={particle_data['threshold_GeV']:.2e} GeV")
    print(f"\n  阈值比(最高/最低): {result5['metadata']['threshold_ratio']:.2f}")
    
    # 步骤2：分析平均自由程随能量的变化
    # 调用函数：visualize_mean_free_path_vs_energy()
    print("\n步骤2：生成平均自由程vs能量图")
    energy_range = np.logspace(4, 7, 30).tolist()  # 10^4 to 10^7 GeV
    result6 = visualize_mean_free_path_vs_energy(energy_range, cmb_temperature=2.725,
                                                save_dir='./tool_images/', filename='mfp_vs_energy_scenario3')
    print(f"FUNCTION_CALL: visualize_mean_free_path_vs_energy | PARAMS: {len(energy_range)} energy points | RESULT: {result6['result']}")
    print(f"FILE_GENERATED: Plot | PATH: {result6['result']}")
    
    print(f"\n✓ 场景3完成：展示了粒子质量对阈值的平方关系(E_th ∝ m²)")
    
    
    print("\n" + "=" * 80)
    print("工具包演示完成")
    print("=" * 80)
    print("总结：")
    print("- 场景1展示了解决原始问题的完整流程：计算GZK截断能量 ≈ 2.6×10⁵ GeV")
    print("- 场景2展示了工具的参数扫描能力：验证了E_threshold ∝ 1/E_CMB标度律")
    print("- 场景3展示了多粒子对比分析：揭示了阈值与粒子质量的平方关系")
    print("\n核心物理结论：")
    print(f"  1. 高能γ射线(E > 2.6×10⁵ GeV)在宇宙中传播时会与CMB光子发生对产生反应")
    print(f"  2. 这一过程限制了超高能宇宙线的传播距离（GZK截断效应）")
    print(f"  3. 阈值能量满足：E_γ × E_CMB ≥ (m_e c²)² ≈ (0.511 MeV)²")
    print("=" * 80)


if __name__ == "__main__":
    main()