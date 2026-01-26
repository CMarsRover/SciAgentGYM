# Filename: physical_chemistry_toolkit.py

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import pandas as pd

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保图片保存目录存在
if not os.path.exists("./images"):
    os.makedirs("./images")

def ideal_gas_calculation(pressure, volume, temperature, gas_constant=8.314, quantity=None, 
                          units={'P': 'Pa', 'V': 'm³', 'T': 'K', 'n': 'mol', 'R': 'J/(mol·K)'}):
    """
    理想气体状态方程计算工具，可以计算压力、体积、温度、物质的量中的任意一个，给定其他三个参数。
    
    基于理想气体状态方程: PV = nRT
    
    Parameters:
    -----------
    pressure : float or None
        气体压力，单位由units['P']指定，默认为Pa
    volume : float or None
        气体体积，单位由units['V']指定，默认为m³
    temperature : float or None
        气体温度，单位由units['T']指定，默认为K
    gas_constant : float, optional
        气体常数，默认为8.314 J/(mol·K)
    quantity : float or None
        物质的量，单位由units['n']指定，默认为mol
    units : dict, optional
        各参数的单位，支持常见单位转换
        
    Returns:
    --------
    dict
        包含所有参数的字典，包括计算得到的未知参数
        
    Examples:
    ---------
    >>> # 计算1 mol气体在273.15 K和101325 Pa下的体积
    >>> ideal_gas_calculation(101325, None, 273.15, quantity=1)
    {'pressure': 101325, 'volume': 0.022414, 'temperature': 273.15, 'quantity': 1, 'gas_constant': 8.314}
    """
    # 单位转换
    P_factor = {'Pa': 1, 'kPa': 1e3, 'bar': 1e5, 'atm': 101325}
    V_factor = {'m³': 1, 'L': 1e-3, 'mL': 1e-6, 'cm³': 1e-6}
    T_factor = {'K': 1, '°C': lambda t: t + 273.15}
    
    # 转换输入单位到SI单位
    if pressure is not None:
        if units['P'] in P_factor:
            P_si = pressure * P_factor[units['P']]
        else:
            raise ValueError(f"不支持的压力单位: {units['P']}")
    else:
        P_si = None
        
    if volume is not None:
        if units['V'] in V_factor:
            V_si = volume * V_factor[units['V']]
        else:
            raise ValueError(f"不支持的体积单位: {units['V']}")
    else:
        V_si = None
        
    if temperature is not None:
        if units['T'] in T_factor:
            if callable(T_factor[units['T']]):
                T_si = T_factor[units['T']](temperature)
            else:
                T_si = temperature * T_factor[units['T']]
        else:
            raise ValueError(f"不支持的温度单位: {units['T']}")
    else:
        T_si = None
    
    # 计算未知参数
    if pressure is None and all(x is not None for x in [volume, temperature, quantity]):
        P_si = quantity * gas_constant * T_si / V_si
    elif volume is None and all(x is not None for x in [pressure, temperature, quantity]):
        V_si = quantity * gas_constant * T_si / P_si
    elif temperature is None and all(x is not None for x in [pressure, volume, quantity]):
        T_si = P_si * V_si / (quantity * gas_constant)
    elif quantity is None and all(x is not None for x in [pressure, volume, temperature]):
        quantity = P_si * V_si / (gas_constant * T_si)
    else:
        raise ValueError("必须提供四个参数中的三个才能计算第四个参数")
    
    # 转换回原始单位
    if units['P'] in P_factor:
        pressure = P_si / P_factor[units['P']]
    
    if units['V'] in V_factor:
        volume = V_si / V_factor[units['V']]
    
    if units['T'] in T_factor:
        if callable(T_factor[units['T']]):
            temperature = T_si - 273.15  # 假设是摄氏度
        else:
            temperature = T_si / T_factor[units['T']]
    
    return {
        'pressure': pressure,
        'volume': volume,
        'temperature': temperature,
        'quantity': quantity,
        'gas_constant': gas_constant
    }

def reaction_kinetics_solver(rate_constants, initial_concentrations, time_span, 
                             reaction_model=None, temperature_dependent=False, 
                             activation_energies=None, pre_exponential_factors=None, 
                             temperature=298.15):
    """
    求解化学反应动力学微分方程，模拟反应物浓度随时间的变化。
    
    Parameters:
    -----------
    rate_constants : list or dict
        反应速率常数，如果temperature_dependent=True，则忽略此参数
    initial_concentrations : list or dict
        各物质的初始浓度 (mol/L)
    time_span : tuple or list
        求解的时间范围 (s)，如 (0, 1000)
    reaction_model : callable, optional
        自定义反应模型函数，接收参数 (t, y, k)，其中t为时间，y为浓度向量，k为速率常数
        如果未提供，将使用默认的一级或二级反应模型
    temperature_dependent : bool, optional
        是否考虑温度对反应速率的影响，默认为False
    activation_energies : list or dict, optional
        各反应的活化能 (J/mol)，当temperature_dependent=True时需要
    pre_exponential_factors : list or dict, optional
        各反应的指前因子，当temperature_dependent=True时需要
    temperature : float, optional
        反应温度 (K)，当temperature_dependent=True时使用
        
    Returns:
    --------
    tuple
        (time_points, concentrations)，其中time_points为时间点数组，
        concentrations为对应的浓度数组，形状为(时间点数, 物质数)
        
    Examples:
    ---------
    >>> # 一级反应 A -> B，速率常数k=0.1 s^-1
    >>> times, concs = reaction_kinetics_solver([0.1], {'A': 1.0, 'B': 0.0}, (0, 50))
    """
    # 将字典形式的初始浓度转换为列表
    if isinstance(initial_concentrations, dict):
        species = list(initial_concentrations.keys())
        y0 = [initial_concentrations[s] for s in species]
    else:
        y0 = initial_concentrations
        species = [f"Species_{i}" for i in range(len(y0))]
    
    # 处理温度依赖的反应速率
    if temperature_dependent:
        if activation_energies is None or pre_exponential_factors is None:
            raise ValueError("温度依赖模式需要提供活化能和指前因子")
        
        R = 8.314  # 气体常数 J/(mol·K)
        
        if isinstance(activation_energies, dict):
            Ea_list = [activation_energies[k] for k in activation_energies]
            A_list = [pre_exponential_factors[k] for k in pre_exponential_factors]
        else:
            Ea_list = activation_energies
            A_list = pre_exponential_factors
        
        # 使用Arrhenius方程计算速率常数
        rate_constants = [A * np.exp(-Ea / (R * temperature)) 
                         for A, Ea in zip(A_list, Ea_list)]
    
    # 默认反应模型（如果未提供自定义模型）
    if reaction_model is None:
        # 简单的一级反应 A -> B 或二级反应 A + B -> C 的默认模型
        if len(y0) == 2:  # 假设是 A -> B
            def reaction_model(t, y, k):
                dAdt = -k[0] * y[0]
                dBdt = k[0] * y[0]
                return [dAdt, dBdt]
        elif len(y0) == 3:  # 假设是 A + B -> C
            def reaction_model(t, y, k):
                dAdt = -k[0] * y[0] * y[1]
                dBdt = -k[0] * y[0] * y[1]
                dCdt = k[0] * y[0] * y[1]
                return [dAdt, dBdt, dCdt]
        else:
            raise ValueError("默认反应模型仅支持一级反应(A->B)或二级反应(A+B->C)，请提供自定义反应模型")
    
    # 定义ODE求解器使用的函数
    def system(t, y):
        return reaction_model(t, y, rate_constants)
    
    # 求解ODE
    solution = solve_ivp(system, time_span, y0, method='RK45', 
                         dense_output=True, rtol=1e-6, atol=1e-9)
    
    # 生成更密集的时间点以获得平滑的曲线
    t_dense = np.linspace(time_span[0], time_span[1], 1000)
    y_dense = solution.sol(t_dense)
    
    # 创建结果数据框
    result_df = pd.DataFrame({
        'Time': t_dense,
        **{species[i]: y_dense[i] for i in range(len(species))}
    })
    
    return t_dense, y_dense, result_df, species

def quantum_energy_levels(system_type, parameters, n_levels=5):
    """
    计算量子系统的能级。
    
    Parameters:
    -----------
    system_type : str
        量子系统类型，可选值：'particle_in_box', 'harmonic_oscillator', 'hydrogen_atom'
    parameters : dict
        系统参数，根据system_type不同而不同：
        - 'particle_in_box': {'mass': 粒子质量(kg), 'box_length': 盒长(m)}
        - 'harmonic_oscillator': {'mass': 粒子质量(kg), 'force_constant': 力常数(N/m)}
        - 'hydrogen_atom': {'reduced_mass': 约化质量(kg)}
    n_levels : int, optional
        计算的能级数量，默认为5
        
    Returns:
    --------
    numpy.ndarray
        能级值数组 (J)
        
    Examples:
    ---------
    >>> # 计算电子在1纳米盒中的能级
    >>> energies = quantum_energy_levels('particle_in_box', 
    ...                                 {'mass': 9.11e-31, 'box_length': 1e-9}, 
    ...                                 n_levels=3)
    """
    h = 6.626e-34  # 普朗克常数 (J·s)
    hbar = h / (2 * np.pi)  # 约化普朗克常数
    e = 1.602e-19  # 电子电荷 (C)
    epsilon_0 = 8.854e-12  # 真空介电常数
    
    energy_levels = np.zeros(n_levels)
    
    if system_type == 'particle_in_box':
        # E_n = (n²π²ħ²)/(2mL²)
        m = parameters['mass']
        L = parameters['box_length']
        for n in range(1, n_levels + 1):
            energy_levels[n-1] = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    
    elif system_type == 'harmonic_oscillator':
        # E_n = (n + 1/2)ħω, ω = √(k/m)
        m = parameters['mass']
        k = parameters['force_constant']
        omega = np.sqrt(k / m)
        for n in range(n_levels):
            energy_levels[n] = (n + 0.5) * hbar * omega
    
    elif system_type == 'hydrogen_atom':
        # E_n = -Ry/n², Ry = me⁴/(8ε₀²h²)
        mu = parameters.get('reduced_mass', 9.11e-31)  # 默认为电子质量
        for n in range(1, n_levels + 1):
            energy_levels[n-1] = -((mu * e**4) / (8 * epsilon_0**2 * h**2)) / (n**2)
    
    else:
        raise ValueError(f"不支持的量子系统类型: {system_type}")
    
    return energy_levels

def thermodynamic_properties(temperature, pressure, volume=None, moles=1.0, 
                             gas_constant=8.314, is_ideal_gas=True, 
                             van_der_waals_params=None):
    """
    计算热力学系统的基本性质。
    
    Parameters:
    -----------
    temperature : float
        系统温度 (K)
    pressure : float
        系统压力 (Pa)
    volume : float, optional
        系统体积 (m³)，如果为None且is_ideal_gas=True，将使用理想气体状态方程计算
    moles : float, optional
        物质的量 (mol)，默认为1.0
    gas_constant : float, optional
        气体常数，默认为8.314 J/(mol·K)
    is_ideal_gas : bool, optional
        是否将系统视为理想气体，默认为True
    van_der_waals_params : dict, optional
        范德华参数，当is_ideal_gas=False时使用，格式为{'a': a值, 'b': b值}
        
    Returns:
    --------
    dict
        包含热力学性质的字典，包括：
        - 'volume': 体积 (m³)
        - 'internal_energy': 内能 (J)
        - 'enthalpy': 焓 (J)
        - 'entropy': 熵 (J/K)
        - 'gibbs_energy': 吉布斯自由能 (J)
        - 'helmholtz_energy': 亥姆霍兹自由能 (J)
    """
    properties = {}
    
    # 计算或使用给定的体积
    if volume is None:
        if is_ideal_gas:
            volume = (moles * gas_constant * temperature) / pressure
        elif van_der_waals_params is not None:
            # 使用范德华方程求解体积
            a = van_der_waals_params['a']
            b = van_der_waals_params['b']
            
            # 定义范德华方程
            def van_der_waals_eq(v):
                return (gas_constant * temperature / (v - b * moles) - a * moles**2 / v**2) - pressure
            
            # 初始猜测值（略大于b*moles）
            initial_guess = b * moles * 1.5
            
            # 求解
            volume = fsolve(van_der_waals_eq, initial_guess)[0]
        else:
            raise ValueError("非理想气体计算需要提供van_der_waals_params参数")
    
    properties['volume'] = volume
    
    # 计算内能
    if is_ideal_gas:
        # 理想单原子气体的内能 U = (3/2)nRT
        properties['internal_energy'] = 1.5 * moles * gas_constant * temperature
    else:
        # 范德华气体的内能包括分子间作用力的贡献
        a = van_der_waals_params['a']
        properties['internal_energy'] = 1.5 * moles * gas_constant * temperature - a * moles**2 / volume
    
    # 计算焓 H = U + PV
    properties['enthalpy'] = properties['internal_energy'] + pressure * volume
    
    # 计算熵（这里使用理想气体的近似公式）
    # 对于理想气体，S = nR*ln(V) + nR*ln(T) + const
    # 这里省略常数项，只计算相对熵变化
    properties['entropy'] = moles * gas_constant * (np.log(volume / moles) + 2.5 * np.log(temperature))
    
    # 计算吉布斯自由能 G = H - TS
    properties['gibbs_energy'] = properties['enthalpy'] - temperature * properties['entropy']
    
    # 计算亥姆霍兹自由能 A = U - TS
    properties['helmholtz_energy'] = properties['internal_energy'] - temperature * properties['entropy']
    
    return properties

def phase_equilibrium_calculator(components, temperature, pressure=101325, 
                                 interaction_parameters=None, method='raoult'):
    """
    计算多组分系统的相平衡。
    
    Parameters:
    -----------
    components : list of dict
        组分列表，每个组分为一个字典，包含：
        - 'name': 组分名称
        - 'mole_fraction': 初始摩尔分数
        - 'antoine_coefficients': Antoine方程系数 [A, B, C]
    temperature : float
        系统温度 (K)
    pressure : float, optional
        系统压力 (Pa)，默认为1个标准大气压
    interaction_parameters : dict, optional
        组分间相互作用参数，用于非理想溶液模型
    method : str, optional
        计算方法，可选值：'raoult'（拉乌尔定律）, 'modified_raoult'（修正的拉乌尔定律）
        
    Returns:
    --------
    dict
        相平衡计算结果，包括：
        - 'vapor_phase': 气相组成
        - 'liquid_phase': 液相组成
        - 'k_values': 各组分的气液分配系数
        - 'bubble_point': 泡点温度 (K)
        - 'dew_point': 露点温度 (K)
    """
    result = {
        'vapor_phase': {},
        'liquid_phase': {},
        'k_values': {}
    }
    
    # 计算各组分的饱和蒸气压
    for comp in components:
        name = comp['name']
        A, B, C = comp['antoine_coefficients']
        # Antoine方程: log10(P_sat) = A - B/(T + C)
        # 其中P_sat单位通常为mmHg，需要转换为Pa
        log_p_sat = A - B / (temperature + C)
        p_sat = 10**log_p_sat * 133.322  # 转换mmHg到Pa
        
        # 计算气液分配系数 K = y/x = P_sat/P (理想情况下)
        k_value = p_sat / pressure
        result['k_values'][name] = k_value
        
        # 根据拉乌尔定律计算气相组成
        x_i = comp['mole_fraction']  # 液相摩尔分数
        
        if method == 'raoult':
            # 拉乌尔定律: y_i * P = x_i * P_sat_i
            y_i = x_i * k_value
        elif method == 'modified_raoult':
            # 修正的拉乌尔定律考虑活度系数
            if interaction_parameters is None:
                raise ValueError("修正的拉乌尔定律需要提供interaction_parameters")
            
            # 简化的活度系数计算（这里使用一个虚构的计算方式）
            gamma_i = 1.0  # 实际应用中需要根据具体模型计算
            y_i = x_i * gamma_i * k_value
        else:
            raise ValueError(f"不支持的计算方法: {method}")
        
        result['vapor_phase'][name] = y_i
        result['liquid_phase'][name] = x_i
    
    # 归一化气相组成
    total_y = sum(result['vapor_phase'].values())
    for name in result['vapor_phase']:
        result['vapor_phase'][name] /= total_y
    
    # 计算泡点和露点（简化计算）
    # 在实际应用中，这需要迭代求解
    
    # 泡点温度估计函数
    def bubble_point_eq(T):
        total = 0
        for comp in components:
            name = comp['name']
            A, B, C = comp['antoine_coefficients']
            x_i = comp['mole_fraction']
            log_p_sat = A - B / (T + C)
            p_sat = 10**log_p_sat * 133.322
            total += x_i * p_sat
        return total - pressure
    
    # 露点温度估计函数
    def dew_point_eq(T):
        total = 0
        for comp in components:
            name = comp['name']
            A, B, C = comp['antoine_coefficients']
            y_i = result['vapor_phase'][name]
            log_p_sat = A - B / (T + C)
            p_sat = 10**log_p_sat * 133.322
            total += y_i / p_sat
        return 1/pressure - total
    
    # 使用当前温度作为初始猜测值
    try:
        result['bubble_point'] = fsolve(bubble_point_eq, temperature)[0]
        result['dew_point'] = fsolve(dew_point_eq, temperature)[0]
    except:
        result['bubble_point'] = None
        result['dew_point'] = None
        print("警告：泡点和露点计算未收敛")
    
    return result

def main():
    """
    主函数：演示如何使用工具函数求解物理化学问题
    """
    print("物理化学计算工具包演示")
    print("=" * 50)
    
    # 示例1：理想气体计算 - 解决氮氧化物(NO)样品的物质的量
    print("\n示例1：理想气体计算")
    print("-" * 50)
    
    # 问题参数
    pressure = 24.5  # kPa
    volume = 250.0  # cm³
    temperature = 19.5  # °C
    
    # 使用理想气体状态方程计算
    result = ideal_gas_calculation(
        pressure=pressure, 
        volume=volume, 
        temperature=temperature, 
        quantity=None,
        units={'P': 'kPa', 'V': 'cm³', 'T': '°C', 'n': 'mol', 'R': 'J/(mol·K)'}
    )
    
    print(f"问题：一个体积为{volume} cm³的容器中收集了一些氮氧化物(NO)样品。")
    print(f"在{temperature}°C时，测得压力为{pressure} kPa。")
    print(f"计算结果：容器中NO的物质的量为 {result['quantity']:.5f} mol")
    
    # 示例2：化学反应动力学 - 一级反应A→B的模拟
    print("\n示例2：化学反应动力学模拟")
    print("-" * 50)
    
    # 反应参数
    k = 0.05  # 反应速率常数，单位：s^-1
    initial_conc = {'A': 1.0, 'B': 0.0}  # 初始浓度，单位：mol/L
    time_span = (0, 100)  # 时间范围，单位：s
    
    # 模拟反应
    times, concs, df, species = reaction_kinetics_solver(
        [k], initial_conc, time_span
    )
    
    print(f"模拟一级反应 A → B，速率常数k = {k} s^-1")
    print(f"初始浓度：[A] = {initial_conc['A']} mol/L, [B] = {initial_conc['B']} mol/L")
    print(f"反应100秒后：[A] = {concs[0, -1]:.4f} mol/L, [B] = {concs[1, -1]:.4f} mol/L")
    
    # 绘制反应动力学曲线
    plt.figure(figsize=(10, 6))
    for i, specie in enumerate(species):
        plt.plot(times, concs[i], label=f'[{specie}]')
    
    plt.xlabel('时间 (s)')
    plt.ylabel('浓度 (mol/L)')
    plt.title('一级反应 A → B 的反应动力学曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('./images/reaction_kinetics.png', dpi=300, bbox_inches='tight')
    
    # 示例3：量子化学计算 - 一维盒中的粒子能级
    print("\n示例3：量子化学计算")
    print("-" * 50)
    
    # 参数设置
    box_length = 1e-9  # 盒长，单位：m (1 nm)
    electron_mass = 9.11e-31  # 电子质量，单位：kg
    n_levels = 5  # 计算的能级数
    
    # 计算能级
    energies = quantum_energy_levels(
        'particle_in_box',
        {'mass': electron_mass, 'box_length': box_length},
        n_levels
    )
    
    # 转换为电子伏特
    energies_eV = energies / 1.602e-19
    
    print(f"一维盒(长度={box_length*1e9} nm)中电子的能级：")
    for n, energy in enumerate(energies_eV, 1):
        print(f"n = {n}: E = {energy:.4f} eV")
    
    # 绘制能级图
    plt.figure(figsize=(8, 6))
    for n, energy in enumerate(energies_eV, 1):
        plt.plot([0, 1], [energy, energy], 'b-', linewidth=2)
        plt.text(1.1, energy, f'n = {n}, E = {energy:.2f} eV', va='center')
    
    plt.xlim(-0.5, 3)
    plt.ylim(0, energies_eV[-1] * 1.2)
    plt.title('一维无限深势阱中的粒子能级')
    plt.ylabel('能量 (eV)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks([])
    plt.savefig('./images/quantum_energy_levels.png', dpi=300, bbox_inches='tight')
    
    # 示例4：热力学性质计算
    print("\n示例4：热力学性质计算")
    print("-" * 50)
    
    # 参数设置
    T = 298.15  # 温度，单位：K
    P = 101325  # 压力，单位：Pa
    n = 1.0     # 物质的量，单位：mol
    
    # 计算理想气体的热力学性质 
if __name__ == "__main__":
    main()