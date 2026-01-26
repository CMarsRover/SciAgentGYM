# Filename: thermodynamics_solver.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import os

def calculate_internal_energy_change(heat_flow, work_done):
    """
    计算热力学系统内能变化，基于热力学第一定律。
    
    热力学第一定律表明：ΔU = Q + W，其中ΔU是内能变化，Q是热量，W是对系统做的功。
    注意：在热力学中，对系统做的功为正，系统对外做的功为负。
    
    Parameters:
    -----------
    heat_flow : float
        流入系统的热量，单位为焦耳(J)。正值表示热量流入系统，负值表示热量流出系统。
    work_done : float
        对系统做的功，单位为焦耳(J)。正值表示对系统做功，负值表示系统对外做功。
    
    Returns:
    --------
    float
        系统内能的变化量，单位为焦耳(J)。
    """
    delta_u = heat_flow + work_done
    return delta_u

def calculate_work_pv_path(p_values, v_values):
    """
    计算PV图上路径的功。
    
    在PV图上，功等于曲线下的面积：W = ∫P·dV
    注意：这里计算的是系统对外做的功，如果要转换为对系统做的功，需要取负值。
    
    Parameters:
    -----------
    p_values : array_like
        压力值数组，单位为帕斯卡(Pa)
    v_values : array_like
        体积值数组，单位为立方米(m³)
    
    Returns:
    --------
    float
        系统对外做的功，单位为焦耳(J)
    """
    # 使用数值积分计算功
    work = simpson(p_values, v_values)
    return work

def simulate_ideal_gas_process(process_type, initial_state, final_state, gas_constant=8.314, moles=1.0):
    """
    模拟理想气体的热力学过程，计算热量、功和内能变化。
    
    Parameters:
    -----------
    process_type : str
        过程类型，可选值: 'isothermal'(等温), 'isobaric'(等压), 'isochoric'(等容), 'adiabatic'(绝热)
    initial_state : tuple
        初始状态，格式为(p, v, t)，分别表示压力(Pa)、体积(m³)和温度(K)
    final_state : tuple
        最终状态，格式为(p, v, t)，分别表示压力(Pa)、体积(m³)和温度(K)
        注意：根据过程类型，某些参数可能会被重新计算
    gas_constant : float, optional
        气体常数，默认为8.314 J/(mol·K)
    moles : float, optional
        气体的摩尔数，默认为1.0 mol
    
    Returns:
    --------
    dict
        包含计算结果的字典，键包括：'heat'(热量), 'work'(系统对外做功), 'delta_u'(内能变化)
    """
    p1, v1, t1 = initial_state
    p2, v2, t2 = final_state
    
    # 确保状态变量的一致性
    if process_type == 'isothermal':
        # 等温过程: T1 = T2, PV = constant
        t2 = t1
        if p2 is None:
            p2 = p1 * v1 / v2
        elif v2 is None:
            v2 = p1 * v1 / p2
    
    elif process_type == 'isobaric':
        # 等压过程: P1 = P2
        p2 = p1
        if v2 is None and t2 is not None:
            v2 = v1 * t2 / t1
        elif t2 is None and v2 is not None:
            t2 = t1 * v2 / v1
    
    elif process_type == 'isochoric':
        # 等容过程: V1 = V2
        v2 = v1
        if p2 is None and t2 is not None:
            p2 = p1 * t2 / t1
        elif t2 is None and p2 is not None:
            t2 = t1 * p2 / p1
    
    elif process_type == 'adiabatic':
        # 绝热过程: PV^γ = constant, γ = Cp/Cv (比热容比)
        gamma = 1.4  # 对于双原子气体
        if p2 is None:
            p2 = p1 * (v1 / v2) ** gamma
        elif v2 is None:
            v2 = v1 * (p1 / p2) ** (1 / gamma)
        t2 = t1 * (p2 * v2) / (p1 * v1)
    
    # 计算热力学量
    # 内能变化 (理想气体内能只与温度有关)
    delta_u = moles * (3/2) * gas_constant * (t2 - t1)  # 单原子气体
    
    # 计算功
    if process_type == 'isothermal':
        work = moles * gas_constant * t1 * np.log(v2 / v1)
    elif process_type == 'isobaric':
        work = p1 * (v2 - v1)
    elif process_type == 'isochoric':
        work = 0
    elif process_type == 'adiabatic':
        work = (p1 * v1 - p2 * v2) / (gamma - 1)
    
    # 计算热量 (根据热力学第一定律)
    if process_type == 'adiabatic':
        heat = 0
    else:
        heat = delta_u - (-work)  # 注意：work是系统对外做功，对系统做功为-work
    
    return {
        'initial_state': (p1, v1, t1),
        'final_state': (p2, v2, t2),
        'heat': heat,
        'work': work,  # 系统对外做功
        'work_on_system': -work,  # 对系统做功
        'delta_u': delta_u
    }

def analyze_cyclic_process(vertices, process_types=None):
    """
    分析PV图上的循环过程，计算总热量、总功和总内能变化。
    
    Parameters:
    -----------
    vertices : list of tuples
        循环的顶点列表，每个顶点是一个(p, v, t)元组，表示压力、体积和温度
        如果温度未知，可以设为None
    process_types : list of str, optional
        每段路径的过程类型列表，可选值包括：'isothermal', 'isobaric', 'isochoric', 'adiabatic'
        如果为None，则默认所有路径为一般过程
    
    Returns:
    --------
    dict
        包含循环分析结果的字典，键包括：'net_heat'(净热量), 'net_work'(净功), 'net_delta_u'(净内能变化),
        'efficiency'(热效率，如果适用), 'processes'(各个过程的详细结果)
    """
    n = len(vertices)
    if process_types is None:
        process_types = ['general'] * n
    
    # 确保process_types长度与顶点数匹配
    if len(process_types) < n:
        process_types.extend(['general'] * (n - len(process_types)))
    
    processes = []
    total_heat = 0
    total_work = 0
    
    # 分析每个过程
    for i in range(n):
        start_vertex = vertices[i]
        end_vertex = vertices[(i+1) % n]
        process_type = process_types[i]
        
        # 对于一般过程，直接计算功
        if process_type == 'general':
            p1, v1, t1 = start_vertex
            p2, v2, t2 = end_vertex
            
            # 简化为线性路径计算功
            avg_p = (p1 + p2) / 2
            work = avg_p * (v2 - v1)
            
            # 如果温度已知，计算内能变化
            if t1 is not None and t2 is not None:
                delta_u = 1.0 * (3/2) * 8.314 * (t2 - t1)  # 假设为单原子气体，1摩尔
            else:
                delta_u = None
                
            # 如果内能变化已知，计算热量
            if delta_u is not None:
                heat = delta_u - (-work)
            else:
                heat = None
                
            process_result = {
                'type': 'general',
                'start': start_vertex,
                'end': end_vertex,
                'work': work,
                'work_on_system': -work,
                'delta_u': delta_u,
                'heat': heat
            }
        else:
            # 使用热力学过程模拟函数
            process_result = simulate_ideal_gas_process(
                process_type, 
                start_vertex, 
                end_vertex
            )
        
        processes.append(process_result)
        
        if process_result['heat'] is not None:
            total_heat += process_result['heat']
        if process_result['work'] is not None:
            total_work += process_result['work']
    
    # 循环过程中内能净变化为零
    net_delta_u = 0
    
    # 计算热效率（如果是热机）
    efficiency = None
    if total_heat > 0 and total_work < 0:  # 热机条件：吸热为正，对外做功为负
        efficiency = -total_work / total_heat
    
    return {
        'net_heat': total_heat,
        'net_work': total_work,
        'net_work_on_system': -total_work,
        'net_delta_u': net_delta_u,
        'efficiency': efficiency,
        'processes': processes
    }

def visualize_pv_diagram(vertices, process_types=None, title="P-V Diagram", save_path=None, show_work=True):
    """
    可视化PV图，显示热力学过程路径。
    
    Parameters:
    -----------
    vertices : list of tuples
        路径的顶点列表，每个顶点是一个(p, v)或(p, v, t)元组
    process_types : list of str, optional
        每段路径的过程类型列表
    title : str, optional
        图表标题
    save_path : str, optional
        保存图像的路径，如果为None则不保存
    show_work : bool, optional
        是否显示功的计算结果
    
    Returns:
    --------
    None
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取P和V值
    points = [(p, v) for p, v, *_ in vertices]
    points.append(points[0])  # 闭合路径
    
    p_values, v_values = zip(*points)
    
    plt.figure(figsize=(10, 8))
    
    # 绘制路径
    plt.plot(v_values, p_values, 'o-', linewidth=2)
    
    # 标记顶点
    for i, (p, v) in enumerate(points[:-1]):
        plt.annotate(f"{i+1}", (v, p), fontsize=12, 
                     xytext=(5, 5), textcoords='offset points')
    
    # 填充区域表示功
    if show_work:
        plt.fill(v_values, p_values, alpha=0.2)
    
    plt.xlabel('体积 V (m³)', fontsize=12)
    plt.ylabel('压力 P (Pa)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    
    # 保存图像
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()

def main():
    """
    主函数：演示如何使用工具函数求解热力学问题
    """
    print("热力学计算工具演示")
    print("-" * 50)
    
    # 示例1：计算内能变化
    print("示例1: 计算内能变化")
    heat_flow = 425  # J
    work_done = -175  # J (负值表示系统对外做功)
    delta_u = calculate_internal_energy_change(heat_flow, work_done)
    print(f"热量 Q = {heat_flow} J")
    print(f"对系统做功 W = {work_done} J")
    print(f"内能变化 ΔU = {delta_u} J")
    print("-" * 50)
    
    # 示例2：分析PV图上的循环过程
    print("示例2: 分析PV图上的循环过程")
    
    # 定义PV图上的顶点 (P, V, T)
    # 这里我们使用相对单位，实际应用中应使用适当的物理单位
    vertices = [
        (100, 1, 300),    # 点1
        (100, 2, 600),    # 点2
        (200, 2, 1200),   # 点3
        (200, 1, 600)     # 点4
    ]
    
    # 定义每段路径的过程类型
    process_types = ['isobaric', 'isochoric', 'isobaric', 'isochoric']
    
    # 分析循环过程
    result = analyze_cyclic_process(vertices, process_types)
    
    # 输出结果
    print("循环过程分析结果:")
    print(f"净热量: {result['net_heat']:.2f} J")
    print(f"净功 (系统对外): {result['net_work']:.2f} J")
   
    
    print(f"净内能变化: {result['net_delta_u']:.2f} J")
    
    if result['efficiency'] is not None:
        print(f"热效率: {result['efficiency']*100:.2f}%")
    
    # 可视化PV图
    save_path = "./images/pv_cycle_diagram.png"
    visualize_pv_diagram(vertices, process_types, 
                         title="热力学循环过程 P-V 图", 
                         save_path=save_path)
    
    print("-" * 50)
    
    # 示例3：模拟理想气体的不同热力学过程
    print("示例3: 模拟理想气体的不同热力学过程")
    
    # 初始状态
    p1 = 101325  # Pa (1 atm)
    v1 = 0.0224  # m³ (1 mol理想气体在标准状态下的体积)
    t1 = 300     # K
    
    # 等温膨胀
    isothermal_result = simulate_ideal_gas_process(
        'isothermal',
        (p1, v1, t1),
        (None, v1*2, None)  # 体积扩大到2倍
    )
    
    # 等压膨胀
    isobaric_result = simulate_ideal_gas_process(
        'isobaric',
        (p1, v1, t1),
        (None, v1*1.5, None)  # 体积扩大到1.5倍
    )
    
    # 等容加热
    isochoric_result = simulate_ideal_gas_process(
        'isochoric',
        (p1, v1, t1),
        (None, None, t1*1.2)  # 温度升高20%
    )
    
    # 绝热膨胀
    adiabatic_result = simulate_ideal_gas_process(
        'adiabatic',
        (p1, v1, t1),
        (None, v1*1.5, None)  # 体积扩大到1.5倍
    )
    
    # 输出结果
    processes = {
        "等温过程": isothermal_result,
        "等压过程": isobaric_result,
        "等容过程": isochoric_result,
        "绝热过程": adiabatic_result
    }
    
    for name, result in processes.items():
        print(f"\n{name}结果:")
        p1, v1, t1 = result['initial_state']
        p2, v2, t2 = result['final_state']
        print(f"初态: P={p1:.2f} Pa, V={v1:.6f} m³, T={t1:.2f} K")
        print(f"末态: P={p2:.2f} Pa, V={v2:.6f} m³, T={t2:.2f} K")
        print(f"热量: Q={result['heat']:.2f} J")
        print(f"功 (系统对外): W={result['work']:.2f} J")
        print(f"内能变化: ΔU={result['delta_u']:.2f} J")
    
    # 示例4：解决科学问题 - 计算PV图上路径的内能变化
    print("\n" + "-" * 50)
    print("示例4: 解决科学问题 - 计算PV图上路径的内能变化")
    print("问题: 当气体在PV图上沿路径123运动时，通过热量流入系统的能量为425 J，")
    print("      对气体做功为-175 J。计算系统内能的变化。")
    
    # 解决方案
    q = 425  # J
    w = -175  # J
    delta_u = calculate_internal_energy_change(q, w)
    
    print("\n解题过程:")
    print("根据热力学第一定律: ΔU = Q + W")
    print(f"其中 Q = {q} J (热量流入为正)")
    print(f"    W = {w} J (对系统做功为正，这里是负值表示系统对外做功)")
    print(f"因此 ΔU = {q} J + ({w} J) = {delta_u} J")
    print(f"\n答案: 系统内能变化为 {delta_u} J")
    
    # 创建一个简化的PV图来说明问题
    # 这里我们创建一个示意图，不需要精确的数值
    simple_vertices = [
        (1, 1, None),  # 点1
        (2, 1, None),  # 点2
        (2, 2, None),  # 点3
        (1, 2, None)   # 点4 (为了闭合图形)
    ]
    
    # 可视化简化的PV图
    save_path = "./images/example_pv_path.png"
    visualize_pv_diagram(simple_vertices[:3], 
                         title="PV图上的路径123示意图",
                         save_path=save_path,
                         show_work=False)

if __name__ == "__main__":
    main()