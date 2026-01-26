import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from scipy.constants import c, speed_of_light  # 光速
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# ========================================
# 1. 核心数学计算模块 (Math Functions)
# ========================================

def calculate_doppler_shift(freq_emit, freq_received):
    """
    计算多普勒频移
    
    Parameters:
    -----------
    freq_emit : float
        发射频率，单位为Hz
    freq_received : float
        接收频率，单位为Hz
        
    Returns:
    --------
    float
        多普勒频移，单位为Hz
    """
    return freq_received - freq_emit


def calculate_blood_velocity(freq_emit, freq_received, v_sound=1540):
    """
    计算血流速度（doppler_blood_velocity的别名）
    
    Parameters:
    -----------
    freq_emit : float
        发射频率，单位为Hz
    freq_received : float
        接收频率，单位为Hz
    v_sound : float, optional
        声波在组织中的传播速度，默认值为1540 m/s
        
    Returns:
    --------
    float
        血流速度，单位为m/s
    """
    return doppler_blood_velocity(freq_emit, freq_received, v_sound)


def analyze_doppler_spectrum(frequencies, amplitudes):
    """
    分析多普勒频谱
    
    Parameters:
    -----------
    frequencies : array-like
        频率数组
    amplitudes : array-like
        幅度数组
        
    Returns:
    --------
    dict
        包含频谱分析结果的字典
    """
    max_amplitude_idx = np.argmax(amplitudes)
    peak_frequency = frequencies[max_amplitude_idx]
    peak_amplitude = amplitudes[max_amplitude_idx]
    
    return {
        'peak_frequency': peak_frequency,
        'peak_amplitude': peak_amplitude,
        'spectrum_width': np.std(amplitudes),
        'mean_frequency': np.mean(frequencies)
    }


def doppler_blood_velocity(freq_emit, freq_received, v_sound=1540):
    """
    计算多普勒血流速度
    基于多普勒效应原理，通过发射频率和接收频率的差值计算血流速度。
    适用于医学超声多普勒血流检测。
    
    Parameters:
    -----------
    freq_emit : float
        发射频率，单位为Hz。超声探头发射的原始频率
    freq_received : float
        接收频率，单位为Hz。从流动血液反射回来的频率
    v_sound : float, optional
        声波在组织中的传播速度，默认值为1540 m/s（人体软组织的典型声速）
    
    Returns:
    --------
    float
        血流速度，单位为m/s。正值表示血流朝向探头，负值表示血流远离探头
    
    Formula:
    --------
    v_blood = (Δf * v_sound) / (2 * f_emit + Δf)
    其中 Δf = f_received - f_emit
    
    Example:
    --------
    >>> v = doppler_blood_velocity(5e6, 5.001e6, 1540)
    >>> print(f"血流速度: {v:.3f} m/s")
    """
    delta_f = freq_received - freq_emit
    v_blood = (delta_f * v_sound) / (2 * freq_emit + delta_f)
    return v_blood

def thin_film_interference_thickness(wavelength, n, m=1, theta=0):
    """
    计算薄膜干涉条件下的薄膜厚度
    
    基于薄膜干涉理论，计算在给定波长、折射率、干涉级次和入射角条件下，
    产生相长干涉或相消干涉所需的薄膜厚度。
    
    Parameters:
    -----------
    wavelength : float
        入射光波长，单位为nm或m。用于干涉计算的光波波长
    n : float
        薄膜材料的折射率。薄膜相对于周围介质的折射率比值
    m : int, optional
        干涉级次，默认为1。表示第m级干涉（m=1为一级干涉，m=2为二级干涉等）
    theta : float, optional
        入射角，单位为弧度，默认为0（垂直入射）。光线与薄膜法线的夹角
    
    Returns:
    --------
    float
        薄膜厚度，单位与wavelength相同。产生指定干涉条件所需的最小厚度
    
    Formula:
    --------
    d = (m * λ) / (2 * n * cos(θ))
    其中：
    - d: 薄膜厚度
    - m: 干涉级次
    - λ: 波长
    - n: 折射率
    - θ: 入射角
    
    Notes:
    ------
    - 当m=1时，计算的是产生一级相长干涉的最小厚度
    - 当入射角θ=0时，公式简化为 d = (m * λ) / (2 * n)
    - 适用于增透膜、反射膜等光学薄膜设计
    
    Example:
    --------
    >>> # 计算550nm绿光在折射率1.5的薄膜中垂直入射时的一级干涉厚度
    >>> thickness = thin_film_interference_thickness(550e-9, 1.5, 1, 0)
    >>> print(f"薄膜厚度: {thickness*1e9:.2f} nm")
    """
    cos_theta = np.cos(theta)
    d = (m * wavelength) / (2 * n * cos_theta)
    return d

def maxwell_field_relation(changing_E=None, changing_B=None):

    """
    基于麦克斯韦方程计算电磁场关系
    
    Parameters:
    -----------
    changing_E : float
        changing_E参数，默认值为None
    changing_B : float
        changing_B参数，默认值为None
    
    Returns:
    --------
    float
        计算结果
    """
    
    result = {
        'produces_B': False,
        'produces_E': False,
        'explanation': ''
    }
    
    if changing_E is not None and changing_E:
        result['produces_B'] = True
        result['explanation'] += "变化电场产生磁场（安培-麦克斯韦定律）。"
    
    if changing_B is not None and changing_B:
        result['produces_E'] = True
        result['explanation'] += "变化磁场产生电场（法拉第电磁感应定律）。"
    
    return result

def uv_remote_sensing_capability():

    """
    计算uv_remote_sensing_capability相关结果
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    float
        计算结果
    """
    
    return {
        'suitable': False,
        'reason': "紫外线易被大气（尤其是臭氧层）吸收，穿透能力弱，"
                  "不适合用于遥感监测地表活动。"
                  "通常使用红外线进行此类遥感。"
    }

# ========================================
# 2. 可视化函数 (Visualization Functions)
# ========================================

def visualize_doppler_ultrasound():

    """
    基于多普勒效应计算血流速度
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    object
        图表对象
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 背景
    ax.add_patch(Rectangle((0, 0.4), 10, 0.2, color='skyblue', alpha=0.5, label='Blood vessel'))
    ax.text(5, 0.7, 'Tissue', fontsize=12, ha='center')
    ax.text(5, 0.5, 'Blood Flow →', fontsize=12, ha='center', color='red')
    
    # 超声发射与接收
    ax.arrow(1, 0.8, 0, -0.2, head_width=0.1, fc='green', label='Emit')
    ax.arrow(1, 0.2, 0, 0.2, head_width=0.1, fc='orange', label='Receive')
    
    # 频率标注
    ax.text(0.5, 0.9, 'f₀ (emit)', color='green', fontsize=10)
    ax.text(0.5, 0.1, 'f (received)', color='orange', fontsize=10)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Doppler Ultrasound for Blood Flow Measurement")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_thin_film_interference():

    """
    计算薄膜干涉相关参数
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    object
        图表对象
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 薄膜结构
    ax.add_patch(Rectangle((1, 0.5), 8, 0.1, color='lightgray', label='Thin Film'))
    ax.add_patch(Rectangle((1, 0.4), 8, 0.1, color='white', label='Substrate'))
    
    # 入射光
    ax.arrow(3, 0.8, 0, -0.2, head_width=0.1, fc='blue')
    ax.text(3.1, 0.7, 'Incident Light', color='blue')
    
    # 反射光束1（空气-膜界面）
    ax.arrow(3, 0.5, 0.5, 0.2, head_width=0.05, fc='red')
    ax.text(3.6, 0.7, 'Reflected 1', color='red')
    
    # 透射与二次反射
    ax.arrow(4, 0.5, 0, -0.1, head_width=0.05, fc='blue')
    ax.arrow(4, 0.4, 0, 0.1, head_width=0.05, fc='green')
    ax.arrow(4, 0.5, 0.5, 0.2, head_width=0.05, fc='green')
    ax.text(4.6, 0.7, 'Reflected 2', color='green')
    
    # 干涉结果
    ax.text(7, 0.8, 'Interference → Color Bands', fontsize=12, color='purple')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Thin-Film Interference on Insect Wings")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_maxwell_fields():

    """
    基于麦克斯韦方程计算电磁场关系
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    object
        图表对象
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 变化磁场产生电场
    ax.add_patch(Circle((2.5, 0.5), 0.3, color='red', alpha=0.3))
    ax.text(2.5, 0.5, '∂B/∂t', ha='center', va='center', fontsize=12, color='darkred')
    ax.add_patch(Circle((2.5, 0.5), 0.6, color='none', ec='blue', lw=2))
    ax.text(2.5, 1.2, 'Induced E', ha='center', color='blue')
    
    # 变化电场产生磁场
    ax.add_patch(Rectangle((6.5, 0.3), 0.6, 0.4, color='blue', alpha=0.3))
    ax.text(6.8, 0.5, '∂E/∂t', ha='center', va='center', fontsize=12, color='white')
    ax.add_patch(Rectangle((6.2, 0.0), 1.2, 0.2, color='none', ec='red', lw=2))
    ax.text(6.8, -0.1, 'Induced B', ha='center', color='red')
    
    ax.text(2.5, 0.1, 'Faraday Law', ha='center', fontsize=10)
    ax.text(6.8, 0.1, 'Ampere-Maxwell Law', ha='center', fontsize=10)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 1.5)
    ax.set_title("Maxwell's Equations: Changing Fields Produce Each Other")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_uv_remote_sensing():

    """
    生成可视化图表
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    object
        图表对象
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 地球与大气
    ax.add_patch(Circle((5, 1), 1, color='blue', alpha=0.5))
    ax.text(5, 1, 'Earth', ha='center', va='center', color='white', fontsize=12)
    
    # 大气层（臭氧吸收UV）
    ax.add_patch(Rectangle((3, 2), 4, 0.3, color='lightgreen', alpha=0.5))
    ax.text(5, 2.15, 'Ozone Layer (Absorbs UV)', ha='center', fontsize=10)
    
    # 紫外线入射被吸收
    ax.arrow(4, 3, 0.5, -0.8, head_width=0.1, fc='purple')
    ax.arrow(5, 3, 0.5, -0.8, head_width=0.1, fc='purple')
    ax.text(4.5, 3.1, 'UV Radiation', color='purple')
    
    # 红外线穿透（对比）
    ax.arrow(6, 3, 0.5, -1.5, head_width=0.1, fc='red', linestyle='--')
    ax.arrow(7, 3, 0.5, -1.5, head_width=0.1, fc='red', linestyle='--')
    ax.text(6.5, 3.1, 'IR Radiation (penetrates)', color='red')
    
    ax.text(5, -0.2, 'UV cannot reach ground sensors → Not suitable for remote sensing', 
            ha='center', fontsize=10, color='purple')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title("Why UV is NOT Suitable for Remote Sensing")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# ========================================
# 3. 综合分析函数 (Coding Functions)
# ========================================

def analyze_physics_statements():

    """
    执行分析计算
    
    Parameters:
    -----------
    无参数
    
    Returns:
    --------
    dict
        分析结果字典
    """
    
    results = {}
    
    # A. 多普勒测血流
    results['A'] = {
        'statement': "Emitting ultrasound... utilizes the Doppler effect.",
        'correct': True,
        'explanation': "正确。利用多普勒频移测量反射波频率变化，可计算血流速度。",
        'math_result': doppler_blood_velocity(5e6, 5.005e6),  # 示例：5MHz → 5.005MHz
        'visual_func': visualize_doppler_ultrasound
    }
    
    # B. 薄膜干涉
    results['B'] = {
        'statement': "Insect wings with colored bands due to thin-film interference.",
        'correct': True,
        'explanation': "正确。透明薄膜上下表面反射光干涉形成彩色条纹。",
        'math_result': thin_film_interference_thickness(550, 1.4),  # 绿光，折射率1.4
        'visual_func': visualize_thin_film_interference
    }
    
    # C. 电磁场互生
    results['C'] = {
        'statement': "Changing E always produces changing B, and vice versa.",
        'correct': False,
        'explanation': "错误。变化电场产生磁场，变化磁场产生电场，"
                       "但产生的场不一定是‘变化’的。且该描述忽略了位移电流等细节。",
        'math_result': maxwell_field_relation(changing_E=True, changing_B=True),
        'visual_func': visualize_maxwell_fields
    }
    
    # D. 紫外线遥感
    results['D'] = {
        'statement': "UV used to survey geothermal, water, forest fires from satellites.",
        'correct': False,
        'explanation': "错误。紫外线被大气强烈吸收，不适合遥感。"
                       "实际使用红外线进行此类监测。",
        'math_result': uv_remote_sensing_capability(),
        'visual_func': visualize_uv_remote_sensing
    }
    
    return results

# ========================================
# 使用示例（可取消注释运行）
# ========================================

if __name__ == "__main__":
    # 分析所有陈述
    analysis = analyze_physics_statements()
    
    print("✅ 正确的陈述是：")
    for key, res in analysis.items():
        if res['correct']:
            print(f"{key}. {res['explanation']}")
    
    # 可视化每个现象（依次展示）
    for key, res in analysis.items():
        print(f"--- 可视化 {key}: {res['statement']} ---")
        res['visual_func']()
