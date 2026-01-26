import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# =============================================
# 底层核心数学模块：Clausius-Clapeyron 方程
# =============================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
def clausius_clapeyron_temperature(
    P1: float,
    T1: float,
    P2: float,
    delta_H_vap: float,
    R: float = 8.314
) -> float:
    """
    基于克劳修斯-克拉佩龙方程计算温度
    
    根据已知的参考压力和温度，以及目标压力，计算对应的温度。
    适用于相变过程（如汽化、升华）的温度计算。
    
    Parameters:
    -----------
    P1 : float
        参考压力，单位为Pa。已知温度T1对应的压力
    T1 : float
        参考温度，单位为K。已知压力P1对应的温度
    P2 : float
        目标压力，单位为Pa。需要计算对应温度的压力
    delta_H_vap : float
        摩尔汽化焓，单位为J/mol。相变过程的焓变
    R : float, optional
        气体常数，默认值为8.314 J/(mol·K)
    
    Returns:
    --------
    float
        目标温度，单位为K。压力P2对应的温度
    
    Formula:
    --------
    ln(P2/P1) = (ΔH_vap/R) * (1/T1 - 1/T2)
    求解得：T2 = 1 / (1/T1 - (R/ΔH_vap) * ln(P2/P1))
    
    Example:
    --------
    >>> T2 = clausius_clapeyron_temperature(101325, 373.15, 230000, 40670)
    >>> print(f"压力锅最高温度: {T2-273.15:.1f}°C")
    """
    import math
    T2 = 1 / (1/T1 - (R / delta_H_vap) * math.log(P2 / P1))
    return T2

def clausius_clapeyron_pressure(
    P1: float,
    T1: float,
    T2: float,
    delta_H_vap: float,
    R: float = 8.314
) -> float:
    """
    基于克劳修斯-克拉佩龙方程计算压力
    
    根据已知的参考压力和温度，以及目标温度，计算对应的压力。
    适用于相变过程（如汽化、升华）的压力计算。
    
    Parameters:
    -----------
    P1 : float
        参考压力，单位为Pa。已知温度T1对应的压力
    T1 : float
        参考温度，单位为K。已知压力P1对应的温度
    T2 : float
        目标温度，单位为K。需要计算对应压力的温度
    delta_H_vap : float
        摩尔汽化焓，单位为J/mol。相变过程的焓变
    R : float, optional
        气体常数，默认值为8.314 J/(mol·K)
    
    Returns:
    --------
    float
        目标压力，单位为Pa。温度T2对应的压力
    
    Formula:
    --------
    ln(P2/P1) = (ΔH_vap/R) * (1/T1 - 1/T2)
    求解得：P2 = P1 * exp((ΔH_vap/R) * (1/T1 - 1/T2))
    
    Example:
    --------
    >>> P2 = clausius_clapeyron_pressure(101325, 373.15, 398.15, 40670)
    >>> print(f"124°C时的蒸汽压: {P2/1000:.1f} kPa")
    """
    import math
    ln_ratio = - (delta_H_vap / R) * (1/T2 - 1/T1)
    P2 = P1 * math.exp(ln_ratio)
    return P2

# =============================================
# 通用可视化函数
# =============================================

def plot_clausius_clapeyron(
    P1: float,
    T1: float,
    delta_H_vap: float,
    T_range: Tuple[float, float] = None,
    P_range: Tuple[float, float] = None,
    substance_name: str = "Substance",
    point_highlight: Tuple[float, float] = None,
    figsize: Tuple[int, int] = (10, 6),
    R: float = 8.314,
    n_points: int = 200,
    temp_margin: float = 10,
    pressure_limit: float = 1e7
) -> None:
    """
    绘制Clausius-Clapeyron方程的饱和蒸气压曲线
    
    创建包含两个子图的综合图表：1. 压力-温度关系图（P vs T）；
    2. 线性化图（lnP vs 1/T），用于验证Clausius-Clapeyron方程的线性关系。
    
    Parameters:
    -----------
    P1 : float
        参考压力，单位为Pa
    T1 : float
        参考温度，单位为K
    delta_H_vap : float
        摩尔汽化焓，单位为J/mol
    T_range : tuple, optional
        温度范围，格式为(T_min, T_max)，单位为K
    P_range : tuple, optional
        压力范围，格式为(P_min, P_max)，单位为Pa
    substance_name : str, optional
        物质名称，用于图表标签，默认为"Substance"
    point_highlight : tuple, optional
        高亮显示的点，格式为(P, T)，单位为(Pa, K)
    figsize : tuple, optional
        图表尺寸，默认为(10, 6)
    R : float, optional
        气体常数，默认为8.314 J/(mol·K)
    n_points : int, optional
        绘图点数，默认为200
    temp_margin : float, optional
        温度范围扩展边距，默认为10K
    pressure_limit : float, optional
        压力上限，用于过滤异常值，默认为1e7 Pa
    
    Returns:
    --------
    None
        显示图表
    """
    
    R = R

    # 确定温度范围
    if P_range:
        P_min, P_max = P_range
        T_min = 1 / (1/T1 - (R / delta_H_vap) * np.log(P_min / P1))
        T_max = 1 / (1/T1 - (R / delta_H_vap) * np.log(P_max / P1))
        T_array = np.linspace(max(T_min - temp_margin, 200), min(T_max + temp_margin, T1 + 100), n_points)
    elif T_range:
        T_array = np.linspace(T_range[0], T_range[1], n_points)
    else:
        T_array = np.linspace(T1 - 50, T1 + 50, n_points)

    # 过滤合法温度（避免负压）
    T_array = T_array[T_array > 0]
    P_array = np.array([
        clausius_clapeyron_pressure(P1, T1, T, delta_H_vap, R) for T in T_array
    ])

    # 过滤掉过大的压力（避免溢出）
    valid_idx = (P_array > 0) & (P_array < pressure_limit)
    T_array = T_array[valid_idx]
    P_array = P_array[valid_idx]

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图：P vs T（线性）
    ax1.plot(T_array, P_array / 1e5, 'b-', label=f'{substance_name} 饱和蒸气压')  # 转换为 bar
    if point_highlight:
        ax1.plot(point_highlight[1], point_highlight[0] / 1e5, 'ro', label='目标点')
    ax1.set_xlabel('温度 T (K)')
    ax1.set_ylabel('蒸气压 P (bar)')
    ax1.set_title(f'{substance_name} 饱和蒸气压曲线')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：lnP vs 1/T（线性化）
    inv_T = 1 / T_array
    ln_P = np.log(P_array)
    ax2.plot(inv_T, ln_P, 'g-', label='lnP ~ 1/T')
    ax2.set_xlabel('1/T (1/K)')
    ax2.set_ylabel('ln(P) (P in Pa)')
    ax2.set_title('Clausius-Clapeyron 线性图')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

# =============================================
# 通用求解器类（封装功能）
# =============================================

class ClausiusClapeyronCalculator:
    
    # 常见物质数据库（标准沸点、汽化焓）
    SUBSTANCE_DB = {
        "water": {
            "T1": 373.15,  # K, 100°C
            "P1": 101325,  # Pa, 1 atm
            "delta_H_vap": 40670  # J/mol
        },
        "ethanol": {
            "T1": 351.5,  # K, 78.4°C
            "P1": 101325,
            "delta_H_vap": 38560
        }
    }

    def __init__(self, substance: str = None, T1: float = None, P1: float = None, delta_H_vap: float = None):
        """
        初始化Clausius-Clapeyron计算器
        
        可以通过预定义物质名称或自定义参数来初始化计算器。
        预定义物质包括水和乙醇的物理参数。
        
        Parameters:
        -----------
        substance : str, optional
            预定义物质名称，可选值："water", "ethanol"
        T1 : float, optional
            参考温度，单位为K
        P1 : float, optional
            参考压力，单位为Pa
        delta_H_vap : float, optional
            摩尔汽化焓，单位为J/mol
        
        Raises:
        -------
        ValueError
            当既未指定substance也未提供完整的(T1, P1, delta_H_vap)参数时抛出
        """
        
        if substance and substance in self.SUBSTANCE_DB:
            data = self.SUBSTANCE_DB[substance]
            self.T1 = data["T1"]
            self.P1 = data["P1"]
            self.delta_H_vap = data["delta_H_vap"]
        elif T1 and P1 and delta_H_vap:
            self.T1 = T1
            self.P1 = P1
            self.delta_H_vap = delta_H_vap
        else:
            raise ValueError("必须指定 substance 或 (T1, P1, delta_H_vap)")

    def solve_temperature(self, pressure: float) -> float:
        """
        根据压力计算对应的温度
        
        使用Clausius-Clapeyron方程，根据给定的压力计算对应的饱和温度。
        
        Parameters:
        -----------
        pressure : float
            目标压力，单位为Pa
        
        Returns:
        --------
        float
            对应的饱和温度，单位为K
        """
        
        return clausius_clapeyron_temperature(
            P1=self.P1,
            T1=self.T1,
            P2=pressure,
            delta_H_vap=self.delta_H_vap
        )

    def solve_pressure(self, temperature: float) -> float:
        """
        根据温度计算对应的压力
        
        使用Clausius-Clapeyron方程，根据给定的温度计算对应的饱和蒸气压。
        
        Parameters:
        -----------
        temperature : float
            目标温度，单位为K
        
        Returns:
        --------
        float
            对应的饱和蒸气压，单位为Pa
        """
        
        return clausius_clapeyron_pressure(
            P1=self.P1,
            T1=self.T1,
            T2=temperature,
            delta_H_vap=self.delta_H_vap
        )

    def visualize(self, P_range: Tuple[float, float] = None, T_range: Tuple[float, float] = None,
                  highlight_pressure: float = None, figsize: Tuple[int, int] = (10, 6),
                  substance_name: str = None, **kwargs):
        """
        可视化Clausius-Clapeyron方程曲线
        
        生成包含压力-温度关系和线性化图表的综合可视化。
        
        Parameters:
        -----------
        P_range : tuple, optional
            压力范围，格式为(P_min, P_max)，单位为Pa
        T_range : tuple, optional
            温度范围，格式为(T_min, T_max)，单位为K
        highlight_pressure : float, optional
            需要高亮显示的压力值，单位为Pa
        figsize : tuple, optional
            图表尺寸，默认为(10, 6)
        substance_name : str, optional
            物质名称，用于图表标签
        **kwargs
            传递给plot_clausius_clapeyron函数的其他参数
        
        Returns:
        --------
        None
            显示图表
        """
        
        point = None
        if highlight_pressure:
            T_at_P = self.solve_temperature(highlight_pressure)
            point = (highlight_pressure, T_at_P)

        # 确定物质名称
        if substance_name is None:
            substance_name = "Water" if hasattr(self, 'SUBSTANCE_DB') else "Custom"

        plot_clausius_clapeyron(
            P1=self.P1,
            T1=self.T1,
            delta_H_vap=self.delta_H_vap,
            P_range=P_range,
            T_range=T_range,
            substance_name=substance_name,
            point_highlight=point,
            figsize=figsize,
            **kwargs
        )

# =============================================
# 示例应用：压力锅中水的最大温度计算
# =============================================

if __name__ == "__main__":
    # instruction1 = "1. Given a set of experimental vapor pressure data for a liquid: the temperatures are T=[300,310,320,330,340]K and the corresponding pressures are P=[3500,5600,8800,13600,20600]Pa , calculate the boiling point of the liquid at standard atmospheric pressure (101.325 kPa)."
    instruction2 = "2. Given the molar enthalpy of vaporization of water \\(\\Delta_{\\mathrm{vap}}H_{\\mathrm{m}}^{\\ominus}=40.67\\mathrm{~kJ}\\cdot\\mathrm{mol}^{-1}\\), and the maximum allowable pressure for a cooking pressure cooker is 0.23\\(\\mathrm{MPa}\\), calculate the maximum temperature that water can reach inside the pressure cooker."
    # 已知条件
    delta_H_vap_water = 40670  # J/mol
    P1 = 101325                # Pa, 1 atm
    T1 = 373.15                # K, 水的正常沸点
    P_max_pressure_cooker = 0.23e6  # 0.23 MPa = 230,000 Pa

    # 创建计算器（以水为例）
    calc = ClausiusClapeyronCalculator(substance="water")

    # 计算压力锅内最大温度
    T_max = calc.solve_temperature(P_max_pressure_cooker)
    T_max_celsius = T_max - 273.15

    print(f"压力锅最大压力: {P_max_pressure_cooker / 1e5:.2f} bar")
    print(f"水在此压力下的沸点: {T_max:.2f} K ({T_max_celsius:.2f} °C)")

    # 可视化
    calc.visualize(
        P_range=(8e4, 3e5),  # 0.8 - 3 bar
        highlight_pressure=P_max_pressure_cooker
    )
