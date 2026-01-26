# Filename: electromagnetic_field_solver.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# 尝试导入更新的matplotlib版本兼容性
try:
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    HAS_LINE3D_COLLECTION = True
except ImportError:
    HAS_LINE3D_COLLECTION = False

class Arrow3D(FancyArrowPatch):
    """
    3D arrow for visualization in matplotlib
    兼容新版本matplotlib的3D绘图要求
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        兼容新版本matplotlib的3D投影方法
        返回z坐标的最小值用于深度排序
        """
        try:
            xs3d, ys3d, zs3d = self._verts3d
            if hasattr(self, 'axes') and self.axes is not None and hasattr(self.axes, 'M'):
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            elif renderer is not None and hasattr(renderer, 'M'):
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            else:
                # 如果没有有效的投影矩阵，返回默认值
                return 0
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            return np.min(zs)
        except Exception:
            # 如果投影失败，返回默认值
            return 0

    def draw(self, renderer):
        """
        绘制3D箭头
        """
        try:
            xs3d, ys3d, zs3d = self._verts3d
            # 尝试多种方式获取投影矩阵
            if hasattr(renderer, 'M'):
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            elif hasattr(self, 'axes') and self.axes is not None and hasattr(self.axes, 'M'):
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            else:
                # 如果无法获取投影矩阵，跳过绘制
                return
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)
        except Exception as e:
            # 如果绘制失败，静默处理
            pass

def add_3d_arrow(ax, start, end, color='red', mutation_scale=20, linewidth=2, arrowstyle='->'):
    """
    在3D图上添加箭头的函数包装（推荐使用）
    
    这是 Arrow3D 类的函数包装，用于在工具注册时避免实例化问题。
    
    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D坐标轴对象
    start : array-like, shape (3,)
        箭头起点坐标 [x, y, z]
    end : array-like, shape (3,)
        箭头终点坐标 [x, y, z]
    color : str, optional
        箭头颜色，默认为 'red'
    mutation_scale : float, optional
        箭头大小缩放因子，默认为 20
    linewidth : float, optional
        箭头线宽，默认为 2
    arrowstyle : str, optional
        箭头样式，默认为 '->'
    
    Returns:
    --------
    Arrow3D
        创建的箭头对象（已添加到axes）
    
    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> add_3d_arrow(ax, [0, 0, 0], [1, 1, 1], color='blue')
    """
    try:
        arrow = Arrow3D(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            mutation_scale=mutation_scale,
            lw=linewidth,
            arrowstyle=arrowstyle,
            color=color
        )
        ax.add_artist(arrow)
        return arrow
    except Exception as e:
        # 如果 Arrow3D 创建失败，使用简化的箭头绘制作为备用
        print(f"⚠️ Arrow3D 创建失败，使用简化箭头: {e}")
        draw_simple_3d_arrow(ax, start, end, color=color, arrow_size=0.3)
        return None


def draw_simple_3d_arrow(ax, start, end, color='red', arrow_size=0.3):
    """
    简化的3D箭头绘制函数，使用基本线条和标记
    作为Arrow3D类的备用方案
    """
    # 绘制主体线条
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=2)
    
    # 计算箭头方向
    direction = np.array(end) - np.array(start)
    direction = direction / np.linalg.norm(direction)
    
    # 绘制箭头头部（使用三角形）
    arrow_head = end - arrow_size * direction
    ax.plot([end[0], arrow_head[0]], [end[1], arrow_head[1]], [end[2], arrow_head[2]], 
            color=color, linewidth=2)

def calculate_magnetic_field_line_current(r_vector, current_vector, current_position):
    """
    计算无限长直线电流在空间某点产生的磁感应强度
    
    基于毕奥-萨伐尔定律，无限长直线电流在距离为r的点产生的磁场大小为:
    B = (μ₀*I)/(2πr)，方向由右手螺旋定则确定
    
    Parameters:
    -----------
    r_vector : numpy.ndarray
        场点位置矢量，形状为(3,)，表示(x,y,z)坐标
    current_vector : numpy.ndarray
        电流方向的单位矢量，形状为(3,)
    current_position : numpy.ndarray
        电流线所在直线上一点的位置矢量，形状为(3,)
    
    Returns:
    --------
    numpy.ndarray
        磁感应强度矢量，形状为(3,)，单位为特斯拉(T)
    """
    mu0 = 4 * np.pi * 1e-7  # 真空磁导率，单位：H/m
    
    # 计算从电流线到场点的最短距离矢量
    # 首先计算从场点到电流线的矢量在电流方向上的投影
    current_unit = current_vector / np.linalg.norm(current_vector)
    r_relative = r_vector - current_position
    projection = np.dot(r_relative, current_unit) * current_unit
    
    # 计算垂直于电流方向的位移矢量
    r_perpendicular = r_relative - projection
    
    # 计算垂直距离
    distance = np.linalg.norm(r_perpendicular)
    
    if distance < 1e-10:  # 避免除以零
        return np.zeros(3)
    
    # 计算磁场方向（右手螺旋定则）
    direction = np.cross(current_unit, r_perpendicular)
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    
    # 计算磁场大小
    magnitude = mu0 * np.linalg.norm(current_vector) / (2 * np.pi * distance)
    
    # 返回磁场矢量
    return magnitude * direction

def calculate_total_magnetic_field(r_vector, current_sources):
    """
    计算多个电流源在空间某点产生的总磁感应强度
    
    Parameters:
    -----------
    r_vector : numpy.ndarray
        场点位置矢量，形状为(3,)，表示(x,y,z)坐标
    current_sources : list of dict
        电流源列表，每个电流源为一个字典，包含：
        - 'position': 电流线上一点的位置矢量
        - 'direction': 电流方向的单位矢量
        - 'magnitude': 电流大小，单位为安培(A)
    
    Returns:
    --------
    numpy.ndarray
        总磁感应强度矢量，形状为(3,)，单位为特斯拉(T)
    """
    total_field = np.zeros(3)
    
    for source in current_sources:
        position = source['position']
        direction = source['direction']
        magnitude = source['magnitude']
        
        # 计算电流矢量
        current_vector = direction * magnitude
        
        # 计算该电流源产生的磁场
        field = calculate_magnetic_field_line_current(r_vector, current_vector, position)
        
        # 累加磁场
        total_field += field
    
    return total_field

def calculate_magnetic_field_grid(x_range, y_range, z_range, current_sources, grid_size=10):
    """
    计算空间网格点上的磁场分布
    
    Parameters:
    -----------
    x_range : tuple
        x坐标范围，形式为(x_min, x_max)
    y_range : tuple
        y坐标范围，形式为(y_min, y_max)
    z_range : tuple
        z坐标范围，形式为(z_min, z_max)
    current_sources : list of dict
        电流源列表
    grid_size : int, optional
        每个维度上的网格点数，默认为10
    
    Returns:
    --------
    tuple
        (X, Y, Z, Bx, By, Bz)，其中X, Y, Z是网格坐标，Bx, By, Bz是对应点的磁场分量
    """
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    z = np.linspace(z_range[0], z_range[1], grid_size)
    
    X, Y, Z = np.meshgrid(x, y, z)
    Bx = np.zeros_like(X)
    By = np.zeros_like(Y)
    Bz = np.zeros_like(Z)
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                r_vector = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                B = calculate_total_magnetic_field(r_vector, current_sources)
                Bx[i,j,k], By[i,j,k], Bz[i,j,k] = B
    
    return X, Y, Z, Bx, By, Bz

def visualize_magnetic_field(current_sources, field_point=None, field_vector=None, plot_type='3d'):
    """
    可视化电流源和磁场
    
    Parameters:
    -----------
    current_sources : list of dict
        电流源列表
    field_point : numpy.ndarray, optional
        要显示磁场的特定点，形状为(3,)
    field_vector : numpy.ndarray, optional
        field_point处的磁场矢量，形状为(3,)
    plot_type : str, optional
        绘图类型，可选 '3d' 或 '2d'，默认为 '3d'
    
    Returns:
    --------
    matplotlib.figure.Figure
        绘制的图形对象
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if plot_type == '3d':
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制电流源
        for i, source in enumerate(current_sources):
            pos = source['position']
            direction = source['direction']
            magnitude = source['magnitude']
            
            # 计算电流线的起点和终点
            # 为了可视化，我们将无限长电流表示为有限长度
            line_length = 5.0  # 可视化长度
            start_point = pos - line_length * direction
            end_point = pos + line_length * direction
            
            # 绘制电流线
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    [start_point[2], end_point[2]], 'r-', linewidth=2, 
                    label=f'电流 {i+1}: {magnitude:.1f}A')
            
            # 添加电流方向箭头（使用函数包装）
            add_3d_arrow(ax, pos, pos + direction, color='red', mutation_scale=20, linewidth=2)
        
        # 如果提供了场点和磁场矢量，则绘制磁场
        if field_point is not None and field_vector is not None:
            # 绘制场点
            ax.scatter(field_point[0], field_point[1], field_point[2], 
                      color='blue', s=100, label='场点')
            
            # 绘制磁场矢量（使用函数包装）
            field_scale = 2.0 / np.linalg.norm(field_vector)  # 缩放因子使箭头可见
            arrow_end = field_point + field_scale * field_vector
            add_3d_arrow(ax, field_point, arrow_end, color='blue', mutation_scale=20, linewidth=2)
            
            # 添加磁场大小标签
            field_magnitude = np.linalg.norm(field_vector)
            ax.text(field_point[0], field_point[1], field_point[2], 
                   f'|B| = {field_magnitude:.2e} T', color='blue')
        
        # 设置坐标轴标签和范围
        ax.set_xlabel('X轴 (m)')
        ax.set_ylabel('Y轴 (m)')
        ax.set_zlabel('Z轴 (m)')
        
        # 设置坐标轴范围
        max_range = 5.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        # 添加图例和标题
        ax.legend()
        plt.title('电流源与磁场可视化')
        
    elif plot_type == '2d':
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制电流源（在xy平面上的投影）
        for i, source in enumerate(current_sources):
            pos = source['position']
            direction = source['direction']
            magnitude = source['magnitude']
            
            # 计算电流线的起点和终点
            line_length = 5.0
            start_point = pos - line_length * direction
            end_point = pos + line_length * direction
            
            # 绘制电流线（xy平面投影）
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 'r-', linewidth=2, 
                    label=f'电流 {i+1}: {magnitude:.1f}A')
            
            # 添加电流方向箭头
            ax.arrow(pos[0], pos[1], direction[0], direction[1], 
                    head_width=0.2, head_length=0.3, fc='r', ec='r')
        
        # 如果提供了场点和磁场矢量，则绘制磁场
        if field_point is not None and field_vector is not None:
            # 绘制场点
            ax.scatter(field_point[0], field_point[1], color='blue', s=100, label='场点')
            
            # 绘制磁场矢量（xy平面投影）
            field_scale = 2.0 / np.linalg.norm(field_vector)
            ax.arrow(field_point[0], field_point[1], 
                    field_scale * field_vector[0], field_scale * field_vector[1], 
                    head_width=0.2, head_length=0.3, fc='blue', ec='blue')
            
            # 添加磁场大小标签
            field_magnitude = np.linalg.norm(field_vector)
            ax.text(field_point[0], field_point[1], 
                   f'|B| = {field_magnitude:.2e} T', color='blue')
        
        # 设置坐标轴标签和范围
        ax.set_xlabel('X轴 (m)')
        ax.set_ylabel('Y轴 (m)')
        
        # 设置坐标轴范围
        max_range = 5.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        # 添加图例和标题
        ax.legend()
        ax.set_title('电流源与磁场可视化 (XY平面投影)')
        ax.grid(True)
    
    return fig

    """
    计算电流元在磁场中受到的电磁力
    
    基于洛伦兹力公式：F = I × L × B，其中I是电流，L是电流元长度矢量，B是磁感应强度
    
    Parameters:
    -----------
    current_vector : numpy.ndarray
        电流矢量，包含方向和大小，形状为(3,)
    position : numpy.ndarray
        电流元位置，形状为(3,)
    magnetic_field_func : function
        计算磁场的函数，接受位置参数，返回磁场矢量
    
    Returns:
    --------
    numpy.ndarray
        电磁力矢量，形状为(3,)，单位为牛顿(N)
    """
    # 计算位置处的磁场
    B = magnetic_field_func(position)
    return B
    
def main():
    """
    主函数：解决两根无限长直导线在原点产生磁场的问题
    
    问题描述：
    - 导线1：位置 y = -a，电流方向沿z轴正方向，电流大小 I
    - 导线2：位置 y = a，电流方向沿x轴正方向，电流大小 I
    - 求解：坐标原点处的磁感应强度
    """
    print("=" * 60)
    print("两根无限长直导线磁场计算")
    print("=" * 60)
    
    # 定义问题参数
    a = 1.0  # 导线到原点的距离，单位：米
    I = 10.0  # 电流大小，单位：安培
    
    print(f"参数设置：")
    print(f"  导线间距：2a = {2*a:.1f} m")
    print(f"  电流大小：I = {I:.1f} A")
    print(f"  导线1位置：y = -{a:.1f} m，电流方向：+z")
    print(f"  导线2位置：y = +{a:.1f} m，电流方向：+x")
    print()
    
    # 定义电流源
    current_sources = [
        {
            'position': np.array([0, -a, 0]),     # 导线1位置：y = -a
            'direction': np.array([0, 0, 1]),     # 导线1方向：+z轴
            'magnitude': I                        # 电流大小
        },
        {
            'position': np.array([0, a, 0]),      # 导线2位置：y = +a
            'direction': np.array([1, 0, 0]),     # 导线2方向：+x轴
            'magnitude': I                        # 电流大小
        }
    ]
    
    # 计算原点处的磁场
    origin = np.array([0.0, 0.0, 0.0])
    B_total = calculate_total_magnetic_field(origin, current_sources)
    
    print("计算结果：")
    print("-" * 40)
    
    # 分别计算每根导线的贡献
    B1 = calculate_magnetic_field_line_current(
        origin, 
        current_sources[0]['direction'] * current_sources[0]['magnitude'], 
        current_sources[0]['position']
    )
    
    B2 = calculate_magnetic_field_line_current(
        origin, 
        current_sources[1]['direction'] * current_sources[1]['magnitude'], 
        current_sources[1]['position']
    )
    
    print(f"导线1贡献：B₁ = [{B1[0]:.4e}, {B1[1]:.4e}, {B1[2]:.4e}] T")
    print(f"导线2贡献：B₂ = [{B2[0]:.4e}, {B2[1]:.4e}, {B2[2]:.4e}] T")
    print(f"总磁场：  B = [{B_total[0]:.4e}, {B_total[1]:.4e}, {B_total[2]:.4e}] T")
    print(f"磁场大小：|B| = {np.linalg.norm(B_total):.4e} T")
    print()

    
    # 可视化结果
    print("生成可视化图形...")
    
    try:
        # 3D可视化
        fig_3d = visualize_magnetic_field(
            current_sources, 
            field_point=origin, 
            field_vector=B_total, 
            plot_type='3d'
        )
        print("3D可视化成功生成")
    except Exception as e:
        print(f"3D可视化失败: {e}")
        print("将只显示2D可视化")
        fig_3d = None
    
    try:
        # 2D可视化
        fig_2d = visualize_magnetic_field(
            current_sources, 
            field_point=origin, 
            field_vector=B_total, 
            plot_type='2d'
        )
        print("2D可视化成功生成")
    except Exception as e:
        print(f"2D可视化失败: {e}")
        fig_2d = None
    
    # 计算空间磁场分布
    print("计算空间磁场分布...")
    X, Y, Z, Bx, By, Bz = calculate_magnetic_field_grid(
        x_range=(-2*a, 2*a),
        y_range=(-2*a, 2*a),
        z_range=(-0.5*a, 0.5*a),
        current_sources=current_sources,
        grid_size=5
    )
    
    # 分析特殊点的磁场
    special_points = [
        ("原点", np.array([0, 0, 0])),
        ("x轴上点", np.array([a, 0, 0])),
        ("y轴上点", np.array([0, a/2, 0])),
        ("z轴上点", np.array([0, 0, a]))
    ]
    
    print("\n特殊点磁场分析：")
    print("-" * 40)
    for name, point in special_points:
        B_point = calculate_total_magnetic_field(point, current_sources)
        print(f"{name:8s}：位置{point}, 磁场{B_point}, |B|={np.linalg.norm(B_point):.2e} T")
    
    # 显示图形
    plt.show()
    
    # 保存结果
    save_results = input("\n是否保存计算结果到文件？(y/n): ")
    if save_results.lower() == 'y':
        save_calculation_results(current_sources, origin, B_total, B_theoretical)
    
    print("\n计算完成！")

def save_calculation_results(current_sources, field_point, calculated_field, theoretical_field):
    """
    保存计算结果到文件
    """
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"magnetic_field_calculation_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("电磁场计算结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("电流源配置：\n")
        for i, source in enumerate(current_sources):
            f.write(f"  导线{i+1}：位置{source['position']}, 方向{source['direction']}, 电流{source['magnitude']}A\n")
        f.write(f"\n场点：{field_point}\n")
        f.write(f"计算结果：{calculated_field}\n")
        f.write(f"理论结果：{theoretical_field}\n")
        f.write(f"计算误差：{np.linalg.norm(calculated_field - theoretical_field) / np.linalg.norm(theoretical_field) * 100:.2f}%\n")
    
    print(f"结果已保存到文件：{filename}")

def interactive_demo():
    """
    交互式演示：允许用户修改参数
    """
    print("\n" + "=" * 60)
    print("交互式参数调整")
    print("=" * 60)
    
    while True:
        try:
            # 获取用户输入
            a = float(input("输入导线间距的一半 a (米，默认1.0): ") or "1.0")
            I = float(input("输入电流大小 I (安培，默认10.0): ") or "10.0")
            
            # 构建电流源
            current_sources = [
                {
                    'position': np.array([0, -a, 0]),
                    'direction': np.array([0, 0, 1]),
                    'magnitude': I
                },
                {
                    'position': np.array([0, a, 0]),
                    'direction': np.array([1, 0, 0]),
                    'magnitude': I
                }
            ]
            
            # 计算磁场
            origin = np.array([0.0, 0.0, 0.0])
            B_total = calculate_total_magnetic_field(origin, current_sources)
            
            print(f"\n结果：原点处磁场 = {B_total}")
            print(f"磁场大小 = {np.linalg.norm(B_total):.4e} T")
            
            # 询问是否继续
            continue_calc = input("\n继续计算其他参数？(y/n): ")
            if continue_calc.lower() != 'y':
                break
                
        except ValueError:
            print("输入错误，请输入有效数字！")
        except KeyboardInterrupt:
            print("\n程序中断")
            break

# 如果直接运行此脚本，执行main函数
if __name__ == "__main__":
    main()
    
