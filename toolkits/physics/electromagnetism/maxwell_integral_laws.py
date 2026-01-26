import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# 1. 数学核心模块：麦克斯韦方程组积分形式对应关系（底层逻辑）
# =============================================================================

def maxwell_integral_laws():
    
    equations = {
        1: {
            "description": "The electric field is a source field",
            "meaning": "电位移矢量D的闭合曲面积分等于自由电荷总量 → 电场有源性",
            "equation": r"$\oint_S \mathbf{D} \cdot d\mathbf{S} = \sum q_f$"
        },
        2: {
            "description": "A changing magnetic field produces a rotational electric field",
            "meaning": "电场E沿闭合路径的线积分等于磁通量变化率的负值 → 法拉第电磁感应定律",
            "equation": r"$\oint_L \mathbf{E} \cdot d\mathbf{l} = -\int_S \frac{\partial \mathbf{B}}{\partial t} \cdot d\mathbf{S}$"
        },
        3: {
            "description": "The magnetic field is a source-free field, without magnetic monopole",
            "meaning": "磁感应强度B的闭合曲面积分为零 → 磁场无源（无磁单极）",
            "equation": r"$\oint_S \mathbf{B} \cdot d\mathbf{S} = 0$"
        },
        4: {
            "description": "Conduction and displacement current produce rotational magnetic field",
            "meaning": "磁场H沿闭合路径的线积分等于全电流（传导+位移）→ 安培-麦克斯韦定律",
            "equation": r"$\oint_L \mathbf{H} \cdot d\mathbf{l} = \int_S \left(\mathbf{j} + \frac{\partial \mathbf{D}}{\partial t}\right) \cdot d\mathbf{S}$"
        }
    }
    return equations

# =============================================================================
# 2. 核心匹配函数：根据描述返回对应的方程编号
# =============================================================================

def match_description_to_equation(description: str) -> int:
    
    desc_lower = description.strip().lower()
    keywords = {
        1: ["electric", "source", "charge", "field"],
        2: ["changing", "magnetic", "produces", "rotational", "electric"],
        3: ["magnetic", "source-free", "monopole", "no", "single"],
        4: ["current", "conduction", "displacement", "rotational", "magnetic", "produce"]
    }
    
    scores = {i: 0 for i in range(1, 5)}
    for word in desc_lower.split():
        word = word.strip(".,:;!?\"'")
        for eq_num, keys in keywords.items():
            if any(word in key or key in word for key in keys):
                scores[eq_num] += 1

    # 返回匹配度最高的方程编号
    best_match = max(scores, key=scores.get)
    return best_match if scores[best_match] > 0 else -1  # -1 表示未匹配

# =============================================================================
# 3. 可视化函数：展示四个麦克斯韦方程的物理图像（简化示意）
# =============================================================================

def visualize_maxwell_equations():
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    equations = maxwell_integral_laws()

    titles = [
        "1. Electric Field Has Sources (Gauss's Law)",
        "2. Changing B → Rotational E (Faraday's Law)",
        "3. Magnetic Field Is Source-Free (No Magnetic Monopole)",
        "4. Currents → Rotational H (Ampère-Maxwell Law)"
    ]

    for i in range(4):
        ax = axes[i]
        ax.set_title(titles[i], fontsize=10, fontweight='bold')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axis('off')

        if i == 0:  # Gauss's Law: 电场有源
            ax.add_patch(plt.Circle((0, 0), 0.5, color='red', label='Charge'))
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                x0, y0 = 0.5 * np.cos(angle), 0.5 * np.sin(angle)
                ax.arrow(x0, y0, 1.2*np.cos(angle), 1.2*np.sin(angle),
                         head_width=0.15, fc='blue', ec='blue')
            ax.text(0, -1.5, "D flux ∝ charge", ha='center', fontsize=9)

        elif i == 1:  # Faraday: 变化B产生涡旋E
            ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, color='gray', alpha=0.3))
            ax.text(0, 0, "∂B/∂t ≠ 0", ha='center', va='center', fontsize=10)
            # 绘制环形电场
            phi = np.linspace(0, 2*np.pi, 20)
            ex = 1.5 * np.cos(phi)
            ey = 1.5 * np.sin(phi)
            ax.plot(ex, ey, 'b-', linewidth=2)
            ax.arrow(1.5, 0, -0.01, 0.01, head_width=0.1, fc='b', ec='b')
            ax.text(0, -2, "E forms closed loops", ha='center', fontsize=9)

        elif i == 2:  # 磁场无源
            ax.quiver([-1, 1], [0, 0], [1, -1], [0, 0], scale=20, color='green')
            ax.quiver([0, 0], [-1, 1], [0, 0], [1, -1], scale=20, color='green')
            ax.add_patch(plt.Circle((0, 0), 0.3, color='green', fill=False, linewidth=2))
            ax.text(0, -2, "B field lines close on themselves", ha='center', fontsize=9)

        elif i == 3:  # 全电流激发磁场
            ax.add_patch(plt.Circle((0, 0), 0.4, color='orange'))
            ax.text(0, 0, "J", ha='center', va='center', fontsize=12, color='white')
            # 磁场环流
            phi = np.linspace(0, 2*np.pi, 20)
            hx = 1.8 * np.cos(phi)
            hy = 1.8 * np.sin(phi)
            ax.plot(hx, hy, 'r-', linewidth=2)
            ax.arrow(1.8, 0, -0.01, 0.01, head_width=0.1, fc='r', ec='r')
            ax.text(0, -2, "H circulates around current", ha='center', fontsize=9)

    plt.suptitle("Maxwell's Integral Equations - Conceptual Visualization", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. 综合求解函数：输入描述，返回方程编号 + 数学表达 + 可视化
# =============================================================================

def solve_maxwell_questions(descriptions):
    
    results = []
    equations = maxwell_integral_laws()
    
    for desc in descriptions:
        eq_num = match_description_to_equation(desc)
        if eq_num != -1:
            result = {
                "description": desc,
                "equation_number": eq_num,
                "math_expression": equations[eq_num]["equation"],
                "physical_meaning": equations[eq_num]["meaning"]
            }
            results.append(result)
            print(f"Description: {desc}")
            print(f"→ Equation ({eq_num}): {equations[eq_num]['equation']}")
            print(f"  Meaning: {equations[eq_num]['meaning']}")
        else:
            print(f"Cannot match: {desc}")
    
    return results

# =============================================================================
# 5. 示例调用（用户问题）
# =============================================================================

if __name__ == "__main__":
    # 用户提供的四个问题描述
    questions = [
        "A changing magnetic field produces a rotational electric field",
        "The magnetic field is a source-free field, without the existence of a single 'magnetic charge' or magnetic monopole",
        "The electric field is a source field",
        "Not only can conduction current produce a rotational magnetic field, but displacement current can also produce a rotational magnetic field"
    ]
    
    # 求解并输出结果
    print("🔍 Solving Maxwell's Equation Matching:")
    answers = solve_maxwell_questions(questions)
    
    # 可视化四个方程的物理图像
    print("\n🎨 Generating conceptual visualization...")
    visualize_maxwell_equations()

    # 输出最终答案编号（可用于填空）
    print("✅ Final Answer Mapping:")
    for i, q in enumerate(questions):
        ans = match_description_to_equation(q)
        print(f"({i+1}) → Equation ({ans})")
