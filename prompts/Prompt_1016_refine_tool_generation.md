## 任务目标
基于物理、化学、材料科学领域知识，构建专业性强、通用性高的检索与计算工具包，支持多场景问题求解，并兼容OpenAI Function Calling规范。

---

## 执行要求

### 1. 领域知识与数学建模
- **识别核心科学原理**：明确问题涉及的物理定律、化学反应机理或材料性质
- **数学模型构建**：将科学问题转化为数学方程组、优化问题或数值计算
- **求解路径设计**：分解为可复用的计算单元，考虑不同参数区间的分支逻辑

### 2. 专业工具库选型
**必须优先使用领域专属库**，包括但不限于：

**物理计算：**
- `scipy.integrate` - 微分方程求解
- `scipy.optimize` - 优化与拟合
- `sympy` - 符号计算

**化学计算：**
- `rdkit` - 分子处理与性质计算
- `pubchempy` - PubChem数据库访问
- `mendeleev` - 元素周期表数据
- `chempy` - 化学计量学与反应平衡

**材料科学：**
- `pymatgen` - 晶体结构分析与相图计算
- `ase` - 原子模拟环境
- `mp-api` - Materials Project数据库接口
- `matminer` - 材料数据挖掘

**可视化工具：**
- `plotly` - 交互式科学可视化（优先）
- `matplotlib` - 基础绘图
- `seaborn` - 统计图表

### 3. 工具函数设计原则

#### 3.1 通用性与可组合性
- **单一职责**：每个函数只做一件事，便于组合调用
- **参数通用化**：避免硬编码，支持数组/批量输入
- **分支逻辑**：用装饰器或策略模式处理不同物理区间/化学环境
- **工具链设计**：高层函数调用底层函数，形成工具生态

#### 3.2 OpenAI Function Calling 兼容规范
每个工具函数必须包含：
````python
def tool_function_name(param1: float, param2: str, param3: Optional[list] = None) -> dict:
    """
    [简洁的一句话描述 - 用于Function Calling的description]
    
    详细的科学原理说明（2-3句话）
    ### 🔧 更新后的代码质量检查清单
    - [ ] **所有函数参数类型为可JSON序列化**：str, int, float, bool, List, Dict
    - [ ] **Python对象构建逻辑在函数内部**，不作为参数传入
    - [ ] **支持多种输入格式**：文件路径、数据库ID、字符串表示
    - [ ] **示例代码使用基础类型调用**，不涉及Python对象
    
    Args:
        param1: 参数的物理/化学意义，单位，取值范围（如：温度/K, 范围273-373）
        param2: 参数说明（如：元素符号，示例'C','N','O'）
        param3: 可选参数说明，默认值选择依据
    
    Returns:
        dict: {
            'result': 主要计算结果（含单位说明）,
            'metadata': {额外信息如收敛状态、使用的数据库等}
        }
    
    Example:
        >>> result = tool_function_name(300.0, 'H2O')
        >>> print(result['result'])
    """
    # 实现代码
    return {"result": value, "metadata": {}}
````


### 4. 工具包架构设计
#### 🎯 设计原则总结：分层架构设计
#### 输出规范：保证每个工具的返回格式都是统一的`{'result': ..., 'metadata': {...}}`格式

##### 原子函数（第一层）

- [ ] **完整的边界条件检查**（类型、范围、特殊值、函数测试）
- [ ] **可独立使用**：
  - [ ] 保证满足返回格式的统一，若返回信息是文件路径，也需要给出具体的路径，将所有过程文件统一保存在`./mid_result/{subject}`这里的`subject`是你需要根据这个题目自己决定选择`physics/chemistry/biology/materials`；
  - [ ] 若遇到返回对象不可序列化的数据结构类似`scipy.sparse.csr_matrix` 是 CSR (Compressed Sparse Row) 格式，一种用于存储稀疏矩阵的高效数据结构。你需要给出如下返回
   `
        summary = f"""稀疏矩阵 (CSR格式):
            - 形状: {sparse_mat.shape}
            - 非零元素: {sparse_mat.nnz} / {sparse_mat.shape[0] * sparse_mat.shape[1]}
            - 稀疏度: {(1 - sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1])) * 100:.2f}%
            - 数据类型: {sparse_mat.dtype}
            已保存到: {filepath}
            可用 scipy.sparse.load_npz() 加载
            """
        return {
            'type': 'sparse_matrix',
            'summary': summary,
            'filepath': filepath,
            'metadata': {
                'shape': sparse_mat.shape,
                'nnz': sparse_mat.nnz,
                'format': 'csr'
            }
        }
    `
- [ ] **对原子函数包含全面的边界条件检查**
  - [ ] 参数类型检查
  - [ ] 参数范围检查
  - [ ] 特殊值处理（零值、空值、极端值）
  - [ ] 函数调用测试（对于callable参数）
- [ ] **保留原子函数供高级用户使用**  
  
##### 组合函数（第二层）

- [ ] **组合函数用于复杂求解任务，这些任务需要多步骤的数学/物理/工程计算、整合多个原子函数的输出与进行高层次的分析和推理。不是简单的原子函数串联，而是实现特定领域的复杂算法或分析流程。**
- [ ] **入参数完全可序列化**
  - [ ] 如果原子函数返回文件路径 → 组合函数应该先理解输出信息是否包含稀疏矩阵或者特殊值，来构造输入
  - [ ] 如果原子函数返回稀疏矩阵字典 → 组合函数接受 dict 格式的矩阵表示
  - [ ] 如果原子函数返回数组 → 组合函数接受 list 格式的数组表示
  - [ ] 避免 LLM 无法构造的参数类型：
        ❌ numpy.ndarray
        ❌ scipy.sparse.csr_matrix
        ❌ pandas.DataFrame
        ❌ 自定义类实例
        ✅ 以上对象的可序列化表示（list/dict）
- [ ] **组合函数内部调用多个原子函数需要写明注释**：`## using file.func_name, and get ** returns`，
- [ ] **所有函数返回遵循上面统一的输出规范**

##### 可视化函数(第三层) 

- [ ] **进行领域专属可视化** 
- [ ] **不需要HTML这样的可视化方式** 
- [ ] **自动处理**：工具会自动将文件保存到 `./tool_images/文件名.扩展名` 
- [ ] **保存文件后必须打印完整路径**，格式：`print(f"image_saved_path: {file_type})`
- [ ] **返回格式仍然遵循输出规范**：`{'result': ..., 'metadata': {...}}`
  
````python
# Filename: <domain>_toolkit.py (如 materials_toolkit.py)

# ============ 第一层：原子工具函数（Atomic Tools） ============
def fetch_property_from_database(identifier: str, property_name: str) -> dict:
    """从免费数据库获取基础数据"""
    pass

def construct_hamiltonian(system_size, interaction_matrix, potential=None, periodic=False) -> dict:
    """
    构建哈密顿量（返回稀疏矩阵）
    Returns:
        return {
            'type': 'sparse_matrix',
            'summary': summary,
            'filepath': filepath,
            'metadata': {
                'shape': sparse_mat.shape,
                'nnz': sparse_mat.nnz,
                'format': 'csr'
            }
        }
    """
    # === 完整边界检查 ===
    if isinstance(system_size, int):
        ...
    return {'type': 'sparse_matrix', ... , 'metadata': {...}}

# ============ 第二层：组合工具函数（Composite Tools） ============

def analyze_commutator(matrix_a: list, matrix_b: list) -> dict:
    """
    计算并分析两个算符的对易子 [A, B] = AB - BA
    
    物理意义：
    - [A, B] = 0: 两个算符对易，可同时测量
    - [A, B] ≠ 0: 两个算符不对易，存在不确定性关系
    
    Args:
        matrix_a: 第一个矩阵（算符A），格式为嵌套列表或稀疏字典
        matrix_b: 第二个矩阵（算符B），格式同上
        
    Returns:
        dict: {'result': summary, 'metadata': {...}}
    """
    # === 参数完全可序列化检查 ===
    if not isinstance(matrix_a, (list, dict)):
        raise TypeError("matrix_a必须是list或dict")
    if not isinstance(matrix_b, (list, dict)):
        raise TypeError("matrix_b必须是list或dict")
    
    # === 反序列化 ===
    from scipy.sparse import csr_matrix
    
    ## 反序列化 matrix_a
    if isinstance(matrix_a, dict):
        A = csr_matrix(
            (matrix_a['data'], (matrix_a['row'], matrix_a['col'])),
            shape=tuple(matrix_a['shape'])
        )
    else:
        A = np.array(matrix_a)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"matrix_a必须是方阵")
    
    ## 反序列化 matrix_b
    ...
    
    # 检查维度匹配
    if A.shape != B.shape:
        raise ValueError(f"两个矩阵维度不匹配: {A.shape} vs {B.shape}")
    
    # === 核心计算：对易子 ===
    commutator = A @ B - B @ A
    ... 

# ============ 第三层：可视化工具（Visualization - 按需） ============

def visualize_domain_specific(data: dict, domain: str, vis_type: str,
                               save_dir: str = './images/',
                               filename: str = None) -> str:
    """
    领域专属可视化工具 - 采用使用各学科标准的可视化方法，请结合目前这个题目的领域生成即可
    
    Args:
        data: 要可视化的数据
        domain: 领域类型 'chemistry'(化学), 'materials'(材料), 'physics'(物理)
        vis_type: 具体可视化类型（见下方说明）
        save_dir: 保存目录
    
    举例领域专属可视化类型：
    
    【化学领域】
    - 'molecule_2d': 2D分子结构图（使用RDKit）
    - 'molecule_3d': 3D分子构象（使用Py3Dmol或RDKit）
    - 'reaction_scheme': 反应方程式图
    - 'spectrum': 光谱图（IR/NMR/UV-Vis）
    - 'energy_diagram': 能级图/势能面
    
    【材料科学领域】
    - 'crystal_structure': 晶体结构图（使用ASE/Pymatgen）
    - 'phase_diagram': 相图（使用Pymatgen）
    - 'band_structure': 能带结构
    - 'dos': 态密度图（Density of States）
    - 'xrd_pattern': XRD衍射图谱
    
    【物理领域】
    - 'field_distribution': 场分布图（电场/磁场/温度场）
    - 'trajectory': 粒子轨迹图
    - 'phase_space': 相空间图
    - 'waveform': 波形图（声波/电磁波）
    - 'vector_field': 矢量场图
    
    【统计分析（通用）】
    - 'histogram': 直方图（分布分析）
    - 'box_plot': 箱线图（离群值检测）
    - 'correlation_heatmap': 相关性热图
    - 'pie_chart': 饼图（成分占比）
    
    Returns:
        str: 保存的图片路径
    """
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 根据领域调用专业库以材料为例domain == 'materials'
    # 使用ASE可视化晶体结构
    from ase.io import write
    from ase import Atoms
    
    atoms = data.get('structure')  # ASE Atoms对象
    save_file = os.path.join(save_path, f"{plot_type}_result.jpg")  # 或.png
    # 保存逻辑
    return save_file


# ============ 第四层：主流程演示 ============
def main():
    """
    演示工具包解决【当前问题】+【至少2个相关场景】
    
    ⚠️ 必须严格按照以下格式编写：
    """
    
    print("=" * 60)
    print("场景1：原始问题求解")
    print("=" * 60)
    print("问题描述：[用1-2句话描述当前要解决的具体问题]")
    print("-" * 60)
    
    # 步骤1：[描述这一步做什么]
    # 调用函数：function_name_1()
    result1 = function_name_1(param1, param2)
    print(f"步骤1结果：{result1['result']}")
    
    # 步骤2：[描述这一步做什么]  
    # 调用函数：function_name_2()，该函数内部调用了 function_name_1()
    result2 = function_name_2(result1['result'], param3)
    print(f"步骤2结果：{result2['result']}")
    
    # 步骤3：[如果有更多步骤]
    # 调用函数：function_name_3()
    final_result1 = function_name_3(result2)
    print(f"✓ 场景1最终答案：{final_result1['result']}\n")
    # 场景二三如上
    
    print("=" * 60)
    print("工具包演示完成")
    print("=" * 60)
    print("总结：")
    print("- 场景1展示了解决原始问题的完整流程")
    print("- 场景2展示了工具的参数泛化能力")
    print("- 场景3展示了工具与数据库的集成能力")


if __name__ == "__main__":
    main()
````

---

### 🔧 代码质量检查清单

在输出代码前，确认：
- [ ] 至少使用2个领域专属库（非numpy/matplotlib）
- [ ] 包含3层工具函数（原子→组合→演示）
- [ ] 每个函数有类型提示和return dict格式
- [ ] main函数严格按照模板格式，包含3个场景
- [ ] **每个场景都有"问题描述"和"调用函数"注释**
- [ ] **每个步骤都用注释标明调用的具体函数名**
- [ ] 可视化代码包含保存逻辑和中文配置
- [ ] 无硬编码值（常量用大写命名放文件顶部）
- [ ] 工具函数间有明确的调用关系

---

## 📤 输出格式
````python
# Filename: <domain>_toolkit.py
"""
<领域名称>计算工具包

主要功能：
1. [功能A]：基于X库实现Y计算
2. [功能B]：调用Z数据库获取W数据
3. [功能C]：组合分析完成P问题

依赖库：
pip install numpy scipy rdkit pymatgen plotly
"""

import numpy as np
from typing import Optional, Union, List, Dict
# 导入领域专属库
from rdkit import Chem
from pymatgen.core import Structure

# 全局常量
PLANCK_CONSTANT = 6.62607015e-34  # J·s

# [完整代码实现...]
````

---

## ⚠️ 关键要求强调

### 关于main函数的硬性要求：

1. **必须包含3个场景**：
   - 场景1：解决给定的原始问题（完整求解过程）
   - 场景2：参数扫描或条件变化分析
   - 场景3：数据库批量查询或跨对象对比

2. **每个场景必须包含**：
   - `print("=" * 60)` 分隔线
   - `print("场景X：XXX")` 标题
   - `print("问题描述：XXX")` 说明这个场景在做什么
   - 每个计算步骤前添加 `# 调用函数：function_name()` 注释
   - 使用 `print(f"✓ 场景X完成：...")` 标记完成

3. **函数调用必须可追溯**：
   - 场景1中必须展示函数间的调用链（如：函数B调用函数A）
   - 注释中明确写出：`# 调用函数：xxx()，该函数内部调用了 yyy()`

4. **输出格式要求⚠️**：
    **结构化函数调用输出**：每个计算步骤必须使用标准化格式，明确标注函数名、参数和结果：
      - 标准格式：`print(f"FUNCTION_CALL: {function_name} | PARAMS: {params} | RESULT: {result}")`
      - 简化格式：`print(f"[CALL] {function_name}({params}) -> {result}")`
      - 推荐示例：`print(f"FUNCTION_CALL: calculate_velocity | PARAMS: time=5.0, acceleration=9.8 | RESULT: 49.0")`
    **文件生成输出**：如果代码生成了文件，必须打印文件路径：
      - 格式：`print(f"FILE_GENERATED: {file_type} | PATH: {file_path}")`
      - 示例：`print(f"FILE_GENERATED: Plot | PATH: ./images/velocity_plot.png")`
    **最终答案输出**：必须使用标准格式输出最终答案：
        - 格式：`print(f"FINAL_ANSWER: {answer}")`
    **输出顺序**：确保最终答案在所有输出的最后一行
    **便于提取**：所有关键输出都使用大写关键字标识，便于正则表达式提取

---

## ⚠️ 关键改进点

1. **多场景适配**：main函数必须展示工具在不同参数/问题下的应用
2. **专业库强制**：明确列出物理/化学/材料的专属库，不能只用通用库
3. **Function Calling格式**：返回dict而非单一值，包含metadata
4. **工具调用链**：高层函数调用低层函数，体现工具生态
5. **数据库集成**：必须展示至少一个免费数据库的访问
6. **🆕 注释规范**：每个函数调用前必须用注释标明函数名和调用关系