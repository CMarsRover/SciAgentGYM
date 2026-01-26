# Toolbox 工具注册系统使用说明

## 📚 目录

1. [系统概述](#系统概述)
2. [核心概念](#核心概念)
3. [工作原理详解](#工作原理详解)
4. [使用指南](#使用指南)
5. [实际应用示例](#实际应用示例)

---

## 系统概述

`Toolbox` 是一个工具注册系统，允许你通过装饰器自动注册工具类或函数，然后通过统一的接口获取和使用它们。

### 主要特点

✅ **装饰器模式** - 使用 `@Toolbox.register()` 轻松注册工具  
✅ **支持两种类型** - 类工具（需要实例化）和函数工具（直接调用）  
✅ **自动名称生成** - 不提供名称时自动从类名/函数名生成  
✅ **名称冲突检测** - 防止重复注册同名工具  
✅ **统一接口** - 通过 `get_tool()` 或 `get_function_tool()` 获取工具  

---

## 核心概念

### 1. 工具注册表 (`_tool_registry`)

```python
_tool_registry = {
    "工具名称": (工具对象, 配置类, 工具类型)
}
```

- **工具名称**: 注册时使用的唯一标识符
- **工具对象**: 实际的类或函数
- **配置类**: 可选的配置类（用于高级用法）
- **工具类型**: "class" 或 "function"

### 2. 注册装饰器 (`@Toolbox.register()`)

装饰器负责：
1. 生成或验证工具名称
2. 检测工具类型（类或函数）
3. 检查名称冲突
4. 将工具添加到注册表
5. 为工具对象添加元数据（`registered_name`, `tool_type`）

### 3. 工具获取方法

- **`get_tool(name, **kwargs)`**: 获取类工具并实例化
- **`get_function_tool(name)`**: 获取函数工具（不实例化）

---

## 工作原理详解

### 代码分析

让我们逐步分析 `Toolbox` 类的核心代码：

#### 1. 类变量：工具注册表

```python
_tool_registry: Dict[str, Tuple[Type, Optional[Type], str]] = {}
```

- 这是**类变量**（不是实例变量），所有 `Toolbox` 实例共享同一个注册表
- 字典键是工具名称（字符串）
- 字典值是元组：`(工具对象, 配置类, 工具类型)`

#### 2. `register()` 方法 - 装饰器工厂

```python
@classmethod
def register(cls, name: str = None, config_cls: Optional[Type] = None) -> Callable:
```

**第一步：返回装饰器函数**

```python
def decorator(subclass: Type) -> Type:
```

当你写 `@Toolbox.register()` 时：
1. Python 先调用 `register()`，它返回 `decorator` 函数
2. 然后 Python 用 `decorator` 装饰你的类/函数

**第二步：生成工具名称**

```python
name_ = name or subclass.__name__.lower().replace("tool", "")
```

- 如果提供了 `name`，直接使用
- 否则从类名生成：
  - `GasKineticsTool` → `gaskineticstool` → `gaskinetics`（移除 "tool"）
  - `CalculateEnergyTool` → `calculateenergytool` → `calculateenergy`

**第三步：检查重复注册**

```python
if name_ in cls._tool_registry:
    if subclass != cls._tool_registry[name_][0]:
        raise ValueError(f"Cannot register '{name_}' multiple times.")
    return subclass  # 如果是同一个对象，允许重复装饰
```

- 如果名称已存在：
  - 是同一个对象 → 允许（重复装饰是安全的）
  - 是不同的对象 → 抛出错误（防止意外覆盖）

**第四步：注册工具**

```python
cls._tool_registry[name_] = (subclass, config_cls)
subclass.registered_name = name_  # 添加元数据
return subclass  # 返回原对象（不改变它）
```

#### 3. `get_tool()` 方法 - 获取并实例化工具

```python
@classmethod
def get_tool(cls, name: str, **kwargs) -> Any:
```

**第一步：解析名称（支持变体）**

```python
base_name = name.split(":")[0]  # "my_tool:variant" → "my_tool"
```

**第二步：查找工具**

```python
if base_name not in cls._tool_registry:
    raise ValueError(f"Unknown tool {base_name}")
```

**第三步：获取工具类和实例化**

```python
tool_cls, _ = cls._tool_registry[base_name]  # 解包元组
return tool_cls(**kwargs)  # 实例化并返回
```

---

## 使用指南

### 方式1：注册类工具

```python
from gym.toolbox import Toolbox

@Toolbox.register(name="my_tool", tool_type="class")
class MyTool:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
    
    def execute(self):
        return f"执行工具: {self.param1}, {self.param2}"

# 使用
tool = Toolbox.get_tool("my_tool", param1="值1", param2="值2")
result = tool.execute()
```

**要点：**
- `tool_type="class"` 表示这是类工具
- `get_tool()` 会自动实例化类，并传递 `**kwargs` 给 `__init__`

### 方式2：注册函数工具

```python
@Toolbox.register(name="calculate_energy", tool_type="function")
def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """计算动能"""
    return 0.5 * mass * velocity ** 2

# 使用
func = Toolbox.get_function_tool("calculate_energy")
result = func(mass=10, velocity=5)
```

**要点：**
- `tool_type="function"` 表示这是函数工具
- `get_function_tool()` 直接返回函数，不实例化

### 方式3：自动名称生成

```python
@Toolbox.register()  # 不提供名称，自动生成
class GasKineticsTool:
    # "GasKineticsTool" → "gaskineticstool" → "gaskinetics"
    pass

# 使用自动生成的名称
tool = Toolbox.get_tool("gaskinetics")
```

### 方式4：批量注册现有函数

如果你的工具文件已经有很多函数，可以批量注册：

```python
from gym.toolbox import Toolbox

# 定义函数
def func1(x):
    return x ** 2

def func2(x, y):
    return x + y

# 批量注册
for func in [func1, func2]:
    Toolbox.register(name=func.__name__, tool_type="function")(func)

# 使用
f1 = Toolbox.get_function_tool("func1")
f2 = Toolbox.get_function_tool("func2")
```

---

## 实际应用示例

### 示例1：为现有工具文件添加注册

假设你有一个 `gas_kinetics.py` 文件：

```python
# gas_kinetics.py
from gym.toolbox import Toolbox

# 原始函数
def calculate_particle_kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity ** 2

# 添加注册装饰器
@Toolbox.register(name="particle_kinetic_energy", tool_type="function")
def calculate_particle_kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity ** 2
```

**或者，在文件末尾批量注册：**

```python
# gas_kinetics.py

def calculate_particle_kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity ** 2

def calculate_average_kinetic_energy(masses, velocities):
    # ... 实现
    pass

# 在文件末尾批量注册
if __name__ != "__main__":
    from gym.toolbox import Toolbox
    
    Toolbox.register(name="particle_kinetic_energy", tool_type="function")(
        calculate_particle_kinetic_energy
    )
    Toolbox.register(name="average_kinetic_energy", tool_type="function")(
        calculate_average_kinetic_energy
    )
```

### 示例2：在工具模块中自动注册

创建一个辅助函数：

```python
# utils.py
from gym.toolbox import Toolbox

def auto_register_module_functions(module, prefix=""):
    """
    自动注册模块中的所有函数
    
    Args:
        module: 模块对象（通过 import 获得）
        prefix: 名称前缀
    """
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("_") and not isinstance(obj, type):
            tool_name = f"{prefix}{name}" if prefix else name
            Toolbox.register(name=tool_name, tool_type="function")(obj)

# 使用
import gas_kinetics
auto_register_module_functions(gas_kinetics, prefix="gas_")
```

### 示例3：查询已注册的工具

```python
from gym.toolbox import Toolbox

# 列出所有工具
all_tools = Toolbox.list_tools()
for name, info in all_tools.items():
    print(f"{name}:")
    print(f"  类型: {info['type']}")
    print(f"  对象名: {info['object_name']}")
    print(f"  文档: {info['docstring'][:50]}...")

# 检查特定工具
if Toolbox.is_registered("my_tool"):
    print("工具已注册")
```

### 示例4：在环境系统中集成

```python
# science_environment.py
from gym.toolbox import Toolbox

class ScienceEnvironment:
    def __init__(self):
        # 导入工具模块（这会触发注册）
        import toolkits.physics.thermodynamics.gas_kinetics
        import toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools
        
        # 现在所有工具都已注册
        self.available_tools = Toolbox.list_tools()
    
    def execute_tool(self, tool_name: str, **kwargs):
        """执行工具"""
        if Toolbox.is_registered(tool_name):
            tool_info = Toolbox.list_tools()[tool_name]
            
            if tool_info['type'] == 'function':
                func = Toolbox.get_function_tool(tool_name)
                return func(**kwargs)
            else:
                tool = Toolbox.get_tool(tool_name, **kwargs)
                return tool.execute()
```

---

## 常见问题

### Q1: 为什么使用类变量而不是实例变量？

**A:** 类变量让所有 `Toolbox` 实例共享同一个注册表，这样无论在哪里获取工具，都能访问到所有已注册的工具。这是单例模式的变体。

### Q2: 装饰器中的 `subclass` 参数名为什么叫这个名字？

**A:** 这个名字是历史原因。实际上它可以是类或函数。在装饰器中：
- `subclass` 是被装饰的对象
- 装饰器返回的是同一个对象（不修改它）

### Q3: 为什么 `register()` 返回一个函数而不是直接注册？

**A:** 这是装饰器模式的标准做法：
1. `@Toolbox.register()` 调用 `register()`，返回 `decorator` 函数
2. Python 用 `decorator` 装饰类/函数
3. `decorator` 函数接收被装饰的对象，执行注册逻辑

### Q4: 如何处理工具依赖？

**A:** 如果你需要工具之间有依赖关系：

```python
@Toolbox.register(name="tool_a")
class ToolA:
    pass

@Toolbox.register(name="tool_b")
class ToolB:
    def __init__(self):
        # 在初始化时获取其他工具
        self.tool_a = Toolbox.get_tool("tool_a")
```

### Q5: 如何在运行时动态注册工具？

**A:** 你可以手动调用装饰器：

```python
def my_function(x):
    return x ** 2

# 动态注册
Toolbox.register(name="square", tool_type="function")(my_function)

# 现在可以使用
square_func = Toolbox.get_function_tool("square")
```

---

## 总结

`Toolbox` 注册系统提供了一种优雅的方式来管理你的工具：

1. **注册简单** - 使用装饰器 `@Toolbox.register()` 即可
2. **获取方便** - 通过统一接口 `get_tool()` 或 `get_function_tool()` 获取
3. **类型安全** - 区分类工具和函数工具，避免误用
4. **可扩展** - 支持自动名称生成、批量注册等高级用法

开始为你的工具添加注册功能吧！🚀
