# 工具封装与LLM调用指南

## 📚 目录

1. [概述](#概述)
2. [工具封装](#工具封装)
3. [工具注册](#工具注册)
4. [LLM工具调用](#llm工具调用)
5. [完整示例](#完整示例)
6. [最佳实践](#最佳实践)

---

## 概述

本指南介绍如何：
- 将现有函数封装为 `EnvironmentTool` 工具类
- 使用 `Toolbox` 注册系统注册工具
- 构建 OpenAI 风格的 tools schema 供 LLM 调用
- 实现完整的 function calling 流程

### 系统架构

```
原始函数 → EnvironmentTool封装 → Toolbox注册 → LLM调用
```

---

## 工具封装

### 1. 创建 EnvironmentTool 子类

所有工具都需要继承 `EnvironmentTool` 基类，并实现 `use()` 方法。

#### 基本结构

```python
from gym.tool import EnvironmentTool
from gym.etities import Observation
import json
import traceback

class MyTool(EnvironmentTool):
    """工具描述"""
    
    # 工具元数据
    name = "my_tool"
    description = "工具的功能描述"
    arguments = {
        "param1": {
            "type": "string",
            "description": "参数1的描述"
        },
        "param2": {
            "type": "number",
            "description": "参数2的描述"
        }
    }
    
    def use(self, environment, action) -> Observation:
        """执行工具操作"""
        try:
            # 解析参数
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            # 获取参数值
            param1 = args.get("param1")
            param2 = args.get("param2", 0)  # 默认值
            
            # 参数验证
            if not param1:
                return Observation(self.name, "错误: 缺少必需参数 param1")
            
            # 调用原始函数
            result = original_function(param1, param2)
            
            # 返回结果（JSON格式）
            return Observation(
                self.name, 
                json.dumps(result, ensure_ascii=False, indent=2)
            )
        
        except Exception as e:
            return Observation(
                self.name, 
                f"错误: {str(e)}\n{traceback.format_exc()}"
            )
```

### 2. 参数处理

#### 参数类型定义

```python
arguments = {
    "smiles": {
        "type": "string",
        "description": "SMILES字符串或化学名称"
    },
    "method": {
        "type": "string",
        "description": "方法选择",
        "enum": ["ETKDG", "ETKDGv3", "basic"]  # 枚举类型
    },
    "max_iters": {
        "type": "integer",
        "description": "最大迭代次数，默认200"
    },
    "molecules": {
        "type": "object",
        "description": "分子列表或字典"
    }
}
```

#### 参数解析模式

```python
def use(self, environment, action) -> Observation:
    # 模式1: action 是参数字典
    if isinstance(action, dict):
        args = action.get("arguments", action)
    else:
        args = action if isinstance(action, dict) else {}
    
    # 模式2: 直接使用 action（如果已经是字典）
    args = action if isinstance(action, dict) else {}
    
    # 获取参数（带默认值）
    param = args.get("param_name", default_value)
```

### 3. 结果返回

#### 成功返回

```python
# 返回JSON格式的结果
result = {
    "status": "success",
    "data": computed_data,
    "message": "操作成功"
}
return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
```

#### 错误处理

```python
try:
    # 执行操作
    result = some_operation()
except Exception as e:
    return Observation(
        self.name, 
        f"错误: {str(e)}\n{traceback.format_exc()}"
    )
```

### 4. 实际示例

```python
from gym.tool import EnvironmentTool
from gym.etities import Observation
from .molecule_analyzer import get_3d_properties
import json
import traceback

@Toolbox.register(name="get_3d_properties")
class Get3DPropertiesTool(EnvironmentTool):
    """计算分子3D几何性质和形状描述符工具"""
    
    name = "get_3d_properties"
    description = "计算分子的3D几何性质和形状描述符，包括主惯性矩、归一化主矩比、分子形状分类等"
    arguments = {
        "smiles": {"type": "string", "description": "SMILES字符串或化学名称"},
        "method": {"type": "string", "description": "3D坐标生成方法：'ETKDG', 'ETKDGv3', 'basic'，默认'ETKDGv3'"},
        "conf_id": {"type": "integer", "description": "构象ID，默认0"}
    }
    
    def use(self, environment, action) -> Observation:
        """计算3D性质"""
        try:
            if isinstance(action, dict):
                args = action.get("arguments", action)
            else:
                args = action if isinstance(action, dict) else {}
            
            smiles = args.get("smiles")
            method = args.get("method", "ETKDGv3")
            conf_id = args.get("conf_id", 0)
            
            if not smiles:
                return Observation(self.name, "错误: 缺少必需参数 smiles")
            
            # 调用原始函数
            result = get_3d_properties(smiles, method, conf_id)
            
            # 转换不可序列化的类型
            if isinstance(result, dict):
                if 'pmi' in result and isinstance(result['pmi'], tuple):
                    result['pmi'] = list(result['pmi'])
                if 'npr' in result and isinstance(result['npr'], tuple):
                    result['npr'] = list(result['npr'])
            
            return Observation(self.name, json.dumps(result, ensure_ascii=False, indent=2))
        
        except Exception as e:
            return Observation(self.name, f"错误: {str(e)}\n{traceback.format_exc()}")
```

---

## 工具注册

### 1. 使用 Toolbox.register 装饰器

最简单的方式是使用 `@Toolbox.register()` 装饰器：

```python
from gym.toolbox import Toolbox
from gym.tool import EnvironmentTool

@Toolbox.register(name="my_tool")
class MyTool(EnvironmentTool):
    name = "my_tool"
    description = "工具描述"
    # ... 其他代码
```

### 2. 自动注册

当模块被导入时，装饰器会自动执行，工具会被注册到 `Toolbox._tool_registry`：

```python
# 导入模块即可触发注册
import toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_gym

# 工具已自动注册，可以直接使用
from gym.toolbox import Toolbox
tool = Toolbox.get_tool("my_tool")
```

### 3. 注册验证

```python
from gym.toolbox import Toolbox

# 检查工具是否已注册
if "my_tool" in Toolbox._tool_registry:
    print("工具已注册")

# 获取所有已注册的工具
registered_tools = list(Toolbox._tool_registry.keys())
print(f"已注册 {len(registered_tools)} 个工具")
```

### 4. 工具获取

```python
from gym.toolbox import Toolbox

# 获取工具实例
tool = Toolbox.get_tool("my_tool")

# 获取工具类
tool_cls, config_cls = Toolbox._tool_registry["my_tool"]
```

---

## LLM工具调用

### 1. 构建 Tools Schema

从 `Toolbox` 注册表构建 OpenAI 风格的 tools schema：

```python
from gym.toolbox import Toolbox

def build_tools_schema_from_gym_tools():
    """从 Toolbox 注册表构建 OpenAI tools schema"""
    tools = []
    
    # 需要使用的工具名称列表
    tool_names = [
        "chem_visualizer",
        "optimize_geometry",
        "get_3d_properties",
    ]
    
    for tool_name in tool_names:
        # 从 Toolbox 获取工具实例
        try:
            tool = Toolbox.get_tool(tool_name)
        except ValueError:
            print(f"[WARN] 工具 {tool_name} 未在 Toolbox 中注册，跳过")
            continue
        
        # 构建 OpenAI 风格的 tool schema
        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        
        # 转换 arguments 为 OpenAI parameters 格式
        if tool.arguments:
            for param_name, param_info in tool.arguments.items():
                if isinstance(param_info, dict):
                    param_schema = {
                        "type": param_info.get("type", "string"),
                        "description": param_info.get("description", ""),
                    }
                    
                    # 处理枚举类型
                    if "enum" in param_info:
                        param_schema["enum"] = param_info["enum"]
                    
                    tool_schema["function"]["parameters"]["properties"][param_name] = param_schema
                    
                    # 设置必需参数
                    if tool.name == "chem_visualizer" and param_name == "molecules":
                        tool_schema["function"]["parameters"]["required"].append(param_name)
                    elif tool.name in ["optimize_geometry", "get_3d_properties"]:
                        if param_name in ["smiles", "method"]:
                            if param_name not in tool_schema["function"]["parameters"]["required"]:
                                tool_schema["function"]["parameters"]["required"].append(param_name)
        
        tools.append(tool_schema)
    
    return tools
```

### 2. 构建工具注册表

创建工具实例映射，用于执行调用：

```python
def build_tool_registry():
    """从 Toolbox 注册表构建工具实例映射"""
    registry = {}
    
    tool_names = [
        "chem_visualizer",
        "optimize_geometry",
        "get_3d_properties",
    ]
    
    for tool_name in tool_names:
        try:
            tool = Toolbox.get_tool(tool_name)
            registry[tool.name] = tool
        except ValueError:
            print(f"[WARN] 工具 {tool_name} 未在 Toolbox 中注册，跳过")
            continue
    
    return registry
```

### 3. 执行工具调用

```python
def run_tool_call(tool, action):
    """执行工具调用，返回 JSON 可序列化的结果"""
    try:
        mock_env = MockEnvironment()  # 模拟环境对象
        observation = tool.use(mock_env, action)
        
        # 解析 observation 中的结果
        try:
            result = json.loads(observation.observation)
            return {
                "status": "success",
                "result": result,
                "raw_observation": observation.observation,
            }
        except json.JSONDecodeError:
            return {
                "status": "success",
                "result": observation.observation,
                "raw_observation": observation.observation,
            }
    except Exception as e:
        import traceback as tb
        return {
            "status": "error",
            "error": str(e),
            "traceback": tb.format_exc(),
        }
```

### 4. LLM Function Calling 流程

```python
from openai import OpenAI

def solve_problem_with_llm(question: str, client: OpenAI, model: str):
    """使用 LLM 和工具解决问题"""
    
    # 1. 构建工具 schema 和注册表
    tools = build_tools_schema_from_gym_tools()
    tool_registry = build_tool_registry()
    
    # 2. 构建初始消息
    messages = [
        {
            "role": "system",
            "content": "你是一个专家助手。请使用提供的工具来解决问题。",
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    
    # 3. Function calling 循环
    max_steps = 20
    for step in range(max_steps):
        # 调用 LLM
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        msg = resp.choices[0].message
        
        # 如果没有工具调用，说明已给出最终答案
        if not msg.tool_calls:
            print(f"[最终答案] {msg.content}")
            break
        
        # 执行工具调用
        tool_messages = []
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # 获取工具
            tool = tool_registry.get(func_name)
            if tool:
                # 执行工具
                tool_result = run_tool_call(tool, args)
            else:
                tool_result = {"status": "error", "error": f"未找到工具: {func_name}"}
            
            # 构建工具消息
            tool_messages.append({
                "role": "assistant",
                "tool_calls": [tool_call],
                "content": None,
            })
            tool_messages.append({
                "role": "tool",
                "name": func_name,
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
        
        # 将工具结果添加到消息列表
        messages.extend(tool_messages)
```

---

## 完整示例

### 示例：分子形状描述符计算

完整代码示例请参考 `func_calling_cases_tool.py`：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 analytical_chemistry_tools_gym.py 封装的工具进行 function calling 示例
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI

# 导入 Toolbox 注册系统
from gym.toolbox import Toolbox

# 导入工具模块以触发注册
import toolkits.chemistry.analytical_chemistry.analytical_chemistry_tools_gym

# API 配置
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.example.com/v1")
MODEL_NAME = os.getenv("FUNCALL_MODEL_NAME", "gpt-4")

class MockEnvironment:
    """模拟环境对象"""
    pass

def build_tools_schema_from_gym_tools():
    """从 Toolbox 注册表构建 OpenAI tools schema"""
    tools = []
    tool_names = ["chem_visualizer", "optimize_geometry", "get_3d_properties"]
    
    for tool_name in tool_names:
        try:
            tool = Toolbox.get_tool(tool_name)
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
            # ... 转换 arguments 为 parameters ...
            tools.append(tool_schema)
        except ValueError:
            continue
    
    return tools

def build_tool_registry():
    """从 Toolbox 注册表构建工具实例映射"""
    registry = {}
    tool_names = ["chem_visualizer", "optimize_geometry", "get_3d_properties"]
    
    for tool_name in tool_names:
        try:
            tool = Toolbox.get_tool(tool_name)
            registry[tool.name] = tool
        except ValueError:
            continue
    
    return registry

def run_tool_call(tool, action):
    """执行工具调用"""
    try:
        mock_env = MockEnvironment()
        observation = tool.use(mock_env, action)
        try:
            result = json.loads(observation.observation)
            return {"status": "success", "result": result}
        except json.JSONDecodeError:
            return {"status": "success", "result": observation.observation}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """主函数"""
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    # 构建工具
    tools = build_tools_schema_from_gym_tools()
    tool_registry = build_tool_registry()
    
    # 构建消息
    messages = [
        {"role": "system", "content": "你是一个计算化学专家..."},
        {"role": "user", "content": "问题描述..."},
    ]
    
    # Function calling 循环
    for step in range(20):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(f"[最终答案] {msg.content}")
            break
        
        # 执行工具调用
        tool_messages = []
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            tool = tool_registry.get(func_name)
            tool_result = run_tool_call(tool, args) if tool else {"error": "未找到工具"}
            
            tool_messages.append({
                "role": "assistant",
                "tool_calls": [tool_call],
                "content": None,
            })
            tool_messages.append({
                "role": "tool",
                "name": func_name,
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })
        
        messages.extend(tool_messages)

if __name__ == "__main__":
    main()
```

---

## 最佳实践

### 1. 工具封装

✅ **推荐做法**：
- 使用清晰的工具名称和描述
- 提供完整的参数文档
- 处理所有可能的异常情况
- 返回结构化的 JSON 结果

❌ **避免**：
- 直接返回 Python 对象（需要序列化）
- 忽略错误处理
- 使用模糊的参数名称

### 2. 工具注册

✅ **推荐做法**：
- 使用 `@Toolbox.register()` 装饰器
- 在模块导入时自动注册
- 使用有意义的工具名称

❌ **避免**：
- 手动管理注册表
- 重复注册同名工具
- 使用过于复杂的名称

### 3. LLM 调用

✅ **推荐做法**：
- 从 `Toolbox` 统一获取工具
- 构建清晰的 system prompt
- 处理工具调用错误
- 限制最大调用次数

❌ **避免**：
- 直接硬编码工具列表
- 忽略工具调用失败
- 无限循环调用

### 4. 错误处理

```python
# 好的错误处理
try:
    result = tool.use(env, action)
    return result
except Exception as e:
    return Observation(
        tool.name,
        json.dumps({
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
    )
```

### 5. 参数验证

```python
# 参数验证示例
def use(self, environment, action) -> Observation:
    args = action if isinstance(action, dict) else {}
    
    # 必需参数检查
    required_params = ["smiles", "method"]
    missing = [p for p in required_params if p not in args or not args[p]]
    if missing:
        return Observation(
            self.name,
            f"错误: 缺少必需参数: {', '.join(missing)}"
        )
    
    # 参数类型验证
    if not isinstance(args["smiles"], str):
        return Observation(self.name, "错误: smiles 必须是字符串")
    
    # 继续执行...
```

---

## 总结

1. **工具封装**：继承 `EnvironmentTool`，实现 `use()` 方法
2. **工具注册**：使用 `@Toolbox.register()` 装饰器自动注册
3. **LLM调用**：从 `Toolbox` 构建 schema 和注册表，实现 function calling 流程

通过这个流程，你可以轻松地将任何函数封装为工具，并让 LLM 自动调用它们来解决问题。










