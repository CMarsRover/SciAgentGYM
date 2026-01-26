# 环境文件系统集成指南

## 概述

环境文件系统已集成到环境构建和工具注册流程中。当创建环境时，会自动初始化文件系统，并根据 case_id 和 domain 自动组织目录结构。

## 自动集成

### 1. 环境构建时自动初始化

当调用 `register_tools_to_env` 或创建 `MinimalSciEnv` 时，环境文件系统会自动初始化：

```python
from gym.tool_loader import register_tools_to_env

# 方式1：显式传递 case_id 和 domain
env, tools, schema, registry = register_tools_to_env(
    tool_protocols,
    function_map,
    case_id="case_001",
    domain="structural_biology"
)

# 方式2：通过 query_data 自动提取
env, tools, schema, registry = register_tools_to_env(
    tool_protocols,
    function_map,
    query_data=query_data  # 会自动提取 case_id 和 domain
)
```

### 2. 自动提取 case_id 和 domain

如果提供了 `query_data`，系统会自动从 metadata 中提取：

- **case_id**: 从以下字段提取（按优先级）：
  - `query_data["id"]`
  - `metadata["id"]`
  - `metadata["case_id"]`
  - `metadata["question_id"]`
  - `metadata["original_question_id"]`

- **domain**: 从 `metadata["subject"]` 提取并映射：
  - "structural biology" → "structural_biology"
  - "molecular biology" → "molecular_biology"
  - "quantum physics" → "quantum_physics"
  - 其他 → 转换为下划线格式

## 在工具中使用

### 方式1：通过环境访问文件系统

```python
from gym.tool import EnvironmentTool
from gym.etities import Observation

class MyTool(EnvironmentTool):
    name = "my_tool"
    
    def use(self, environment, action):
        # 通过环境访问文件系统
        fs = environment.file_system
        
        # 保存中间结果（会自动使用环境的 case_id 和 domain）
        result = fs.save_result(
            domain=environment._domain or "default",
            filename="my_result",
            data={"result": "..."},
            case_id=environment._case_id
        )
        
        return Observation(self.name, f"结果已保存: {result['filepath']}")
```

### 方式2：使用便捷函数（推荐）

```python
from gym.core.environment_fs import save_mid_result, load_mid_result

def my_tool_function(environment, **kwargs):
    # 从环境获取 case_id 和 domain
    case_id = getattr(environment, '_case_id', None)
    domain = getattr(environment, '_domain', None) or "default"
    
    # 保存结果
    save_result = save_mid_result(
        domain=domain,
        filename="analysis_result",
        data={"data": "..."},
        case_id=case_id
    )
    
    if save_result["success"]:
        return f"保存成功: {save_result['filepath']}"
    else:
        return f"保存失败: {save_result['error']}"
```

## 目录结构

根据 case_id 和 domain，文件会自动组织到以下目录：

```
gym/mid_result/
├── structural_biology/
│   ├── case_001/
│   │   ├── result1.json
│   │   └── result2.pkl
│   └── case_002/
│       └── analysis.json
└── physics/
    └── calculation.json
```

## 示例：完整工具实现

```python
from gym.tool import EnvironmentTool
from gym.etities import Observation
from gym.core.environment_fs import save_mid_result, load_mid_result

class ProteinAnalysisTool(EnvironmentTool):
    name = "analyze_protein"
    description = "分析蛋白质结构"
    arguments = {
        "pdb_id": {"type": "string", "description": "PDB ID"}
    }
    
    def use(self, environment, action):
        pdb_id = action.get("pdb_id")
        
        # 从环境获取上下文
        case_id = getattr(environment, '_case_id', None)
        domain = getattr(environment, '_domain', None) or "structural_biology"
        
        # 检查是否有缓存
        cached = load_mid_result(
            domain=domain,
            filename=f"analysis_{pdb_id}",
            case_id=case_id
        )
        
        if cached["success"]:
            return Observation(self.name, f"使用缓存结果: {cached['data']}")
        
        # 执行分析
        analysis_result = {
            "pdb_id": pdb_id,
            "residues": 150,
            "resolution": 2.5
        }
        
        # 保存结果
        save_result = save_mid_result(
            domain=domain,
            filename=f"analysis_{pdb_id}",
            data=analysis_result,
            case_id=case_id
        )
        
        if save_result["success"]:
            return Observation(
                self.name,
                f"分析完成，结果已保存: {save_result['filepath']}"
            )
        else:
            return Observation(
                self.name,
                f"分析失败: {save_result['error']}"
            )
```

## 迁移指南

### 旧代码（不推荐）

```python
import os
import json

MID_RESULT_DIR = "./mid_result/chemistry"
os.makedirs(MID_RESULT_DIR, exist_ok=True)

def save_result(filename, data):
    filepath = os.path.join(MID_RESULT_DIR, f"{filename}.json")
    with open(filepath, "w") as f:
        json.dump(data, f)
    return filepath
```

### 新代码（推荐）

```python
from gym.core.environment_fs import save_mid_result

def save_result(environment, filename, data):
    case_id = getattr(environment, '_case_id', None)
    domain = getattr(environment, '_domain', None) or "chemistry"
    
    result = save_mid_result(
        domain=domain,
        filename=filename,
        data=data,
        case_id=case_id
    )
    
    if result["success"]:
        return result["filepath"]
    else:
        raise Exception(f"保存失败: {result['error']}")
```

## 注意事项

1. **环境必须存在**: 工具函数需要接收 `environment` 参数才能访问文件系统
2. **自动目录组织**: 如果提供了 case_id 和 domain，文件会自动组织到对应目录
3. **向后兼容**: 如果没有提供 case_id 或 domain，文件会保存到领域根目录
4. **线程安全**: 文件系统使用全局单例，可在多线程环境中使用
