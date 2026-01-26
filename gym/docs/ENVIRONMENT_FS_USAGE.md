# 环境文件系统使用指南

## 概述

统一的环境文件系统管理模块 (`gym.core.environment_fs`) 用于管理所有中间结果文件，避免工具中使用绝对路径导致文件存放位置混乱。

所有中间结果统一存放在 `gym/mid_result/` 目录下，按领域和题目ID细分。

## 目录结构

```
gym/mid_result/
├── structural_biology/
│   ├── case_001/
│   │   ├── result1.json
│   │   └── result2.pkl
│   └── case_002/
│       └── analysis.json
├── physics/
│   └── calculation.json
└── chemistry/
    └── reaction_data.json
```

## 基本使用

### 1. 导入模块

```python
from gym.core.environment_fs import (
    save_mid_result,
    load_mid_result,
    get_mid_result_path,
    get_environment_fs
)
```

### 2. 保存中间结果

```python
# 基本用法
result = save_mid_result(
    domain="structural_biology",
    filename="protein_analysis",
    data={"sequence": "ATCG", "length": 4}
)

if result["success"]:
    print(f"保存成功: {result['filepath']}")
else:
    print(f"保存失败: {result['error']}")

# 按题目ID细分
result = save_mid_result(
    domain="structural_biology",
    filename="structure_data",
    data={"pdb_id": "1ABC", "resolution": 2.5},
    case_id="case_001"  # 会保存到 structural_biology/case_001/
)

# 使用不同格式
result = save_mid_result(
    domain="physics",
    filename="matrix_data",
    data=[[1, 2], [3, 4]],
    format="pickle"  # 支持 json, pickle, txt
)
```

### 3. 加载中间结果

```python
# 基本用法
result = load_mid_result(
    domain="structural_biology",
    filename="protein_analysis"
)

if result["success"]:
    data = result["data"]
    print(f"加载成功: {data}")
else:
    print(f"加载失败: {result['error']}")

# 从特定题目ID目录加载
result = load_mid_result(
    domain="structural_biology",
    filename="structure_data",
    case_id="case_001"
)

# 自动检测格式（会尝试 .json, .pkl, .txt）
result = load_mid_result(
    domain="physics",
    filename="matrix_data"
    # format 参数可选，会自动检测
)
```

### 4. 获取文件路径

```python
# 获取文件路径（不创建文件）
filepath = get_mid_result_path(
    domain="structural_biology",
    filename="analysis",
    case_id="case_001",
    format="json"
)

print(f"文件路径: {filepath}")
# 输出: gym/mid_result/structural_biology/case_001/analysis.json
```

### 5. 高级用法

```python
from gym.core.environment_fs import get_environment_fs

# 获取文件系统实例
fs = get_environment_fs()

# 列出所有文件
files = fs.list_results(
    domain="structural_biology",
    case_id="case_001",
    pattern="*.json"  # 可选的文件名模式
)

# 删除文件
delete_result = fs.delete_result(
    domain="structural_biology",
    filename="old_result",
    case_id="case_001"
)

# 获取领域目录
domain_dir = fs.get_domain_dir("structural_biology", "case_001")
print(f"目录: {domain_dir}")
```

## 在工具中使用

### 替换旧的 save_mid_result 函数

**旧代码（不推荐）:**
```python
import os
import json

MID_RESULT_DIR = "./mid_result/chemistry"
os.makedirs(MID_RESULT_DIR, exist_ok=True)

def save_mid_result(subject: str, filename: str, data: dict) -> dict:
    dir_path = f"./mid_result/{subject}"
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, f"{filename}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return {"result": filepath}
```

**新代码（推荐）:**
```python
from gym.core.environment_fs import save_mid_result

def save_mid_result(subject: str, filename: str, data: dict, case_id: str = None) -> dict:
    result = save_mid_result(
        domain=subject,
        filename=filename,
        data=data,
        case_id=case_id
    )
    if result["success"]:
        return {"result": result["filepath"]}
    else:
        return {"result": None, "error": result["error"]}
```

## 支持的领域

- `physics` - 物理学
- `chemistry` - 化学
- `materials` - 材料科学
- `astronomy` - 天文学
- `geography` - 地理学
- `structural_biology` - 结构生物学
- `molecular_biology` - 分子生物学
- `quantum_physics` - 量子物理学
- `life_science` - 生命科学
- `earth_science` - 地球科学
- `computer_science` - 计算机科学

注意：如果使用未列出的领域名称，系统仍然会创建目录，但建议使用标准领域名称。

## 文件格式

- **JSON** (`.json`): 用于结构化数据，人类可读
- **Pickle** (`.pkl`): 用于复杂Python对象，二进制格式
- **Text** (`.txt`): 用于纯文本数据

## 最佳实践

1. **始终使用领域名称**: 明确指定 `domain` 参数
2. **使用题目ID细分**: 对于特定题目的结果，使用 `case_id` 参数
3. **检查返回结果**: 始终检查 `success` 字段
4. **统一格式**: 在同一工具中尽量使用相同的文件格式
5. **清理临时文件**: 使用 `delete_result` 清理不再需要的文件

## 示例：完整工具函数

```python
from gym.core.environment_fs import save_mid_result, load_mid_result

def analyze_protein_structure(pdb_id: str, case_id: str):
    """分析蛋白质结构"""
    
    # 检查是否已有缓存结果
    cached = load_mid_result(
        domain="structural_biology",
        filename=f"analysis_{pdb_id}",
        case_id=case_id
    )
    
    if cached["success"]:
        print("使用缓存结果")
        return cached["data"]
    
    # 执行分析
    analysis_result = {
        "pdb_id": pdb_id,
        "residues": 150,
        "resolution": 2.5,
        "chains": ["A", "B"]
    }
    
    # 保存结果
    save_result = save_mid_result(
        domain="structural_biology",
        filename=f"analysis_{pdb_id}",
        data=analysis_result,
        case_id=case_id
    )
    
    if save_result["success"]:
        print(f"结果已保存: {save_result['filepath']}")
        return analysis_result
    else:
        print(f"保存失败: {save_result['error']}")
        return None
```

## 注意事项

1. **路径安全**: 文件名和 case_id 中的特殊字符会被自动清理
2. **目录自动创建**: 所有必要的目录会自动创建
3. **线程安全**: 全局单例实例，可在多线程环境中使用
4. **错误处理**: 所有函数都返回包含 `success` 字段的字典，便于错误处理
