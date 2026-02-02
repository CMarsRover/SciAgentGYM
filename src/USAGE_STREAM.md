# 科学工具生成器 - 流式/非流式使用说明

## 概述

科学工具生成器支持两种输出模式：
- **非流式模式（默认）**：一次性返回完整响应，适合短响应或快速处理
- **流式模式**：逐块返回响应，适合长响应或避免超时

## 使用方式

### 1. 命令行使用

#### 非流式模式（默认）

```bash
# 使用默认的非流式模式
python src/sci_tool_generator.py --data-file ./data.json

# 或者显式指定不使用流式
python src/sci_tool_generator.py --data-file ./data.json
```

#### 流式模式

```bash
# 使用流式模式（添加 --stream 参数）
python src/sci_tool_generator.py --data-file ./data.json --stream
```

### 2. 代码中使用

#### 非流式模式（默认）

```python
from sci_tool_generator import generate_sci_tool

# 默认使用非流式模式
result = generate_sci_tool(
    query="你的问题",
    answer="标准答案",
    image_paths=["path/to/image.png"],
    subfield="chemistry"
)

# 或者显式指定 use_stream=False
result = generate_sci_tool(
    query="你的问题",
    answer="标准答案",
    image_paths=["path/to/image.png"],
    subfield="chemistry",
    use_stream=False
)
```

#### 流式模式

```python
from sci_tool_generator import generate_sci_tool

# 使用流式模式
result = generate_sci_tool(
    query="你的问题",
    answer="标准答案",
    image_paths=["path/to/image.png"],
    subfield="chemistry",
    use_stream=True  # 启用流式输出
)
```

## 模式对比

### 非流式模式

**优点：**
- 响应速度快（一次性返回）
- 代码简洁
- 适合短响应（< 5000 tokens）

**缺点：**
- 长响应可能超时
- 无法实时看到进度

**适用场景：**
- 第一轮分析（通常响应较短）
- 搜索总结（通常响应中等）
- 快速测试和调试

### 流式模式

**优点：**
- 避免长响应超时
- 可以实时看到进度
- 适合长响应（> 5000 tokens）

**缺点：**
- 响应速度稍慢（需要等待所有块）
- 代码稍微复杂

**适用场景：**
- 第二轮工具生成（通常响应很长）
- 复杂问题的处理
- 生产环境中的长任务

## 函数签名

```python
def generate_sci_tool(
    query: str,
    answer: str,
    image_paths: List[str] = None,
    search_results: str = None,
    subfield: str = None,
    use_stream: bool = False  # 新增参数：是否使用流式输出
) -> List[ConversationTurn]:
    """生成科学工具的主函数
    
    Args:
        query: 科学问题
        answer: 标准答案
        image_paths: 图片路径列表
        search_results: 预定义的搜索结果（可选）
        subfield: 学科子领域
        use_stream: 是否使用流式输出（默认False，使用非流式。设置为True可避免长响应超时）
    
    Returns:
        完整的对话历史列表
    """
```

## 实际应用建议

### 推荐配置

1. **开发/测试阶段**：使用非流式模式
   ```bash
   python src/sci_tool_generator.py --data-file ./test_data.json
   ```

2. **生产环境/长任务**：使用流式模式
   ```bash
   python src/sci_tool_generator.py --data-file ./production_data.json --stream
   ```

3. **混合使用**（高级用法）：
   - 第一轮分析：非流式（响应短）
   - 搜索总结：非流式（响应中等）
   - 工具生成：流式（响应长）

### 性能对比

| 模式 | 响应时间 | 超时风险 | 适用场景 |
|------|----------|----------|----------|
| 非流式 | 快 | 高（长响应） | 短响应、测试 |
| 流式 | 稍慢 | 低 | 长响应、生产 |

## 示例

### 示例1：非流式模式

```python
# 处理简单问题（非流式）
result = generate_sci_tool(
    query="计算1+1等于几？",
    answer="2",
    subfield="mathematics",
    use_stream=False
)
```

### 示例2：流式模式

```python
# 处理复杂问题（流式）
result = generate_sci_tool(
    query="设计一个完整的蛋白质结构预测工具",
    answer="...",
    image_paths=["protein_structure.png"],
    subfield="bioinformatics",
    use_stream=True  # 使用流式避免超时
)
```

## 注意事项

1. **默认行为**：如果不指定 `use_stream` 参数，默认使用非流式模式
2. **超时处理**：如果遇到超时错误，建议切换到流式模式
3. **进度显示**：流式模式会显示实时进度，非流式模式只在完成后显示结果
4. **API限制**：根据API的token限制，选择合适的模式

## 故障排查

### 问题1：遇到超时错误

**解决方案**：切换到流式模式
```bash
python src/sci_tool_generator.py --data-file ./data.json --stream
```

### 问题2：响应速度慢

**解决方案**：如果响应较短，使用非流式模式
```bash
python src/sci_tool_generator.py --data-file ./data.json
```

### 问题3：无法看到进度

**解决方案**：使用流式模式可以看到实时进度
```bash
python src/sci_tool_generator.py --data-file ./data.json --stream
```

## 更新日志

- **v1.1.0**：添加流式/非流式模式切换功能
- **v1.0.0**：初始版本，仅支持非流式模式

