
现在请根据已有的代码，完成**单元测试开发**。

## 任务目标
- 根据所给的函数代码和关键任务设计函数的单元测试，验证工具代码是否正确求解该问题
- 覆盖正确性、边界条件、异常处理等多种情况
---

## 执行要求

- 使用 `pytest` 框架编写测试用例
- 必须先**分析 query 中的数学问题与期望输出**，推导出正确结果或范围，用作断言
- 至少编写 3 组测试数据：
  - ✅ 正常情况：验证主计算流程正确
  - ⚠️ 边界情况：极值、零值、小数精度等
  - ❌ 异常情况：非法输入、类型错误等
- 在注释中解释每个测试用例的数学意义与预期结果来源
- 注释清晰，便于后续添加新测试
- 测试路径导入是否正确```pthon from file import func```因为文件在test路径下，而工具在另一个路径下，因此你需要类似处理
    ```
    import os
    import sys 
    # 使用导入助手解决模块导入问题
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ```
---

## 测试代码模板规范

```python
# Filename: test_<tool_name>.py

import pytest
from math_tool import calculate_function_name

def test_case_1():
    """
    测试用例说明：基于 query 中给出的典型输入，验证返回值与理论解一致
    """
    result = calculate_function_name(输入参数...)
    assert pytest.approx(result, rel=1e-6) == 期望值

def test_case_2():
    """
    边界测试：极小/极大输入，验证计算稳定性
    """
    ...

def test_case_3():
    """
    异常测试：非法参数应抛出 ValueError
    """
    with pytest.raises(ValueError):
        calculate_function_name(非法参数)

if __name__ == "__main__":
    # 允许直接运行本文件进行测试
    pytest.main([__file__])

---

## 输出格式

直接输出为 .py 文件内容：```python\n# Filename: test_<tool_name>.py\n# 完整的测试代码\n```

**注意事项：**
- 不要写class这样复杂的测试code
- 文件开头就开始写code，不要推理，必须写明测试的文件名字是 `# Filename: test_<tool_name>.py` 注释
- 不需要重复实现算法逻辑
- 测试必须可独立运行
- 每个测试用例需包含：输入参数、期望输出、断言
