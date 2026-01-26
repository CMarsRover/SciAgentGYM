"""
简单回归测试：验证工具加载器在处理带有复杂类（如 Arrow3D）的工具文件时不会报错，
并能正确提取出函数型工具。

运行方式（在项目根目录）：

    python gym/test/test_tool_loader_arrow3d.py

预期行为：
- 不出现 "Arrow3D.__init__() missing ..." 之类的错误
- 能看到成功导入的函数列表
"""

from __future__ import annotations

from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中，以便可以导入 gym 包
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gym.core.tool_loader import dynamic_import_tool_functions


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    # 这里直接使用 src/tools 版本的电磁场求解器
    tool_path = "src/tools/electromagnetic_field_solver_157.py"

    print(f"🔍 项目根目录: {project_root}")
    print(f"🔧 测试工具文件: {tool_path}")

    # subject / topic 仅用于路径推断，这里给出与目录结构一致的值，便于未来扩展
    functions = dynamic_import_tool_functions(
        tool_path=tool_path,
        subject="Physics",
        topic="Electromagnetism",
    )

    print("\n=== 导入结果 ===")
    if not functions:
        print("⚠️ 未导入到任何函数，请检查路径或工具文件。")
    else:
        names = sorted(functions.keys())
        print(f"✅ 共导入 {len(names)} 个函数：")
        for name in names:
            print(f" - {name}")


if __name__ == "__main__":
    main()

