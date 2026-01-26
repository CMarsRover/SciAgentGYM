import sys
from pathlib import Path
from typing import Any,Optional,Dict,List
# 添加项目根目录到 Python 路径，以便可以导入 gym 包
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gym.utils.client_manager import get_client, list_models, DEFAULT_MODEL, test_all_models
from gym.config.config import TOOL_TRACE_SUFFIX
# 测试客户端管理器
print("=== 多模型客户端管理器测试 ===")
print(f"支持的模型: {list_models()}")

# 测试默认模型
print(f"\n测试默认模型: {DEFAULT_MODEL}")
try:
    client = get_client()
    print(f"默认模型客户端初始化成功: {client.model_name}")
except Exception as e:
    print(f"默认模型客户端初始化失败: {e}")

# 测试所有模型的可用性
# print("\n测试所有模型的可用性...")
# test_results = test_all_models()
# for model, is_available in test_results.items():
#     status = "✓" if is_available else "✗"
#     print(f"{status} {model}") 

def _coerce_truthy_flag(value: Any) -> bool:
    """辅助函数，将常见格式的标志位转换为布尔值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "y", "on"}
    return False


def _should_use_react_prompt(use_tools: bool, mode_name: Optional[str], metadata: Dict[str, Any]) -> bool:
    if not use_tools:
        return False
    suffix = (TOOL_TRACE_SUFFIX or "").strip()
    explicit_flag = metadata.get("use_react_prompt")
    if explicit_flag is not None:
        return _coerce_truthy_flag(explicit_flag)
    if mode_name:
        lowered = mode_name.lower()
        if "react" in lowered:
            return True
        if suffix and mode_name.endswith(suffix):
            return True
    # 默认使用 ReAct 流程，除非显式关闭
    return True

REACT_TOOL_SYSTEM_PROMPT = (
    "你是一个顶尖的 AI 科学助手，你的任务是解决来自物理、化学、生物等领域的复杂、多模态、多步骤问题。\n"
    "\n"
    "# 任务指令：两阶段方法\n"
    "\n"
    "你必须严格遵循以下两个阶段来回答问题：\n"
    "\n"
    "**阶段 1: 规划 (Planning)**\n"
    "在开始执行任何操作之前，你必须首先生成一个高阶的、分步骤的“计划”。这个计划应该概述你打算如何分解问题、使用哪些工具、以及工具之间的依赖关系。\n"
    "\n"
    "**阶段 2: 执行 (Execution)**\n"
    "在提交计划后，你将开始逐一执行。你必须严格遵循以下的 \"ReAct\" 格式 (Thought, Action, Observation)，直到你得出最终答案。\n"
    "\n"
    "---\n"
    "\n"
    "**重要：可视化要求**\n"
    "\n"
    "在解决问题的过程中，你**必须**调用可视化/绘图相关的工具（如 `visualize_*`、`plot_*`、`draw_*` 等）来生成图表或图像，以便：\n"
    "1. 直观展示计算结果（如光谱图、分子结构图、数据曲线等）\n"
    "2. 验证和确认你的计算是否正确\n"
    "3. 帮助理解问题的物理/化学含义\n"
    "\n"
    "即使问题没有明确要求画图，你也应该主动使用可视化工具来辅助分析和呈现结果。\n"
    "\n"
    "---\n"
    "\n"
    "**输出格式规范**\n"
    "\n"
    "**[阶段 1: 规划]**\n"
    "Plan:\n"
    "\n"
    "1. [你的计划步骤 1，例如：分析输入图像中的分子结构]\n"
    "2. [你的计划步骤 2，例如：使用 `analyze_molecule` 工具获取 SMILES]\n"
    "3. [你的计划步骤 3，例如：将 SMILES 传入 `calculate_properties` 工具]\n"
    "4. [你的计划步骤 4，例如：使用 `visualize_*` 或 `plot_*` 工具绘制结果图表]\n"
    "5. ...\n"
    "6. [你的计划步骤 N，例如：总结所有属性并回答问题]\n"
    "\n"
    "**[阶段 2: 执行]**\n"
    "Thought: [描述你当前的思考，你正处于计划的第几步，以及你为什么需要调用这个特定的工具。]\n"
    "Action: 请你在这一步发出工具调用\n"
    "\n"
    "[在你提交 Action 后，你将收到一个 Observation]\n"
    "\n"
    "Observation: [这里将由评测系统插入工具的返回结果]\n"
    "\n"
    "[然后你重复这个循环]\n"
    "\n"
    "Thought: [基于上一步的 Observation，你的新思考。]\n"
    "Action: 请你在这一步发出工具调用\n"
    "\n"
    "...\n"
    "[当你收集到所有信息并准备好回答时]\n"
    "\n"
    "Thought: [我已收集到所有必要信息，现在可以生成最终答案了。]\n"
    "Final Answer: [这里是对原始问题的最终、完整回答。]"
)

