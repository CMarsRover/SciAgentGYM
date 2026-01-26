from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from gym.core.tool_loader import build_tool_registry_from_local,build_tools_schema_from_local_tools
from gym.entities import Observation
from gym.tool import EnvironmentTool, ToolCall, _parameters_to_arguments_schema, GenericFunctionTool
from gym.toolbox import Toolbox
from gym.core.environment_fs import EnvironmentFileSystem, get_environment_fs
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class StepOutput:
    """
    一个简单的 step 返回数据结构，方便后续和 RL / function calling 逻辑对接。

    - observation: 工具返回的 Observation（或字符串）
    - done: 是否结束当前 episode
    - info: 额外调试信息
    """

    observation: Observation | str
    done: bool = False
    info: Dict = None


class TooledEnv:
    """
    一个极简的“带工具环境”基类。

    设计目标：
    - 提供 add_tool / get_tool 能力，把所有基于 EnvironmentTool 的工具挂到环境上
    - 提供 reset / step 接口，便于和 LLM function calling、简单 RL loop 对接
    - 不强依赖 debug_gym 的 RepoEnv，只做最小可行实现
    """

    def __init__(self, case_id: Optional[str] = None, domain: Optional[str] = None) -> None:
        # key: tool.name ；value: EnvironmentTool 实例
        self._tools: Dict[str, EnvironmentTool] = {}
        self._step_count: int = 0
        
        # 环境文件系统：统一管理中间结果
        # 如果提供了 case_id 和 domain，文件系统会自动按这些信息组织目录
        self._case_id = case_id
        self._domain = domain
        self._file_system: Optional[EnvironmentFileSystem] = None
        self._init_file_system()
    
    def _init_file_system(self):
        """初始化环境文件系统"""
        self._file_system = get_environment_fs()
    
    @property
    def file_system(self) -> EnvironmentFileSystem:
        """获取环境文件系统实例"""
        if self._file_system is None:
            self._init_file_system()
        return self._file_system
    
    def get_mid_result_dir(self) -> str:
        """
        获取当前环境的中间结果目录路径（用于工具中）
        
        Returns:
            str: 领域目录路径（相对于项目根目录）
        """
        if self._domain:
            domain_dir = self.file_system.get_domain_dir(self._domain, self._case_id)
            # 返回相对路径
            from pathlib import Path
            project_root = Path(__file__).resolve().parent.parent.parent
            return str(domain_dir.relative_to(project_root))
        return "gym/mid_result"

    # ===== 工具相关接口 =====
    def add_tool(self, tool: EnvironmentTool) -> None:
        """
        将一个 EnvironmentTool 实例挂载到环境中。
        优先使用 tool.name 作为 key（与 function calling 中的 function.name 对齐）。
        """
        if not getattr(tool, "name", None):
            raise ValueError("EnvironmentTool 实例缺少 name 属性，无法注册到环境中。")

        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> EnvironmentTool:
        """
        根据工具名获取工具。
        - 支持传入形如 `tool_name:extra_suffix` 的名字（只取冒号前部分），
          方便兼容 OpenAI / Anthropic 的 tool_call_id 风格。
        """
        base_name = name.split(":", 1)[0]
        if base_name not in self._tools:
            raise KeyError(f"环境中未注册名为 '{base_name}' 的工具。当前可用工具: {list(self._tools.keys())}")
        return self._tools[base_name]

    @property
    def tools(self) -> Dict[str, EnvironmentTool]:
        """只读视图，便于调试/展示。"""
        return dict(self._tools)

    # ===== 与 RL / 对话 loop 对接的基础接口 =====
    def reset(self):
        """
        重置环境。
        这里不强制返回类型，由子类决定返回什么 observation 结构。
        """
        self._step_count = 0
        raise NotImplementedError

    def step(self, action: ToolCall) -> StepOutput:
        """
        执行一步交互。
        基类只定义接口，不实现具体逻辑。
        """
        raise NotImplementedError


class MinimalSciEnv(TooledEnv):
    """
    一个最小可用的科学工具环境雏形示例。

    使用方式示例：

    ```python
    from gym.env import MinimalSciEnv

    # 假设 toolkits 中已有名为 "chem_visualizer" 的工具
    env = MinimalSciEnv(
        tool_names=["chem_visualizer"],
        case_id="case_001",
        domain="chemistry"
    )
    obs0 = env.reset()

    action = ToolCall(
        id="call-1",
        name="chem_visualizer",
        arguments={"molecules": ["c1ccc2cc3ccccc3cc2c1"]},
    )
    step_out = env.step(action)
    print(step_out.observation)
    
    # 使用环境文件系统保存中间结果
    env.file_system.save_result(
        domain="chemistry",
        filename="analysis_result",
        data={"result": "..."},
        case_id="case_001"
    )
    ```
    """

    def __init__(
        self, 
        tool_names: Optional[list[str]] = None,
        case_id: Optional[str] = None,
        domain: Optional[str] = None
    ) -> None:
        super().__init__(case_id=case_id, domain=domain)

        # 可选：通过 Toolbox 自动从已注册的工具中加载
        if tool_names:
            for name in tool_names:
                tool = Toolbox.get_tool(name)
                self.add_tool(tool)

    def reset(self) -> Dict:
        """
        返回一个非常简单的起始 observation。
        真实场景可以在这里放入题目描述 / 上下文等。
        """
        self._step_count = 0
        return {
            "observation": "MinimalSciEnv 已重置，请通过 ToolCall 调用环境中的科学工具。",
            "available_tools": list(self.tools.keys()),
        }

    def step(self, action: ToolCall) -> StepOutput:
        """
        核心逻辑：
        - 根据 ToolCall.name 找到对应的 EnvironmentTool
        - 调用工具：tool(self, **action.arguments)
          - 这里会走 EnvironmentTool.__call__，自动记录 history 并返回 Observation
        - 将 Observation 包装成 StepOutput，方便上层使用
        """
        self._step_count += 1

        tool = self.get_tool(action.name)
        # EnvironmentTool.__call__ 定义为 tool(environment, *args, **kwargs)
        # 各具体工具的 use(self, environment, action) 约定第二个参数是一个字典 action
        # 因此这里将 LLM 解析后的参数字典整体作为 action 传入
        obs: Observation = tool(self, action.arguments)

        return StepOutput(
            observation=obs,
            done=False,
            info={
                "step": self._step_count,
                "tool_name": tool.name,
                "tool_call_id": action.id,
            },
        )


def _build_env_and_tools_from_loaded(
    tool_protocols: List[Dict[str, Any]],
    function_map: Dict[str, Any],
    case_id: Optional[str] = None,
    domain: Optional[str] = None,
):
    """
    根据已加载好的 usage_tool_protocol + function_map，为单个案例构建独立环境：
    - 每个函数包装成一个 GenericFunctionTool 实例；
    - 在 MinimalSciEnv 上注册所有工具；
    - 使用 gym.tool_loader 的工具，构建 OpenAI tools schema 与本地 registry。
    
    Args:
        tool_protocols: 工具协议列表
        function_map: 函数映射字典
        case_id: 可选的题目ID，用于环境文件系统目录组织
        domain: 可选的领域名称，用于环境文件系统目录组织
    """
    env = MinimalSciEnv(tool_names=None, case_id=case_id, domain=domain)
    tool_instances: List[EnvironmentTool] = []

    for tool_entry in tool_protocols:
        if not isinstance(tool_entry, dict):
            continue
        fn_block = tool_entry.get("function") or {}
        name = fn_block.get("name")
        if not name:
            continue

        func = function_map.get(name)
        if not callable(func):
            continue

        description = fn_block.get("description", "")
        params_schema = fn_block.get("parameters") or {}
        arguments = _parameters_to_arguments_schema(params_schema)

        tool_inst = GenericFunctionTool(
            name=name,
            description=description,
            arguments=arguments,
            func=func,
        )
        env.add_tool(tool_inst)
        tool_instances.append(tool_inst)

    if not tool_instances:
        return None, None, None

    tools_schema = build_tools_schema_from_local_tools(tool_instances)
    tool_registry = build_tool_registry_from_local(tool_instances)
    env.reset()

    return env, tools_schema, tool_registry

