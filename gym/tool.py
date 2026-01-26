from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass
from functools import wraps
import json
from gym.entities import Event, Observation

# 兼容 debug_gym 中的异常类型；在当前最小环境中如果不存在，则定义为普通异常
try:  # pragma: no cover - optional dependency
    from debug_gym.workspace.errors import WorkspaceError, UnrecoverableTerminalError  # type: ignore
except Exception:  # pragma: no cover - fallback
    class WorkspaceError(Exception):
        """占位 WorkspaceError，使 EnvironmentTool 的异常处理在无 debug_gym 时也能工作。"""

        pass

    class UnrecoverableTerminalError(Exception):
        """占位 UnrecoverableTerminalError，用于兼容原有接口。"""

        pass

## 核心数据结构：记录每次调用的参数与返回结果
@dataclass
class Record:
    args: tuple
    kwargs: dict
    observation: Observation  

## 一次工具调用的请求
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]  

def track_history(func):
    @wraps(func)
    def wrapper(self, environment, *args, **kwargs):
        """Decorator to track the history of tool usage.
        History does not include the environment instance (first argument).
        """
        # 确保 history 已初始化（处理类属性为 None 的情况）
        if not hasattr(self, "history") or self.history is None:
            self.history = []
        observation = func(self, environment, *args, **kwargs)
        record = Record(args=args, kwargs=kwargs, observation=observation)
        self.history.append(record)
        return observation

    return wrapper


## 所有工具的抽象基类，需要被继承，子类实现use()
class EnvironmentTool(ABC):
    name: str = None
    arguments: Dict[str, Any] = None
    description: str = None
    history: list[Record] = None
    # Shell commands to run during terminal setup when this tool is used.
    # These commands will be executed before the environment is ready.
    # Example: ["apt-get update && apt-get install -y tree"]
    setup_commands: tuple[str, ...] = ()

    def __init__(self):
        self.history = []

    @track_history
    def __call__(self, *args, **kwargs) -> Observation:
        """Forwards `tool()` to the tool.use() method and
        tracks the history of tool usage."""
        try:
            return self.use(*args, **kwargs)
        except WorkspaceError as e:
            return Observation(self.name, str(e))
        except UnrecoverableTerminalError:
            # Ensure fatal terminal failures propagate so the environment can terminate the episode.
            raise
        except Exception as e:
            # Handle exceptions and return an observation
            return Observation(
                self.name, str(e)
            )  # to handle cases where the LLM hallucinates and provide invalid arguments

    def register(self, environment):
        from debug_gym.gym.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        # Auto-subscribe to events that have handlers
        for event in Event:
            if hasattr(self, event.handler_name):
                environment.event_hooks.subscribe(event, self)

        # Run setup commands if this tool has any and the environment is already
        # initialized (tool added after reset). Otherwise, they'll run via
        # on_env_reset when reset() is called (all tools are subscribed to ENV_RESET
        # because EnvironmentTool defines on_env_reset).
        if self.setup_commands:
            if (
                hasattr(environment, "workspace")
                and environment.workspace is not None
                and environment.workspace.working_dir is not None
            ):
                # Environment already reset, run setup commands now
                for cmd in self.setup_commands:
                    environment.terminal.run(cmd, raises=False)

    def unregister(self, environment):
        from debug_gym.gym.envs.env import RepoEnv

        if not isinstance(environment, RepoEnv):
            raise ValueError("The environment must be a RepoEnv instance.")

        # Unsubscribe from all events
        for event in Event:
            if hasattr(self, event.handler_name):
                environment.event_hooks.unsubscribe(event, self)

    @abstractmethod
    def use(self, environment, action) -> Observation:
        """This method is invoked directly by `tool()` or by event handlers,
        and should be overridden by subclasses. Returns an observation which
        includes the tool's name and the result of the action.
        Don't call this method directly, use `tool()` instead to track history.
        """
        pass

    def queue_event(self, environment, event: Event, **kwargs) -> None:
        environment.queue_event(event, source=self, **kwargs)

    def on_env_reset(self, environment, **kwargs) -> Observation:
        """Reset the tool state on environment reset.
        Please call `super().on_env_reset()` if subclass overrides this method.
        """
        self.history = []

        # Run setup commands if this tool has any
        for cmd in self.setup_commands:
            environment.terminal.run(cmd, raises=False)

        return None

    def __str__(self):
        args = ", ".join(f"{k}:{v['type'][0]}" for k, v in self.arguments.items())
        return f"{self.name}({args}): {self.description.split('.')[0].strip()}."


def convert_to_json_serializable(obj):
    """
    递归地将对象转换为 JSON 可序列化的格式。
    处理 numpy 数组、scipy 稀疏矩阵等常见科学计算类型。
    """
    import numpy as np
    try:
        import scipy.sparse as sp
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
    
    # 首先检查是否是 numpy 标量类型（必须在检查数组之前）
    # 使用更全面的 numpy 类型检查
    if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
        # numpy 标量类型（int8, int16, int32, int64, float16, float32, float64, etc.）
        try:
            if isinstance(obj, (np.integer, np.floating, np.complexfloating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
        except (AttributeError, TypeError):
            pass
    
    # 处理 numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # 处理 scipy 稀疏矩阵
    if HAS_SCIPY and isinstance(obj, sp.spmatrix):
        # 转换为密集矩阵然后转换为列表
        # 对于大矩阵，只返回形状和稀疏度信息
        try:
            if obj.shape[0] * obj.shape[1] <= 10000:  # 小矩阵，转换为密集格式
                return obj.toarray().tolist()
            else:  # 大矩阵，只返回元数据
                return {
                    "_type": "sparse_matrix",
                    "shape": list(obj.shape),
                    "format": obj.format,
                    "nnz": obj.nnz,
                    "dtype": str(obj.dtype),
                    "note": "矩阵太大，未序列化完整数据"
                }
        except Exception:
            return {
                "_type": "sparse_matrix",
                "shape": list(obj.shape),
                "format": obj.format,
                "error": "无法转换为密集格式"
            }
    
    # 处理字典
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    
    # 处理列表和元组
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    
    # 尝试直接使用 numpy 的 item() 方法（适用于标量）
    if hasattr(obj, 'item') and hasattr(obj, 'dtype'):
        try:
            return obj.item()
        except (AttributeError, ValueError, TypeError):
            pass
    
    # 其他类型直接返回（Python 原生类型应该已经是可序列化的）
    return obj


class GenericFunctionTool(EnvironmentTool):
  
    """
    将普通 Python 函数包装为 EnvironmentTool，便于通过 MinimalSciEnv 统一调度。
    """

    def __init__(self, name: str, description: str, arguments: Dict[str, Any], func):
        super().__init__()  # 调用父类 __init__ 初始化 history
        self.name = name
        self.description = description or ""
        # arguments 结构与 func_calling_cases_em_161.py 中工具类保持一致：
        # {param_name: {"type": "...", "description": "...", ...}}
        self.arguments = arguments or {}
        self._func = func

    def use(self, environment, action) -> Observation:
        """
        action 一般是 dict 或 {"arguments": {...}}，内部直接调用原始函数。
        """
        import traceback as tb

        try:
            args = action.get("arguments", action) if isinstance(action, dict) else {}
            result = self._func(**args)

            if isinstance(result, (dict, list)):
                payload = result
            elif isinstance(result, tuple):
                payload = list(result)
            else:
                payload = {"result": result}

            # 转换为 JSON 可序列化格式
            payload = convert_to_json_serializable(payload)

            return Observation(self.name, json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception as e:  # pragma: no cover
            return Observation(
                self.name,
                f"错误: {str(e)}\n{tb.format_exc()}",
            )


def _parameters_to_arguments_schema(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 usage_tool_protocol 里的 JSON Schema parameters 转成 EnvironmentTool.arguments 格式。
    输入示例:
        {
          "type": "object",
          "properties": {
            "theta": {"type": "number", "description": "..."},
            ...
          },
          "required": ["theta", ...]
        }
    输出示例:
        {
          "theta": {"type": "number", "description": "..."},
          ...
        }
    """
    if not isinstance(params, dict):
        return {}

    props = params.get("properties") or {}
    if not isinstance(props, dict):
        return {}

    arguments: Dict[str, Any] = {}
    for name, info in props.items():
        if not isinstance(info, dict):
            continue
        arg_schema: Dict[str, Any] = {
            "type": info.get("type", "number"),
            "description": info.get("description", ""),
        }
        if "enum" in info:
            arg_schema["enum"] = info["enum"]
        if info.get("type") == "array":
            arg_schema["items"] = info.get("items") or {"type": "number"}
        arguments[name] = arg_schema

    return arguments