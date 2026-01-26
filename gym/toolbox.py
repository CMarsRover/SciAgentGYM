from typing import Any, Callable, Dict, Optional, Type


class Toolbox:
    _tool_registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str = None, config_cls: Optional[Any] = None) -> Callable:
        def decorator(subclass: Type) -> Type:
            name_ = name or subclass.__name__.lower().replace("tool", "")
            if name_ in cls._tool_registry:
                # 已存在同名工具：
                # - 如果是同一个类重复注册，直接返回，避免重复写入；
                # - 如果是不同类使用同名注册，则采用“只保留最后一个”的策略，
                #   用新的类覆盖旧的注册，以防止因多次导入/定义导致报错。
                existing_cls, existing_config_cls = cls._tool_registry[name_]
                if subclass is existing_cls:
                    return subclass

                # 覆盖为最新的实现（只保留最后一个）
                cls._tool_registry[name_] = (subclass, config_cls or existing_config_cls)
                subclass.registered_name = name_
                return subclass

            cls._tool_registry[name_] = (subclass, config_cls)
            subclass.registered_name = name_
            return subclass

        return decorator

    @classmethod
    def get_tool(cls, name: str, **kwargs) -> Any:
        base_name = name.split(":")[0]
        if base_name not in cls._tool_registry:
            raise ValueError(f"Unknown tool {base_name}")

        tool_cls, _ = cls._tool_registry[base_name]

        return tool_cls(**kwargs)