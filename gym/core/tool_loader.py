#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具加载和注册模块

核心流程：
1. 从测试案例的 usage_tool_protocol 中提取工具定义
2. 验证工具协议格式
3. 从 function_path 动态导入 Python 函数
4. 将函数包装成 GenericFunctionTool
5. 注册到 MinimalSciEnv 环境
6. 构建 OpenAI tools schema 供 LLM 调用
"""   
import json
import traceback
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Union
from copy import deepcopy
from gym.tool import EnvironmentTool, GenericFunctionTool, _parameters_to_arguments_schema
from gym.entities import Observation
from gym.toolbox import Toolbox
from gym.tool import ToolCall
from gym.core.data_loader import deduplicate_usage_tool_entries 

_TOOLKITS_INDEX: Optional[Dict[str, Dict[str, Path]]] = None
# 文件位于 gym/core/tool_loader.py，向上两级到项目根目录
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root)) 
# ========== 辅助函数 ==========

def build_case_context(test_case: Dict[str, Any]) -> Dict[str, Any]:
        """提取案例上下文信息，用于日志记录。"""
        metadata = test_case.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        dataset_source = (
            metadata.get("dataset_key")
            or metadata.get("dataset_name")
            or metadata.get("dataset")
            or test_case.get("filename")
            or metadata.get("source")
            or metadata.get("trace_root")
        )

        case_id = (
            test_case.get("id")
            or metadata.get("original_question_id")
            or metadata.get("case_id")
            or metadata.get("question_id")
        )

        return {
            "case_id": case_id,
            "dataset": dataset_source,
            "filename": test_case.get("filename"),
            "subject": metadata.get("subject"),
            "topic": metadata.get("topic"),
            "test_type": metadata.get("test_type"),
            }

def validate_function_tool(tool_entry: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    校验单个函数式工具协议的结构。
    
    返回:
        (is_valid, error_reason)
    """
    if not isinstance(tool_entry, dict):
        return False, "工具条目不是对象类型"

    function_block = tool_entry.get("function")
    if not isinstance(function_block, dict):
        return False, "function 字段必须是对象"

    if not function_block.get("name"):
        return False, "缺少函数名称"

    parameters = function_block.get("parameters")
    if parameters is None:
        return True, None
    if not isinstance(parameters, dict):
        return False, "parameters 必须是对象"

    declared_type = parameters.get("type")
    if declared_type not in (None, "object"):
        return False, "parameters.type 只能是 object"

    return True, None

def log_invalid_tool(
    case_context: Dict[str, Any],
    tool_entry: Dict[str, Any],
    reason: str,
    parent_class: Optional[str] = None,
) -> None:
    """将无效的工具定义记录到控制台（简化版，不写文件）"""
    tool_name = (tool_entry.get("function") or {}).get("name") or tool_entry.get("class_name")
    print(f"⚠️ 无效工具: {tool_name} (案例 {case_context.get('case_id')}): {reason}")

def find_tool_file(function_path: str, subject: Optional[str] = None, topic: Optional[str] = None) -> Optional[Path]:
    """
    根据 function_path 查找实际的工具文件。
    
    优先使用 toolkits/{subject}/{topic}/filename 路径结构。
    
    支持路径格式：
    - toolkits/{subject}/{topic}/filename.py (优先)
    - ./tools/filename.py
    - tools/filename.py
    - toolkits/{discipline}/filename.py (回退)
    
    参数:
        function_path: 工具文件路径（如 "./tools/circular_motion_solver_107.py"）
        subject: 学科名称（如 "Physics"），从 metadata 中获取
        topic: 主题名称（如 "Mechanics"），从 metadata 中获取
    """
    # 文件位于 gym/core/，向上两级到项目根目录
    project_root = Path(__file__).resolve().parents[2]
    
    clean_path = str(function_path).strip()
    if clean_path.startswith('./'):
        clean_path = clean_path[2:]
    
    # 提取文件名（去除路径前缀）
    if 'tools/' in clean_path:
        filename = clean_path.split('tools/')[-1]
    elif '/' in clean_path:
        filename = clean_path.split('/')[-1]
    else:
        filename = clean_path
    
    possible_paths = []
    
    # 优先级1: 使用 subject 和 topic 构建路径 toolkits/{subject}/{topic}/filename
    if subject and topic:
        # 将 subject 和 topic 转换为小写，并处理空格/下划线/连字符的格式差异
        # subject 也需要处理空格（如 "Materials Science" -> "materials_science"）
        subject_lower = subject.lower().strip().replace(' ', '_')
        
        # Subject 映射：处理常见的命名差异
        # 例如 "Materials" -> "materials_science"
        subject_mappings = {
            'materials': 'materials_science',
            'material': 'materials_science',
        }
        subject_variants = {subject_lower}
        if subject_lower in subject_mappings:
            subject_variants.add(subject_mappings[subject_lower])
        
        topic_lower = topic.lower().strip()
        
        # 尝试多种可能的 topic 格式变体（处理空格、下划线和连字符的差异）
        topic_variants = set()
        topic_variants.add(topic_lower)  # 原始格式
        
        # 处理空格、下划线和连字符的各种组合
        # 空格转下划线
        topic_variants.add(topic_lower.replace(' ', '_'))
        # 空格转连字符
        topic_variants.add(topic_lower.replace(' ', '-'))
        # 下划线转空格
        topic_variants.add(topic_lower.replace('_', ' '))
        # 下划线转连字符
        topic_variants.add(topic_lower.replace('_', '-'))
        # 连字符转下划线
        topic_variants.add(topic_lower.replace('-', '_'))
        # 连字符转空格
        topic_variants.add(topic_lower.replace('-', ' '))
        
        # 处理 "X Ray" -> "x-ray" 这种情况（X-ray Diffraction Analysis）
        # 匹配 "x ray", "x_ray", "x-ray" 等格式
        import re
        # 将 "x ray" 或 "x_ray" 转换为 "x-ray"
        if re.search(r'\bx[\s_-]ray\b', topic_lower, re.IGNORECASE):
            # 替换各种格式的 "x ray" 为 "x-ray"
            xray_variant = re.sub(r'\bx[\s_]ray\b', 'x-ray', topic_lower, flags=re.IGNORECASE)
            topic_variants.add(xray_variant)
            # 对转换后的结果，再处理空格和下划线
            topic_variants.add(xray_variant.replace(' ', '_'))
            topic_variants.add(xray_variant.replace(' ', '-'))
            topic_variants.add(xray_variant.replace('_', '-'))
            topic_variants.add(xray_variant.replace('-', '_'))
        
        # 构建路径：toolkits/{subject}/{topic}/filename
        for subject_var in subject_variants:
            for topic_var in topic_variants:
                possible_paths.append(project_root / "toolkits" / subject_var / topic_var / filename)
    
    # 优先级2: 直接路径（如果已经是完整路径）
    possible_paths.append(project_root / clean_path)
    
    # 优先级3: toolkits 目录下的直接查找
    possible_paths.append(project_root / "toolkits" / filename)
    
    # 优先级4: src/tools 目录（新增：支持 src/tools/ 路径）
    possible_paths.append(project_root / "src" / "tools" / filename)
    
    # 优先级5: tools 目录
    possible_paths.append(project_root / "tools" / filename)
    
    # 优先级6: 如果路径包含 tools/，尝试在各个学科目录下查找
    if 'tools/' in clean_path:
        for discipline in ['physics', 'chemistry', 'materials_science', 'astronomy', 'statistics', 'life_science']:
            possible_paths.append(project_root / "toolkits" / discipline / filename)
    
    # 尝试所有可能的路径
    found_path = None
    for path in possible_paths:
        if path.exists() and path.is_file():
            found_path = path
            break

    # 优先级7（兜底）：如果通过 subject/topic 拼路径都没命中，做一次 toolkits/**/{filename}
    # 全局递归搜索。这处理了 case metadata 里 subject/topic 与工具实际所在
    # 目录不一致的情形（例如 issue #3 multi #62/68/73/79/81）。工具文件名带数字
    # 后缀（如 mechanics_of_materials_toolkit_claude_10.py），在整个 toolkits 里
    # 是唯一的，所以全局搜索不会产生歧义。
    if found_path is None:
        toolkits_root = project_root / "toolkits"
        if toolkits_root.exists():
            matches = list(toolkits_root.rglob(filename))
            if matches:
                found_path = matches[0]
                if len(matches) > 1:
                    print(
                        f"⚠️ 兜底搜索找到多个 {filename}，使用第一个: {found_path}"
                    )
                else:
                    print(
                        f"ℹ️ 兜底搜索命中: {filename} → {found_path.relative_to(project_root)}"
                        f" (metadata subject/topic: {subject!r}/{topic!r})"
                    )

    # 如果仍没找到，打印调试信息
    if found_path is None:
        print(f"⚠️ 找不到工具文件: {function_path}")
        print(f"   尝试的路径（前10个）:")
        for i, path in enumerate(possible_paths[:10], 1):
            print(f"     {i}. {path} {'✓' if path.exists() else '✗'}")
        if len(possible_paths) > 10:
            print(f"     ... 还有 {len(possible_paths) - 10} 个路径未显示")
        print(f"   Subject: {subject}, Topic: {topic}")
    
    return found_path

def dynamic_import_tool_functions(tool_path: str, subject: Optional[str] = None, topic: Optional[str] = None) -> dict:
    """
    动态导入指定路径的工具函数。
    
    参数:
        tool_path: 工具文件路径（如 "./tools/xps_spectroscopy_toolkit_0001.py"）
        subject: 学科名称（如 "Physics"），从 metadata 中获取
        topic: 主题名称（如 "Mechanics"），从 metadata 中获取
    
    返回:
        dict: {function_name: callable} 映射
    """
    try:
        # 查找实际文件（使用 subject 和 topic 信息）
        file_path = find_tool_file(tool_path, subject=subject, topic=topic)
        if not file_path:
            print(f"❌ 找不到工具文件: {tool_path}")
            return {}
        
        # 使用 importlib.util 从文件路径导入
        module_name = file_path.stem
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            print(f"❌ 无法创建模块规范: {tool_path}")
            return {}
        
        # 确保项目根目录在 sys.path 中（文件位于 gym/core/）
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # 关键补丁（修复 Issue #3 类型附带发现）：把工具文件所在目录加入 sys.path
        # 让形如 `from thin_film_interference import ...` 这类"同目录绝对 import"能解析。
        parent_dir = str(file_path.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 提取模块中的所有可调用函数
        functions = {}
        for name in dir(module):
            if name.startswith('_'):
                continue
            
            obj = getattr(module, name)
            if not callable(obj):
                continue
            
            # 确保是在这个模块中定义的
            if hasattr(obj, '__module__') and obj.__module__ == module.__name__:
                if inspect.isfunction(obj):
                    functions[name] = obj
                elif inspect.isclass(obj):
                    # 对于类，尝试两种方式：
                    # 1. 如果可以无参实例化，直接实例化并提取方法
                    # 2. 如果有必填参数，尝试提取静态方法/类方法，或提示需要函数包装
                    try:
                        sig = inspect.signature(obj)
                        required_params = [
                            p for p in sig.parameters.values()
                            if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                          inspect.Parameter.POSITIONAL_OR_KEYWORD)
                            and p.default is inspect._empty
                            and p.name != "self"
                        ]
                        
                        if required_params:
                            # 类有必填参数，尝试提取静态方法/类方法
                            static_methods_found = False
                            for attr_name in dir(obj):
                                if attr_name.startswith('_'):
                                    continue
                                attr = getattr(obj, attr_name)
                                if inspect.isfunction(attr) or inspect.ismethod(attr):
                                    # 检查是否是静态方法或类方法
                                    if isinstance(attr, staticmethod):
                                        functions[f"{name}.{attr_name}"] = attr.__func__
                                        static_methods_found = True
                                    elif isinstance(attr, classmethod):
                                        # 类方法也可以包装成函数
                                        def make_classmethod_wrapper(cm):
                                            def wrapper(*args, **kwargs):
                                                return cm.__func__(obj, *args, **kwargs)
                                            return wrapper
                                        functions[f"{name}.{attr_name}"] = make_classmethod_wrapper(attr)
                                        static_methods_found = True
                            
                            if not static_methods_found:
                                # 没有找到静态方法，提示开发者需要函数包装
                                print(
                                    f"⚠️ 类 {name} 需要必填参数 {[p.name for p in required_params]}，"
                                    f"无法自动实例化。建议：\n"
                                    f"   1. 在工具文件中提供函数包装（推荐）\n"
                                    f"   2. 或将该类改为无参构造 + 内部配置加载\n"
                                    f"   3. 或提供静态方法/类方法供工具调用"
                                )
                            else:
                                print(
                                    f"✅ 类 {name} 有必填参数，已提取静态方法/类方法作为工具"
                                )
                            continue

                        # 无参构造，直接实例化
                        instance = obj()
                        for method_name in dir(instance):
                            if not method_name.startswith('_') and callable(getattr(instance, method_name)):
                                method = getattr(instance, method_name)
                                if inspect.ismethod(method):
                                    combined_name = f"{name}.{method_name}"
                                    functions[combined_name] = method
                                    if method_name not in functions:
                                        functions[method_name] = method
                    except Exception as e:
                        print(f"⚠️ 无法为类 {name} 创建实例: {e}")
        
        print(f"✅ 成功导入 {len(functions)} 个函数: {list(functions.keys())}")
        return functions
        
    except Exception as e:
        print(f"❌ 导入工具函数失败 {tool_path}: {e}")
        traceback.print_exc()
        return {}

# ========== 核心函数 ==========

def load_tools_for_case(test_case: dict) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    根据测试案例加载对应的工具协议和函数。
    
    流程：
    1. 从 test_case 中提取 usage_tool_protocol
    2. 验证每个工具协议格式
    3. 从 function_path 动态导入函数
    4. 返回工具协议列表和函数映射
    
    参数:
        test_case: 测试案例字典，包含 usage_tool_protocol 字段
    
    返回:
        (tools, function_map):
        - tools: 验证后的工具协议列表
        - function_map: {function_name: callable} 映射
    """
    raw_tools = test_case.get('usage_tool_protocol', []) or []

    tools = []
    function_map = {}
    loaded_paths = set()
    metadata = test_case.get('metadata') or {}
    scope_marker = metadata.get('topic_protocol_scope') or metadata.get('subject_protocol_scope')
    apply_dedup = bool(scope_marker) or metadata.get('with_all_tools') is True
    case_context = build_case_context(test_case)

    def _register_tool(tool_entry: Dict[str, Any], parent_class: str = None):
        """内部函数：注册单个工具"""
        is_valid, error_msg = validate_function_tool(tool_entry)
        if not is_valid:
            log_invalid_tool(
                case_context,
                tool_entry,
                error_msg or "未知的工具协议错误",
                parent_class,
            )
            return

        # 确保我们有一个可写的 additionalProperties 副本
        addl = dict(tool_entry.get("additionalProperties") or {})
        tool_path = addl.get("function_path")

        if tool_path and tool_path not in loaded_paths:
            # 从 metadata 中提取 subject 和 topic，用于构建正确的路径
            subject = metadata.get("subject")
            topic = metadata.get("topic")

            # 先解析出实际文件路径，并把信息写回 tool_entry，后续 register_tools_to_env
            # 可以据此推断对应的 *_tools_gym.py 模块
            resolved_file = find_tool_file(tool_path, subject=subject, topic=topic)
            if resolved_file:
                try:
                    root = Path(__file__).resolve().parents[2]
                    rel_path = resolved_file.relative_to(root)
                    addl["resolved_file_path"] = str(rel_path)
                    addl["resolved_dir"] = str(rel_path.parent)
                    tool_entry["additionalProperties"] = addl
                except Exception:
                    # 相对路径计算失败并不影响后续函数导入，仅影响是否能自动找到 *_tools_gym
                    pass

            # 再按原有逻辑导入底层函数，作为通用回退实现
            functions = dynamic_import_tool_functions(
                tool_path,
                subject=subject,
                topic=topic,
            )
            function_map.update(functions)
            loaded_paths.add(tool_path)

        tools.append(tool_entry)

    # 处理所有工具
    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue

        # 移除跳过 essential_circuit_analysis_guide 的逻辑，这个函数应该被正常加载
        # if isinstance(function_block, dict) and function_block.get('name') == 'essential_circuit_analysis_guide':
        #     continue

        if 'additionalProperties' in tool and 'function_path' in tool['additionalProperties']:
            _register_tool(tool)

        elif 'class_name' in tool and 'tools' in tool:
            print(f"处理类格式工具: {tool['class_name']}")
            nested_tools = tool.get('tools', [])
            if not isinstance(nested_tools, list):
                log_invalid_tool(case_context, tool, "类工具的 tools 字段不是列表", tool.get('class_name'))
                continue
            for nested_tool in nested_tools:
                if not isinstance(nested_tool, dict):
                    continue
                _register_tool(nested_tool, parent_class=tool.get('class_name'))
        else:
            log_invalid_tool(case_context, tool, "未识别的工具协议结构")

    # 去重处理
    if apply_dedup:
        tools = deduplicate_usage_tool_entries(tools)
    
    return tools, function_map

def infer_toolkits_path_from_metadata(
    metadata: Dict[str, Any],
    toolkits_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    根据 metadata 中的 subject 和 topic 自动推断 toolkits 目录路径。
    
    映射规则：
    - subject: "Physics" → "physics"
    - topic: "Optics" → "optics"
    - 路径: toolkits/{subject}/{topic}/
    
    Args:
        metadata: 包含 subject 和 topic 的 metadata 字典
        toolkits_root: toolkits 根目录路径，默认为项目根目录下的 toolkits
    
    Returns:
        推断出的工具目录路径，如果无法推断则返回 None
    
    Example:
        >>> metadata = {"subject": "Physics", "topic": "Optics"}
        >>> path = infer_toolkits_path_from_metadata(metadata)
        >>> print(path)  # toolkits/physics/optics
    """
    if not isinstance(metadata, dict):
        return None
    
    subject = metadata.get("subject", "")
    topic = metadata.get("topic", "")
    
    if not subject or not topic:
        return None
    
    # 转换为目录名格式：小写 + 空格替换为下划线
    def normalize_name(name: str) -> str:
        return name.lower().strip().replace(" ", "_").replace("-", "_")
    
    subject_dir = normalize_name(subject)
    topic_dir = normalize_name(topic)
    
    # 构建路径（文件位于 gym/core/，向上两级到项目根目录）
    root = Path(__file__).resolve().parents[2]
    if toolkits_root:
        if isinstance(toolkits_root, str):
            toolkits_base = Path(toolkits_root)
        else:
            toolkits_base = toolkits_root
        if not toolkits_base.is_absolute():
            toolkits_base = root / toolkits_base
    else:
        toolkits_base = root / "toolkits"
    
    # 尝试精确匹配
    candidate_path = toolkits_base / subject_dir / topic_dir
    if candidate_path.exists() and candidate_path.is_dir():
        return candidate_path
    
    # 尝试模糊匹配（处理命名差异）
    # 例如 "Condensed Matter Physics" -> "condensed_matter_physics"
    if (toolkits_base / subject_dir).exists():
        subject_path = toolkits_base / subject_dir
        # 查找最匹配的子目录
        for subdir in subject_path.iterdir():
            if subdir.is_dir():
                # 检查是否包含 topic 关键词
                subdir_normalized = normalize_name(subdir.name)
                if topic_dir in subdir_normalized or subdir_normalized in topic_dir:
                    return subdir
                # 检查是否有重叠词
                topic_words = set(topic_dir.split("_"))
                subdir_words = set(subdir_normalized.split("_"))
                if topic_words & subdir_words:  # 有交集
                    return subdir
    
    # 直接在 toolkits 下搜索 topic 目录
    for subject_candidate in toolkits_base.iterdir():
        if not subject_candidate.is_dir():
            continue
        for topic_candidate in subject_candidate.iterdir():
            if topic_candidate.is_dir():
                topic_normalized = normalize_name(topic_candidate.name)
                if topic_dir == topic_normalized or topic_dir in topic_normalized:
                    return topic_candidate
    
    return None

def build_toolkits_path_index(
    toolkits_root: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Path]]:
    """
    构建 toolkits 目录的层级索引，用于快速查找。
    
    Returns:
        嵌套字典 {subject: {topic: path}}
    
    Example:
        >>> index = build_toolkits_path_index()
        >>> print(index["physics"]["optics"])  # Path to optics directory
    """
    # 文件位于 gym/core/，向上两级到项目根目录
    root = Path(__file__).resolve().parents[2]
    if toolkits_root:
        toolkits_base = Path(toolkits_root) if isinstance(toolkits_root, str) else toolkits_root
        if not toolkits_base.is_absolute():
            toolkits_base = root / toolkits_base
    else:
        toolkits_base = root / "toolkits"
    
    index: Dict[str, Dict[str, Path]] = {}
    
    if not toolkits_base.exists():
        return index
    
    for subject_dir in toolkits_base.iterdir():
        if not subject_dir.is_dir() or subject_dir.name.startswith((".", "_")):
            continue
        
        subject_name = subject_dir.name.lower()
        index[subject_name] = {}
        
        for topic_dir in subject_dir.iterdir():
            if not topic_dir.is_dir() or topic_dir.name.startswith((".", "_")):
                continue
            
            topic_name = topic_dir.name.lower()
            index[subject_name][topic_name] = topic_dir
    
    return index

def get_toolkits_index() -> Dict[str, Dict[str, Path]]:
    """获取 toolkits 索引（带缓存）"""
    global _TOOLKITS_INDEX
    if _TOOLKITS_INDEX is None:
        _TOOLKITS_INDEX = build_toolkits_path_index()
    return _TOOLKITS_INDEX

def load_all_tools_from_directory(
    directory_path: Union[str, Path],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    从指定目录批量加载所有工具函数。
    
    扫描目录下的所有 Python 文件，提取带有特定标记的函数作为工具。
    优先使用 *_tools_gym.py 中已注册的工具，其次扫描其他 .py 文件。
    
    Args:
        directory_path: 工具目录路径（绝对路径或相对于项目根目录的路径）
        include_patterns: 包含的文件名模式列表，如 ["*_solver*.py", "*_tools*.py"]
        exclude_patterns: 排除的文件名模式列表，如 ["__init__.py", "test_*.py"]
    
    Returns:
        (tool_protocols, function_map):
        - tool_protocols: 工具协议列表
        - function_map: {function_name: callable} 映射
    
    Example:
        >>> tool_protocols, function_map = load_all_tools_from_directory(
        ...     "toolkits/physics/optics",
        ...     exclude_patterns=["test_*.py", "__init__.py"]
        ... )
    """
    import fnmatch
    import inspect
    
    # 文件位于 gym/core/，向上两级到项目根目录
    root = Path(__file__).resolve().parents[2]
    
    # 处理路径
    if isinstance(directory_path, str):
        dir_path = Path(directory_path)
    else:
        dir_path = directory_path
    
    if not dir_path.is_absolute():
        dir_path = root / dir_path
    
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"⚠️ 目录不存在: {dir_path}")
        return [], {}
    
    # 默认排除模式
    # code_block_*.py / example_*.py / usage_*.py / demo_*.py 是文档/演示片段，
    # 不是真实工具，避免它们污染扫描结果（且这些文件常引用不存在的辅助模块）。
    default_exclude = [
        "__init__.py", "__pycache__",
        "test_*.py", "convert_*.py",
        "example_*.py", "usage_*.py", "demo_*.py", "code_block_*.py",
    ]
    exclude_patterns = list(exclude_patterns or []) + default_exclude
    
    tool_protocols: List[Dict[str, Any]] = []
    function_map: Dict[str, Any] = {}
    registered_names: set = set()
    
    # 确保路径在 sys.path 中
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # 关键补丁：把 topic 目录本身也加入 sys.path，让工具间的
    # 同目录 import（如 `from thin_film_interference import ...`）能解析。
    if str(dir_path) not in sys.path:
        sys.path.insert(0, str(dir_path))
    
    # 1. 优先加载 *_tools_gym.py（已通过 @Toolbox.register 注册的工具）
    gym_files = list(dir_path.glob("*_tools_gym.py"))
    for gym_file in gym_files:
        try:
            rel_path = gym_file.relative_to(root)
            module_name = ".".join(rel_path.with_suffix("").parts)
            importlib.import_module(module_name)
            print(f"✅ 已导入工具注册模块: {module_name}")
        except Exception as e:
            print(f"⚠️ 导入 {gym_file.name} 失败: {e}")
    
    # 从 Toolbox 获取已注册的工具
    registry = getattr(Toolbox, "_tool_registry", {})
    for tool_name, (tool_cls, _) in registry.items():
        if tool_name in registered_names:
            continue
        
        # 检查工具是否来自目标目录（通过模块名判断）
        tool_module = getattr(tool_cls, "__module__", "")
        dir_module_prefix = ".".join(dir_path.relative_to(root).parts)
        
        if dir_module_prefix in tool_module:
            # 构建工具协议
            tool_inst = tool_cls()
            protocol = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": getattr(tool_inst, "description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": getattr(tool_inst, "arguments", {}),
                        "required": list(getattr(tool_inst, "arguments", {}).keys()),
                    }
                },
                "additionalProperties": {
                    "source": "toolbox_registry",
                    "resolved_dir": str(dir_path.relative_to(root)),
                }
            }
            tool_protocols.append(protocol)
            registered_names.add(tool_name)
            print(f"✅ 从 Toolbox 加载工具: {tool_name}")
    
    # 2. 扫描其他 Python 文件，查找可导出的函数
    py_files = list(dir_path.glob("*.py"))
    for py_file in py_files:
        # 检查排除模式
        should_exclude = False
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(py_file.name, pattern):
                should_exclude = True
                break
        if should_exclude:
            continue
        
        # 检查包含模式（如果指定了的话）
        if include_patterns:
            should_include = False
            for pattern in include_patterns:
                if fnmatch.fnmatch(py_file.name, pattern):
                    should_include = True
                    break
            if not should_include:
                continue
        
        # 跳过已处理的 *_tools_gym.py
        if py_file.name.endswith("_tools_gym.py"):
            continue
        
        try:
            rel_path = py_file.relative_to(root)
            module_name = ".".join(rel_path.with_suffix("").parts)
            module = importlib.import_module(module_name)
            
            # 遍历模块中的所有函数
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # 跳过私有函数和已注册的函数
                if name.startswith("_") or name in registered_names:
                    continue
                
                # 跳过导入的函数（只保留在当前模块定义的）
                if getattr(obj, "__module__", "") != module_name:
                    continue
                
                # 获取函数签名
                sig = inspect.signature(obj)
                params_schema = {"type": "object", "properties": {}, "required": []}
                
                for param_name, param in sig.parameters.items():
                    if param_name in ("self", "cls"):
                        continue
                    
                    # 推断参数类型
                    param_type = "string"  # 默认类型
                    if param.annotation != inspect.Parameter.empty:
                        ann = param.annotation
                        if ann in (int, float):
                            param_type = "number"
                        elif ann == bool:
                            param_type = "boolean"
                        elif ann in (list, tuple):
                            param_type = "array"
                        elif ann == dict:
                            param_type = "object"
                    
                    params_schema["properties"][param_name] = {
                        "type": param_type,
                        "description": "",
                    }
                    
                    # 判断是否必需
                    if param.default == inspect.Parameter.empty:
                        params_schema["required"].append(param_name)
                
                # 构建工具协议
                protocol = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": obj.__doc__ or "",
                        "parameters": params_schema,
                    },
                    "additionalProperties": {
                        "function_path": str(rel_path),
                        "resolved_dir": str(rel_path.parent),
                        "source": "directory_scan",
                    }
                }
                tool_protocols.append(protocol)
                function_map[name] = obj
                registered_names.add(name)
                print(f"✅ 从文件扫描加载函数: {name} (from {py_file.name})")
                
        except Exception as e:
            print(f"⚠️ 处理文件 {py_file.name} 时出错: {e}")
    
    print(f"\n📦 从目录 {dir_path.name} 共加载 {len(tool_protocols)} 个工具")
    return tool_protocols, function_map

def register_tools_to_env(
    tool_protocols: List[Dict[str, Any]],
    function_map: Dict[str, Any],
    case_id: Optional[str] = None,
    domain: Optional[str] = None,
    query_data: Optional[Dict[str, Any]] = None,
    auto_load_dirs: Optional[List[Union[str, Path]]] = None,
    auto_infer_from_metadata: bool = False,
) -> Tuple[Any, List[EnvironmentTool], List[Dict[str, Any]], Dict[str, EnvironmentTool]]:
    """
    将工具协议和函数映射注册到环境中。
    
    优先级：
    1. 如果在预生成的 *_tools_gym.py 中已经有对应的 EnvironmentTool（经 Toolbox 注册），
       则优先使用这些工具类构建环境（类似 func_calling_cases_em_161.py 的局部环境）。
    2. 若找不到预注册工具，则退回到通用的 GenericFunctionTool 包装动态导入的函数。
    
    Args:
        tool_protocols: 工具协议列表
        function_map: 函数映射字典
        case_id: 可选的题目ID，用于环境文件系统目录组织
        domain: 可选的领域名称，用于环境文件系统目录组织
        query_data: 可选的查询数据，用于自动提取 case_id 和 domain
        auto_load_dirs: 可选的目录列表，自动从这些目录加载所有工具函数
                        例如: ["toolkits/physics/optics", "toolkits/chemistry"]
        auto_infer_from_metadata: 是否自动从 query_data 的 metadata 中推断工具目录
                                   根据 subject 和 topic 字段自动定位 toolkits/{subject}/{topic}/
    
    Returns:
        tuple: (env, tool_instances, tools_schema, tool_registry)
    
    Example:
        # 方式1：使用 tool_protocols 指定工具
        env, tools, schema, registry = register_tools_to_env(tool_protocols, function_map)
        
        # 方式2：自动加载目录下所有工具
        env, tools, schema, registry = register_tools_to_env(
            [], {},  # 空的 protocols 和 function_map
            auto_load_dirs=["toolkits/physics/optics"]
        )
        
        # 方式3：混合使用（指定工具 + 目录加载）
        env, tools, schema, registry = register_tools_to_env(
            tool_protocols, function_map,
            auto_load_dirs=["toolkits/physics/optics"]
        )
        
        # 方式4：自动从 metadata 推断工具目录（推荐）
        # query_data 的 metadata 包含 {"subject": "Physics", "topic": "Optics"}
        # 将自动加载 toolkits/physics/optics/ 下的所有工具
        env, tools, schema, registry = register_tools_to_env(
            [], {},
            query_data=query_data,
            auto_infer_from_metadata=True
        )
    """
    # 延迟导入以避免循环导入
    from gym.env import MinimalSciEnv
    
    # 用于去重的工具名称集合
    existing_tool_names: set = set()
    
    # 先收集已有工具协议的名称
    for protocol in tool_protocols:
        if isinstance(protocol, dict):
            fn_block = protocol.get("function") or {}
            name = fn_block.get("name")
            if name:
                existing_tool_names.add(name)
    
    # 如果启用自动推断，从 query_data 的 metadata 推断工具目录
    if auto_infer_from_metadata and query_data:
        metadata = query_data.get("metadata") or {}
        if isinstance(metadata, dict):
            inferred_path = infer_toolkits_path_from_metadata(metadata)
            if inferred_path:
                subject = metadata.get("subject", "unknown")
                topic = metadata.get("topic", "unknown")
                print(f"🔍 自动推断工具目录: {subject}/{topic} -> {inferred_path}")
                
                # 加载推断目录下的所有工具
                dir_protocols, dir_functions = load_all_tools_from_directory(inferred_path)
                
                # 去重：只添加新工具，跳过已存在的
                new_protocols = []
                for protocol in dir_protocols:
                    if isinstance(protocol, dict):
                        fn_block = protocol.get("function") or {}
                        name = fn_block.get("name")
                        if name and name not in existing_tool_names:
                            new_protocols.append(protocol)
                            existing_tool_names.add(name)
                        elif name:
                            print(f"⏭️ 跳过重复工具: {name}")
                
                tool_protocols = list(tool_protocols) + new_protocols
                # 更新 function_map，同样去重
                for name, func in dir_functions.items():
                    if name not in function_map:
                        function_map[name] = func
            else:
                subject = metadata.get("subject", "")
                topic = metadata.get("topic", "")
                if subject or topic:
                    print(f"⚠️ 无法推断工具目录: subject={subject}, topic={topic}")
    
    # 如果指定了 auto_load_dirs，从这些目录加载工具
    if auto_load_dirs:
        for dir_path in auto_load_dirs:
            dir_protocols, dir_functions = load_all_tools_from_directory(dir_path)
            
            # 去重：只添加新工具
            new_protocols = []
            for protocol in dir_protocols:
                if isinstance(protocol, dict):
                    fn_block = protocol.get("function") or {}
                    name = fn_block.get("name")
                    if name and name not in existing_tool_names:
                        new_protocols.append(protocol)
                        existing_tool_names.add(name)
                    elif name:
                        print(f"⏭️ 跳过重复工具: {name}")
            
            tool_protocols = list(tool_protocols) + new_protocols
            for name, func in dir_functions.items():
                if name not in function_map:
                    function_map[name] = func
    
    # 如果提供了 query_data，尝试从中提取 case_id 和 domain
    if query_data and (case_id is None or domain is None):
        metadata = query_data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        
        if case_id is None:
            case_id = (
                query_data.get("id")
                or metadata.get("id")
                or metadata.get("case_id")
                or metadata.get("question_id")
                or metadata.get("original_question_id")
            )
            if case_id:
                case_id = str(case_id)
        
        if domain is None:
            # 从 subject 或 topic 推断 domain
            subject = metadata.get("subject", "").lower()
            topic = metadata.get("topic", "").lower()
            
            # 领域映射
            domain_mapping = {
                "structural biology": "structural_biology",
                "molecular biology": "molecular_biology",
                "quantum physics": "quantum_physics",
                "life science": "life_science",
                "earth science": "earth_science",
                "computer science": "computer_science",
            }
            
            # 尝试从 subject 或 topic 映射
            domain = domain_mapping.get(subject) or domain_mapping.get(topic)
            if not domain:
                # 如果没有映射，使用 subject（如果存在）
                domain = subject.replace(" ", "_") if subject else None

    # 在注册前，尝试根据 load_tools_for_case 写回来的 resolved_dir
    # 自动导入对应目录下的 *_tools_gym.py，从而触发 @Toolbox.register
    try:
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        gym_modules: set[str] = set()
        for entry in tool_protocols:
            if not isinstance(entry, dict):
                continue
            addl = entry.get("additionalProperties") or {}
            resolved_dir = addl.get("resolved_dir")
            if not resolved_dir:
                continue
            dir_path = root / resolved_dir
            if not dir_path.exists() or not dir_path.is_dir():
                continue
            gym_file = dir_path / f"{dir_path.name}_tools_gym.py"
            if not gym_file.exists():
                continue
            rel = gym_file.relative_to(root)
            module_name = ".".join(rel.with_suffix("").parts)
            gym_modules.add(module_name)

        for module_name in gym_modules:
            try:
                importlib.import_module(module_name)
                print(f"✅ 已导入工具注册模块: {module_name}")
            except Exception as e:
                print(f"⚠️ 导入工具注册模块失败 {module_name}: {e}")
    except Exception as e:
        # 导入 *_tools_gym 失败不致命，只影响是否能复用预生成的 EnvironmentTool
        print(f"⚠️ 预加载 *_tools_gym 模块时出错: {e}")

    # 创建环境，传入 case_id 和 domain 以便文件系统自动组织目录
    env = MinimalSciEnv(tool_names=None, case_id=case_id, domain=domain)
    tool_instances: List[EnvironmentTool] = []

    for tool_entry in tool_protocols:
        if not isinstance(tool_entry, dict):
            continue

        fn_block = tool_entry.get("function") or {}
        name = fn_block.get("name")
        if not name:
            continue

        tool_inst: Optional[EnvironmentTool] = None

        # 1) 优先从 Toolbox 中拿已经注册好的 EnvironmentTool 子类
        registry = getattr(Toolbox, "_tool_registry", {})
        cls_and_cfg = registry.get(name)
        if cls_and_cfg:
            tool_cls, _ = cls_and_cfg
            try:
                tool_inst = tool_cls()
                print(f"✅ 使用 Toolbox 中的预注册工具类: {name} -> {tool_cls.__name__}")
            except Exception as e:
                print(f"⚠️ 实例化 Toolbox 工具 {name} 失败: {e}，回退到 GenericFunctionTool")
                tool_inst = None

        # 2) 回退：如果没有预注册工具类，则使用 GenericFunctionTool 包装底层函数
        if tool_inst is None:
            func = function_map.get(name)
            if not callable(func):
                print(f"⚠️ 工具 {name} 在 function_map 中找不到对应的函数，跳过注册")
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

        # 注册到环境
        env.add_tool(tool_inst)
        tool_instances.append(tool_inst)

    if not tool_instances:
        print("⚠️ 没有成功注册任何工具")
        return env, [], [], {}

    # 构建 OpenAI tools schema
    tools_schema = build_tools_schema_from_local_tools(tool_instances)

    # 构建工具注册表
    tool_registry = build_tool_registry_from_local(tool_instances)

    # 重置环境
    env.reset()

    print(f"✅ 成功注册 {len(tool_instances)} 个工具到环境")
    return env, tool_instances, tools_schema, tool_registry


def build_tools_schema_from_local_tools(tools_instances) -> List[Dict[str, Any]]:
    """
    从本文件中绑定到环境的工具实例构建 OpenAI tools schema。
    """
    tools: List[Dict[str, Any]] = []
    for tool in tools_instances:
        tool_schema: Dict[str, Any] = {
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

        if getattr(tool, "arguments", None):
            for param_name, param_info in tool.arguments.items():
                if isinstance(param_info, dict):
                    param_type = param_info.get("type", "number")
                    param_schema: Dict[str, Any] = {
                        "type": param_type,
                        "description": param_info.get("description", ""),
                    }
                    if param_type == "array":
                        items_schema = param_info.get("items") or {"type": "number"}
                        param_schema["items"] = items_schema
                    if "enum" in param_info:
                        param_schema["enum"] = param_info["enum"]

                    tool_schema["function"]["parameters"]["properties"][param_name] = (
                        param_schema
                    )

                    tool_schema["function"]["parameters"]["required"].append(
                        param_name
                    )

        tools.append(tool_schema)

    return tools


def build_tool_registry_from_local(tools_instances) -> Dict[str, Any]:
    """
    构建本地工具实例映射表：name -> instance。
    """
    registry: Dict[str, Any] = {}
    for tool in tools_instances:
        registry[tool.name] = tool
    return registry


def run_tool_call(
    env: Any, tool_name: str, action: Dict[str, Any], tool_call_id: str
) -> Dict[str, Any]:
    """
    执行工具调用，返回 JSON 可序列化的结果。
    这里通过 MinimalSciEnv.step + ToolCall 完整走一遍"agent-环境-工具"交互流程。
    
    参数:
        env: MinimalSciEnv 实例（使用 Any 类型以避免循环导入）
    """
    try:
        tool_action = ToolCall(
            id=tool_call_id,
            name=tool_name,
            arguments=action,
        )

        step_out = env.step(tool_action)
        observation = step_out.observation  # 可能是 Observation，也可能是字符串

        # 统一转成字符串
        obs_str = (
            observation.observation
            if hasattr(observation, "observation")
            else str(observation)
        )

        # 解析 observation 中的结果
        try:
            result = json.loads(obs_str)
            return {
                "status": "success",
                "result": result,
                "raw_observation": obs_str,
            }
        except json.JSONDecodeError:
            # 如果不是 JSON，直接返回字符串
            return {
                "status": "success",
                "result": obs_str,
                "raw_observation": obs_str,
            }
    except Exception as e:
        import traceback as tb

        return {
            "status": "error",
            "error": str(e),
            "traceback": tb.format_exc(),
        }


# ========== 便捷函数：完整流程 ==========

def prepare_env_from_query(
    query_data: Dict[str, Any],
) -> Tuple[Any, List[EnvironmentTool], List[Dict[str, Any]], Dict[str, EnvironmentTool]]:
    """
    简化入口：根据 query_data 的 metadata 自动推断并加载工具目录。
    
    这是推荐的工具加载方式，通过 metadata 中的 subject 和 topic 字段
    自动推断 toolkits/{subject}/{topic}/ 目录，并加载该目录下的所有工具。
    
    优势：
    - 单一入口，无需先调用 load_tools_for_case()
    - 自动加载同一子类下的所有工具
    - 避免重复加载
    
    参数:
        query_data: 测试案例数据，包含 metadata.subject 和 metadata.topic
    
    返回:
        (env, tool_instances, tools_schema, tool_registry)
    
    示例:
        >>> query_data = {
        ...     "id": "case_001",
        ...     "question": "计算薄膜干涉...",
        ...     "metadata": {"subject": "Physics", "topic": "Optics"}
        ... }
        >>> env, tools, schema, registry = prepare_env_from_query(query_data)
    """
    # 延迟导入以避免循环导入
    from gym.env import MinimalSciEnv
    
    metadata = query_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    
    # 提取 case_id
    case_id = (
        query_data.get("id")
        or metadata.get("id")
        or metadata.get("case_id")
        or metadata.get("question_id")
        or metadata.get("original_question_id")
    )
    if case_id:
        case_id = str(case_id)
    
    # 从 subject 推断 domain
    subject = metadata.get("subject", "").lower()
    domain = None
    if subject:
        domain_mapping = {
            "structural biology": "structural_biology",
            "molecular biology": "molecular_biology",
            "quantum physics": "quantum_physics",
            "life science": "life_science",
            "earth science": "earth_science",
            "computer science": "computer_science",
        }
        domain = domain_mapping.get(subject) or subject.replace(" ", "_")
    
    # 通过路径推断加载工具目录
    inferred_path = infer_toolkits_path_from_metadata(metadata)
    
    if not inferred_path:
        subject_val = metadata.get("subject", "")
        topic_val = metadata.get("topic", "")
        print(f"⚠️ 无法推断工具目录: subject={subject_val}, topic={topic_val}")
        # 返回空环境
        env = MinimalSciEnv(tool_names=None, case_id=case_id, domain=domain)
        env.reset()
        return env, [], [], {}
    
    subject_val = metadata.get("subject", "unknown")
    topic_val = metadata.get("topic", "unknown")
    print(f"🔍 自动推断工具目录: {subject_val}/{topic_val} -> {inferred_path}")
    
    # 加载目录下所有工具
    tool_protocols, function_map = load_all_tools_from_directory(inferred_path)
    
    if not tool_protocols:
        print(f"⚠️ 目录 {inferred_path} 下未找到任何工具")
        env = MinimalSciEnv(tool_names=None, case_id=case_id, domain=domain)
        env.reset()
        return env, [], [], {}
    
    # 注册到环境（不再需要 auto_infer_from_metadata，因为已经加载完成）
    return register_tools_to_env(
        tool_protocols,
        function_map,
        case_id=case_id,
        domain=domain,
        query_data=query_data,
        auto_infer_from_metadata=False,  # 已经加载完成，无需再推断
    )

