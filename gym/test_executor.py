"""
精简版测试执行模块（实验性）

目标：
- 在不影响现有 `gym/test_executor.py` 行为的前提下，提供一套更清晰、职责单一的执行流程。
- 优先把「环境 & 工具加载」这部分逻辑简化出来，便于后续迭代。

设计要点：
- 复用旧模块中已经验证过的工具与环境构建逻辑：`_build_env_and_tools_from_loaded`
- 封装一个最小可用的单题调用入口：`simple_test_query`
  - 输入：单个 `query_data`（包含 question / metadata / usage_tool_protocol 等）
  - 输出：模型最终自然语言回答（纯文本），以及可选的对话轨迹（返回值中）

注意：
- 本文件当前不参与原有批量测试 / 评分 / pass@k 等流程，只用于探索和验证新的结构。
- 后续如果效果稳定，可以逐步从这里抽取公共组件，反向简化老的 `test_executor.py`。
"""

from __future__ import annotations

import json
import base64
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _is_text_only_mode() -> bool:
    """
    Issue #6 —— 纯文本模式开关。

    True 时，工具生成的图像 base64 **不会** 被回传给模型，只把结构化 JSON
    结果（去掉 base64 字段后）作为普通 tool message 返回。用于兼容不支持
    ``image_url`` 消息的 OpenAI 兼容 API / 纯文本模型。

    通过 ``SCIAGENT_TEXT_ONLY=1``（或 ``true`` / ``yes``）开启。
    """
    return os.environ.get("SCIAGENT_TEXT_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}


def _strip_embedded_file_fields(result: Any) -> Any:
    """
    在纯文本模式下，剥离 tool result 里所有超大 base64 字段，避免污染 context。
    保留：
      - ``_generated_artifacts``：非图像产物的路径元信息（数据库 / 数据文件 / 报告）
      - ``_embedded_images``：图像产物的路径 / mime / size 元信息（**去掉 base64 字段**）
      - 旧字段 ``_embedded_file_name`` / ``_generated_file_path``：合并成 ``generated_file_path``
    """
    if not isinstance(result, dict):
        return result
    stripped: Dict[str, Any] = {}
    for k, v in result.items():
        if k == "_embedded_images" and isinstance(v, list):
            # 保留元信息，去掉每张图的 base64
            stripped[k] = [
                {ik: iv for ik, iv in img.items() if ik != "base64"} if isinstance(img, dict) else img
                for img in v
            ]
            continue
        if k.startswith("_embedded_file_"):
            # 旧字段：只跳过 base64 本身，其它保留
            if k != "_embedded_file_base64":
                stripped[k] = v
            continue
        stripped[k] = v

    generated_path = result.get("_generated_file_path") or result.get("_embedded_file_name")
    if generated_path and "generated_file_path" not in stripped:
        stripped["generated_file_path"] = generated_path
        stripped["note"] = (
            "text-only mode: embedded image base64 stripped; "
            "agent should treat generated_file_path / _generated_artifacts as artifact references."
        )
    return stripped


# ---------------------------------------------------------------------------
# Tool-generated artifact detection (Issue #6)
# ---------------------------------------------------------------------------
# 由 `scripts/scan_tool_return_paths.py` 扫全库 golden_answer 得到：工具返回值里
# 表示"这里有一个刚落盘的文件"的键名多达 30 种（image_save_path / plot_path /
# composition_pie_chart / …），且相当多是"每个工具起个专属名"。所以侦测策略
# 采用"扫值不扫键"：递归遍历 result 里所有字符串，按扩展名归类。
_IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg"}
_DATABASE_EXTS: set[str] = {".db", ".sqlite", ".sqlite3"}
_DATA_EXTS: set[str] = {
    ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml",
    ".npz", ".npy", ".parquet", ".h5", ".hdf5", ".mat", ".nc",
}
_REPORT_EXTS: set[str] = {".txt", ".md", ".html", ".log"}

_PATH_LIKE_RE = None  # 延迟编译


def _get_path_like_re():
    global _PATH_LIKE_RE
    if _PATH_LIKE_RE is None:
        import re
        exts = "|".join(
            ext.lstrip(".")
            for ext in (_IMAGE_EXTS | _DATABASE_EXTS | _DATA_EXTS | _REPORT_EXTS)
        )
        # 允许任意非空白字符（含中文/CJK/UTF-8 路径），只要以支持的扩展名结尾。
        # 排除空白避免把"一段带路径的自然语句"当成路径匹配。
        _PATH_LIKE_RE = re.compile(rf"^\S+\.({exts})$", re.IGNORECASE)
    return _PATH_LIKE_RE


def _classify_path_by_ext(s: Any) -> Optional[str]:
    """把字符串按后缀分类为 image / database / data / report。不是路径样字符串返回 None。"""
    if not isinstance(s, str) or "\n" in s or len(s) > 400 or not s.strip():
        return None
    m = _get_path_like_re().match(s.strip())
    if not m:
        return None
    ext = "." + m.group(1).lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _DATABASE_EXTS:
        return "database"
    if ext in _DATA_EXTS:
        return "data"
    if ext in _REPORT_EXTS:
        return "report"
    return None


def _mime_from_ext(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".bmp"):
        return "image/bmp"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".svg"):
        return "image/svg+xml"
    return "image/png"


def _detect_generated_artifacts(
    result: Any,
    arguments: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    递归扫描工具的 result + 调用参数，找出所有已经落盘的产物文件。
    返回按发现顺序去重的 list::
        [{path: 绝对/相对路径 str, key_trail: str, category: str, size_bytes: int|None}, ...]
    key_trail 保留完整访问链路（如 "result.metadata.figure_path"），
    方便日志和后续回溯排查。
    """
    seen_paths: set[str] = set()
    hits: List[Dict[str, Any]] = []

    def walk(obj: Any, trail: str) -> None:
        if isinstance(obj, str):
            cls = _classify_path_by_ext(obj)
            if cls is None:
                return
            raw = obj.strip()
            resolved = _resolve_existing_path(raw)
            if resolved is None or not resolved.exists():
                return
            path_str = str(resolved)
            if path_str in seen_paths:
                return
            seen_paths.add(path_str)
            try:
                size = resolved.stat().st_size
            except OSError:
                size = None
            hits.append({
                "path": path_str,
                "key_trail": trail,
                "category": cls,
                "size_bytes": size,
            })
        elif isinstance(obj, dict):
            for k, v in obj.items():
                # 跳过我们自己塞进去的 _embedded_* 元字段，避免重复
                if isinstance(k, str) and k.startswith("_embedded_file_"):
                    continue
                walk(v, f"{trail}.{k}" if trail else str(k))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{trail}[{i}]")

    walk(result, "result")
    if arguments is not None:
        walk(arguments, "args")
    return hits


def _encode_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _build_image_content_part(b64: str, mime_type: str, provider: str) -> Dict[str, Any]:
    """按 provider 组装单张图的 content part（Anthropic vs OpenAI-compatible）。"""
    if provider == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": b64,
            },
        }
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }

# 直接复用现有 agent / data_loader / tool_loader 等模块
from gym.agent import (
    get_client,
    DEFAULT_MODEL,
    REACT_TOOL_SYSTEM_PROMPT,
    TOOL_TRACE_SUFFIX,
    _should_use_react_prompt,
    _coerce_truthy_flag,
)
from gym.core.tool_loader import load_tools_for_case, register_tools_to_env, run_tool_call, prepare_env_from_query
from gym.config.config import SUPPORTED_MODELS
from gym.config.dataset_config import get_trace_root
from gym.env import MinimalSciEnv
from gym.core.data_loader import (
    process_question_with_images_from_metadata,
    extract_golden_answer_template,
    extract_augmented_answer_template,
    extract_structured_answer_from_response,
    ensure_metadata_summary,
    group_cases_by_topic,
    aggregate_usage_tool_protocol_for_cases,
    load_test_cases_from_dataset,
    load_augmented_test_cases_from_dataset,
    load_refined_test_cases_from_dataset,
)
from gym.core.exceptions import TestSkipException

CORE_DIR = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[1]


def _get_tool_mode_answer_prompt() -> str:
    """
    工具模式下使用的简化答案格式模板（仅保留 Answer 部分）。
    目前统一为：要求返回原语言、最终答案用 LaTeX \\boxed{} 包裹。
    """
    return (
        "You should strictly respond in this exact format and answer the question in its original language:\n"
        "example:###Answer###\n"
        "$\\boxed{}$"
    )


def _get_text_mode_reasoning_answer_prompt() -> str:
    """
    纯文本模式下使用的答案格式模板（包含 Reasoning Process + Answer）。
    """
    return (
        "You should strictly respond in this exact format and answer the question in its original language:\n"
        "###Reasoning Process###\n"
        "{Your step by step reasoning process here}\n"
        "\n"
        "###Answer###\n"
        "{The final answer wrapped in LaTeX boxed format $\\boxed{}$}"
    )


def _get_react_tool_system_prompt() -> str:
    """
    ReAct 工具模式下使用的 system prompt。
    直接复用 agent.REACT_TOOL_SYSTEM_PROMPT。
    """
    return REACT_TOOL_SYSTEM_PROMPT


def _build_env_and_tools_for_case(
    query_data: Dict[str, Any],
    auto_infer_from_metadata: bool = True,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    """
    基于单个案例，构建最小环境 + tools schema + tool_registry。

    简化版本：使用 prepare_env_from_query() 单一入口，
    通过 metadata 中的 subject/topic 自动推断并加载 toolkits/{subject}/{topic}/ 下的所有工具。
    
    Args:
        query_data: 测试案例数据
        auto_infer_from_metadata: 是否根据 metadata 中的 subject/topic 自动推断并加载工具目录
                                   默认为 True（推荐），使用路径推断批量加载
    """
    # 初始化变量（用于调试输出和兼容性）
    tool_protocols = None
    function_map = None
    tool_instances = None
    
    if auto_infer_from_metadata:
        # 使用简化入口：通过路径推断批量加载同一子类下的所有工具
        env, tool_instances, tools_schema, tool_registry = prepare_env_from_query(query_data)
    else:
        # 兼容模式：使用旧的加载方式
        tool_protocols, function_map = load_tools_for_case(query_data)
        
        # 提取 case_id 和 domain
        metadata = query_data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        
        case_id = (
            query_data.get("id")
            or metadata.get("id")
            or metadata.get("case_id")
            or metadata.get("question_id")
            or metadata.get("original_question_id")
        )
        if case_id:
            case_id = str(case_id)
        
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
        
        env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
            tool_protocols,
            function_map,
            case_id=case_id,
            domain=domain,
            query_data=query_data,
            auto_infer_from_metadata=False,
        )

    # 兼容「当前案例没有任何工具」或旧实现返回 None 的情况
    if env is None:
        env = MinimalSciEnv(tool_names=None)
    if tools_schema is None:
        tools_schema = []
    if tool_registry is None:
        tool_registry = {}

    # 调试输出：当前案例实际可用的工具情况
    tool_count = len(tool_instances) if tool_instances else len(tool_protocols or [])
    func_count = len(function_map or {}) if function_map else tool_count
    print(
        f"[simple_test_query] 已加载工具数: {tool_count}，"
        f"环境中工具数: {len(tool_registry or {})}"
    )
    if tool_registry:
        names_preview = ", ".join(list(tool_registry.keys())[:10])
        print(f"[simple_test_query] 可用工具名称(前10): {names_preview}")
    else:
        print("[simple_test_query] 当前案例未注册任何工具，将退化为纯对话模式。")

    return env, tools_schema, tool_registry


def _content_to_text(content: Any) -> str:
    """将消息内容安全转换为文本（兼容字符串和多模态结构）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            try:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_value = part.get("text", "")
                    if isinstance(text_value, str) and text_value.strip():
                        parts.append(text_value.strip())
            except Exception:
                continue
        return "\n\n".join(parts)
    try:
        return str(content)
    except Exception:
        return ""


def _extract_choice_message(response, context_label: str):
    """
    提取响应中的第一条消息，若choices为空则抛出更易排查的异常（精简版）。
    """
    choices = getattr(response, "choices", None)
    if not choices:
        debug_payload = None
        if hasattr(response, "model_dump"):
            try:
                debug_payload = response.model_dump()
            except Exception:
                debug_payload = None
        if debug_payload is None:
            try:
                debug_payload = response.__dict__
            except Exception:
                debug_payload = repr(response)
        raise RuntimeError(f"{context_label} 模型响应未包含choices，原始响应: {debug_payload}")
    return choices[0].message


def _is_supported_image_path(path: Optional[str]) -> bool:
    """
    粗略判断字符串是否可能是图片路径（后缀检查）。
    用于 simple_test_query 中的最小图片嵌入逻辑。
    """
    if not path or not isinstance(path, str):
        return False
    lowered = path.strip().lower()
    return lowered.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))


def _resolve_existing_path(path_str: Optional[str]) -> Optional["Path"]:
    """
    简化版路径解析：只做最基本的 exists 检查，避免依赖 test_executor 中的复杂搜索逻辑。
    """
    if not path_str or not isinstance(path_str, str):
        return None
    candidate = Path(path_str).expanduser()
    if candidate.exists():
        return candidate
    return None


def _refresh_preview_image(image_path: Optional[str]) -> None:
    """
    简化版预览更新：当前仅打印提示，不做实际文件复制。
    避免与主执行器的预览逻辑耦合。
    """
    if not image_path:
        return
    print(f"🖼️ 工具生成图片: {image_path}")


def _derive_trace_path_for_debug(
    model_name: str,
    use_tools: bool,
    case_id: Any,
    mode_name: str,
    metadata: Dict[str, Any],
    dataset_filename: str,
) -> Path:
    """
    调试用的精简版 trace 路径推导函数，尽量简单直观：

    根据数据集是否为单模态 / 多模态，自动选择固定的父目录：

        data_analysis/tracetoanalyze/tracesmerged_single_questions
        data_analysis/tracetoanalyze/tracesmerged_questions

    并在其下按如下结构组织：

        [父目录] / model_name / mode_name / {case_id}_trace.json

    注意：这里有意不依赖 gym.test_executor 内部的路径推导逻辑，避免复杂的导入和 sys.modules 副作用。
    """
    # 判断单模态 / 多模态：仅根据数据集文件名是否包含 "single"
    is_single = "single" in dataset_filename.lower()

    # 工程根目录：gym 上一层
    project_root = Path(__file__).resolve().parents[1]
    traces_root = project_root / "data_analysis" / "tracetoanalyze"

    if is_single:
        base_root = traces_root / "tracesmerged_single_questions"
    else:
        base_root = traces_root / "tracesmerged_questions"

    # 进一步按模型名与模式分子目录
    model_dir = model_name or DEFAULT_MODEL
    base_dir = base_root / model_dir / mode_name
    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir / f"{case_id}_trace.json"


def _resolve_trace_root(metadata: Optional[Dict[str, Any]]) -> Path:
    """Determine the base trace directory for a given case."""
    if not isinstance(metadata, dict):
        metadata = {}
    trace_root_override = metadata.get('trace_root')
    if isinstance(trace_root_override, str) and trace_root_override.strip():
        override_path = Path(trace_root_override)
        if not override_path.is_absolute():
            override_path = Path(trace_root_override)
        return override_path
    dataset_key = metadata.get('dataset_key')
    return get_trace_root(dataset_key)


def _resolve_dataset_folder(metadata: Optional[Dict[str, Any]]) -> str:
    """
    根据 dataset_key 是否包含 "single" 来区分单模态和多模态数据集文件夹。
    
    规则：
    - dataset_key 包含 "single" -> "merged_single_questions" (单模态)
    - 否则 -> "merged_questions" (多模态)
    
    优先级：
    1. metadata['dataset_key'] (如果存在)
    2. 从 get_current_dataset_key() 获取
    3. 默认 'merged_questions' (多模态)
    """
    from gym.config.dataset_config import get_current_dataset_key
    
    dataset_key = None
    if isinstance(metadata, dict):
        dataset_key = metadata.get('dataset_key')
    
    if not dataset_key:
        dataset_key = get_current_dataset_key()
    
    # 根据 dataset_key 中是否包含 "single" 来判断
    if dataset_key and "single" in dataset_key.lower():
        return "merged_single_questions"
    else:
        return "merged_questions"


def _sanitize_trace_tag(tag: str) -> str:
    """将trace标签转换为安全的文件夹名称"""
    cleaned = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in tag.strip())
    return cleaned or "default"


def _resolve_mode_folder(use_tools: bool, override: Optional[str] = None) -> str:
    """
    返回模式目录名称。
    - 显式 override 时尊重传入值（可带/不带 react 后缀）。
    - 默认 with_tools 不再自动追加后缀；react 版本需显式指定带后缀的 override。
    """
    if override:
        return override
    base = "with_tools" if use_tools else "without_tools"
    return base


def _derive_trace_path(
    model_name: str,
    use_tools: bool,
    case_id,
    trace_tag: Optional[str] = None,
    mode_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    dataset_filename: Optional[str] = None,
):
    """根据模型、模式和标签生成trace路径
    
    Args:
        model_name: 模型名称
        use_tools: 是否使用工具
        case_id: 案例ID
        trace_tag: 轨迹标签
        mode_name: 模式名称
        metadata: 元数据
        dataset_filename: 数据集文件名（用于判断 single/multi），如果提供则优先使用
    """
    mode_folder = _resolve_mode_folder(use_tools, mode_name)
    sanitized_tag = _sanitize_trace_tag(trace_tag) if trace_tag else None
    
    # 根据数据集文件名判断是 single 还是 multi
    # 优先使用传入的 dataset_filename，其次使用 metadata 中的 _dataset_filename
    filename_to_check = dataset_filename
    if not filename_to_check and isinstance(metadata, dict):
        filename_to_check = metadata.get('_dataset_filename')
    
    if filename_to_check and "single" in filename_to_check.lower():
        data_type_folder = "orignal_data_single"
    else:
        # 如果没有提供文件名，使用 metadata 中的 dataset_key 判断
        dataset_folder = _resolve_dataset_folder(metadata)
        if "single" in dataset_folder.lower():
            data_type_folder = "orignal_data_single"
        else:
            data_type_folder = "orignal_data_multi"
    
    base_dir = _resolve_trace_root(metadata) / model_name / data_type_folder / mode_folder
    if sanitized_tag:
        base_dir /= sanitized_tag
    trace_path = base_dir / f"{case_id}_trace.json" 
    
    return trace_path, sanitized_tag


def _record_skip_event(
    trace_path: Optional[Path],
    case_id: Any,
    model_name: str,
    mode_desc: str,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist skip information alongside printing a friendly log."""
    case_label = case_id if case_id is not None else "unknown"
    print(f"⏭️ 案例 {case_label} ({model_name} · {mode_desc}) 跳过，原因：{reason}")
    payload = {
        "id": case_id,
        "model": model_name,
        "mode": mode_desc,
        "reason": reason,
        "timestamp": time.time(),
    }
    if extra:
        payload["details"] = extra

    if trace_path is None:
        return

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = trace_path.name
    if base_name.endswith("_trace.json"):
        skip_name = base_name.replace("_trace.json", "_skip.json")
    else:
        skip_name = f"{trace_path.stem}_skip.json"
    skip_path = trace_path.with_name(skip_name)
    try:
        with skip_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"📝 跳过记录已保存: {skip_path}")
    except Exception as exc:
        print(f"⚠️ 保存跳过记录失败 ({skip_path}): {exc}")


def _parse_glm_text_tool_calls(content: str, provider: Optional[str] = None) -> Optional[List[Any]]:
    """
    解析 GLM 模型返回的文本格式工具调用。
    
    GLM 在使用 ReAct prompt 时，可能返回如下格式：
    Action: tool_name
    <arg_key>param1</arg_key>
    <arg_value>value1</arg_value>
    ...
    </tool_call>
    
    Args:
        content: 助手消息的文本内容
        provider: 模型提供商（用于判断是否需要解析）
    
    Returns:
        解析后的 tool_calls 列表，格式与 OpenAI 兼容，如果无法解析则返回 None
    """
    import re
    import uuid
    
    # 只对 GLM/ZhipuAI 进行解析
    if provider not in ("zhipuai", "glm"):
        return None
    
    if not content or not isinstance(content, str):
        return None
    
    # 查找 Action: 开头的工具调用
    # 匹配模式：Action: tool_name 后跟参数，直到 </tool_call> 或下一个 Action/Thought/Final Answer
    action_pattern = r'Action:\s*(\w+)\s*\n(.*?)(?=</tool_call>|\nAction:|\nThought:|\nFinal Answer:|$)'
    matches = list(re.finditer(action_pattern, content, re.DOTALL | re.IGNORECASE))
    
    # 如果没有匹配到，尝试更宽松的模式（不要求换行）
    if not matches:
        action_pattern = r'Action:\s*(\w+)\s*(.*?)(?=</tool_call>|\nAction:|\nThought:|\nFinal Answer:|$)'
        matches = list(re.finditer(action_pattern, content, re.DOTALL | re.IGNORECASE))
    
    if not matches:
        # 调试：打印部分内容以便排查
        content_preview = content[:500] if len(content) > 500 else content
        print(f"⚠️  未找到 Action: 模式，内容预览: {repr(content_preview)}")
        return None
    
    tool_calls = []
    for match in matches:
        tool_name = match.group(1).strip()
        args_text = match.group(2).strip()
        
        if not tool_name:
            continue
        
        print(f"🔍 找到工具调用: {tool_name}，参数文本: {args_text[:200]}...")
        
        # 解析参数：查找 <arg_key> 和 <arg_value> 对
        arg_pattern = r'<arg_key>([^<]+)</arg_key>\s*<arg_value>([^<]+)</arg_value>'
        arg_matches = list(re.finditer(arg_pattern, args_text))
        
        arguments = {}
        for arg_match in arg_matches:
            key = arg_match.group(1).strip()
            value_str = arg_match.group(2).strip()
            
            # 尝试解析值（可能是 JSON、数字、字符串等）
            try:
                # 尝试解析为 JSON（处理数组、对象等）
                value = json.loads(value_str)
            except json.JSONDecodeError:
                # 尝试解析为数字
                try:
                    if '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    # 检查是否是 null
                    if value_str.lower() in ('null', 'none'):
                        value = None
                    else:
                        # 保持为字符串
                        value = value_str
            
            arguments[key] = value
            print(f"   参数: {key} = {value} (类型: {type(value).__name__})")
        
        if not arguments:
            print(f"⚠️  工具 {tool_name} 未找到任何参数（可能是无参数工具）")
        
        # 如果找到了工具名，创建 tool_call 对象（即使没有参数也可以）
        if tool_name:
            # 创建类似 OpenAI 格式的 tool_call
            tool_call_dict = {
                'id': f'call_{uuid.uuid4().hex[:16]}',
                'type': 'function',
                'function': {
                    'name': tool_name,
                    'arguments': json.dumps(arguments, ensure_ascii=False)
                }
            }
            
            # 创建一个简单的对象来模拟 tool_call
            class ToolCall:
                def __init__(self, data):
                    self.id = data['id']
                    self.type = data['type']
                    self.function = type('Function', (), {
                        'name': data['function']['name'],
                        'arguments': data['function']['arguments']
                    })()
            
            tool_calls.append(ToolCall(tool_call_dict))
            print(f"✅ 解析到工具调用: {tool_name}，参数: {arguments}")
    
    return tool_calls if tool_calls else None


def _detect_provider(model_name: str) -> Optional[str]:
    """
    根据模型名称在 SUPPORTED_MODELS 中查找 provider，用于判断是否需要走 GLM 文本工具调用解析逻辑。
    """
    cfg = SUPPORTED_MODELS.get(model_name) or {}
    provider = cfg.get("provider")
    if isinstance(provider, str):
        return provider.lower()
    return None


def _build_basic_user_message(
    query_data: Dict[str, Any],
    test_type: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    构造最基础的用户消息：
    - 使用 metadata 中的图片信息，通过 process_question_with_images_from_metadata 统一处理
    - 对于有图片的情况，图片在前、文本在后
    - 暂不注入复杂的「模板 / 结构化 JSON 格式要求」，只保证问题本身被清洗与保留
    """
    question_data = process_question_with_images_from_metadata(query_data)
    user_text = question_data.get("text") or str(query_data.get("question", ""))

    if not user_text.strip():
        raise ValueError("问题内容为空，无法构造用户消息。")

    # Issue #6: text-only 模式下丢弃 image_url 段
    images = [] if _is_text_only_mode() else (question_data.get("images") or [])
    if images:
        content_parts: List[Dict[str, Any]] = []
        # 先放所有图片
        for img in images:
            mime_type = img.get("mime_type") or "image/png"
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img['base64']}",
                    },
                }
            )
        # 再放文本
        content_parts.append({"type": "text", "text": user_text})
        user_message: Dict[str, Any] = {"role": "user", "content": content_parts}
    else:
        user_message = {"role": "user", "content": user_text}

    return question_data, user_message


def simple_test_query(
    query_data: Dict[str, Any],
    model_name: Optional[str] = None,
    use_tools: bool = True,
    max_rounds: int = 50,
    auto_infer_from_metadata: bool = True,
) -> Dict[str, Any]:
    """
    精简版单题执行入口：

    功能：
    - 为单个案例加载本地工具与环境（通过 `_build_env_and_tools_from_loaded`）
    - 构造最基础的多模态用户消息
    - 在有工具的情况下，执行一个「单工具调用链」的多轮对话循环
    - 返回：
        {
          "final_answer": <str>,       # 模型最终自然语言回答（文本）
          "messages": [...],           # 对话轨迹（可选，用于调试）
          "model_name": <str>,
          "used_tools": bool,
        }
    说明：
    - 不做模板 / 评分 / trace 落盘，只关注「问题 → 工具调用 → 回答」主干流程。
    
    Args:
        query_data: 测试案例数据
        model_name: 指定的模型名称
        use_tools: 是否使用工具
        max_rounds: 最大对话轮数
        auto_infer_from_metadata: 是否根据 metadata 中的 subject/topic 自动推断并加载工具目录
                                   默认为 True，会自动加载 toolkits/{subject}/{topic}/ 下的所有工具
    """
    current_model = model_name or DEFAULT_MODEL
    client = get_client(current_model)

    # 1. 基于_build_env_and_tools_for_case始终构建环境与（可能为空的）工具注册表（支持自动推断）
    env, tools_schema, tool_registry = _build_env_and_tools_for_case(
        query_data, auto_infer_from_metadata=auto_infer_from_metadata
    )

    # 2. 基于_build_basic_user_message构造用户消息（多模态）——与旧版 test_query 复用同一模块
    metadata = query_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    test_type = metadata.get("test_type", "normal")
    question_data, user_message = _build_basic_user_message(query_data, test_type)

    # 注入格式模板要求，确保模型输出符合 extract_boxed_answer 的预期格式
    user_content_base = question_data.get("text") or ""
    if not user_content_base.strip():
        raise ValueError("问题内容为空，无法构造用户消息。")

    # 如果使用工具，移除格式要求中的推理过程部分
    if use_tools and tools_schema:
        # 查找格式要求的起始位置
        format_start = user_content_base.find("You should strictly respond in this exact format")
        if format_start != -1:
            # 提取格式要求之前的问题部分
            question_part = user_content_base[:format_start].strip()

            # 构建简化的格式要求（只保留Answer部分）
            simplified_format = _get_tool_mode_answer_prompt()

            # 组合新的用户内容
            user_content = question_part + "\n" + simplified_format

            print("✂️ 已移除推理过程格式要求")
        else:
            # 构建简化的格式要求（只保留Answer部分）
            simplified_format = _get_tool_mode_answer_prompt()

            user_content = user_content_base + "\n" + simplified_format
    else:
        # 纯文本模式：构建包含推理过程的格式要求
        simplified_format = _get_text_mode_reasoning_answer_prompt()

        user_content = user_content_base + "\n" + simplified_format

    # 更新用户消息中的文本部分（保持多模态结构不变）
    if isinstance(user_message.get("content"), list):
        # 多模态：找到最后一个 text 类型片段并替换为带格式的文本
        replaced = False
        for part in reversed(user_message["content"]):
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = user_content
                replaced = True
                break
        if not replaced:
            user_message["content"].append({"type": "text", "text": user_content})
    else:
        # 纯文本：直接替换为带格式的文本
        user_message["content"] = user_content

    # 如果使用工具，添加 ReAct system prompt 以更好地引导工具调用
    messages: List[Dict[str, Any]] = []
    if use_tools and tools_schema:
        messages.append({"role": "system", "content": _get_react_tool_system_prompt()})
    messages.append(user_message)

    # 3. 主循环：有工具 vs 无工具
    if not use_tools or not tools_schema:
        # 纯文本 / 纯多模态单轮调用
        response = client.chat_completions_create(messages=messages)
        assistant_message = _extract_choice_message(response, "simple_text_only")
        final_content = _content_to_text(getattr(assistant_message, "content", None))
        messages.append(
            {
                "role": "assistant",
                "content": getattr(assistant_message, "content", None),
            }
        )
        return {
            "final_answer": final_content,
            "messages": messages,
            "model_name": current_model,
            "used_tools": False,
        }

    # 4. 工具模式多轮对话
    provider = _detect_provider(current_model)
    round_count = 0
    final_answer_text: str = ""

    while round_count < max_rounds:
        round_count += 1

        response = client.chat_completions_create(
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        assistant_message = _extract_choice_message(response, f"simple_round_{round_count}")
        messages.append(assistant_message)

        # 打印 / 记录文本形式，方便调用方调试
        assistant_text = _content_to_text(getattr(assistant_message, "content", None))

        # 尝试读取工具调用
        tool_calls = getattr(assistant_message, "tool_calls", None)

        # 针对 GLM：如果 tool_calls 为空，尝试从文本中解析
        if not tool_calls and provider in ("zhipuai", "glm"):
            print(f"[simple_test_query] GLM 文本格式解析：provider={provider}, 内容长度={len(assistant_text)}")
            if "Action:" in assistant_text:
                print(f"[simple_test_query] 检测到 Action: 模式，开始解析...")
            parsed = _parse_glm_text_tool_calls(assistant_text, provider)
            if parsed:
                print(f"[simple_test_query] ✅ 成功解析 GLM 文本格式工具调用，数量: {len(parsed)}")
                tool_calls = parsed
            else:
                print(f"[simple_test_query] ⚠️ GLM 文本格式解析失败，内容预览: {assistant_text[:300]}")

        # 调试输出：本轮工具调用数量
        if tool_calls:
            print(f"[simple_test_query] 第 {round_count} 轮返回工具调用数: {len(tool_calls)}")
        else:
            print(f"[simple_test_query] 第 {round_count} 轮未返回工具调用，结束对话。")

        # 如果没有工具调用，视为对话结束
        if not tool_calls:
            final_answer_text = assistant_text
            break

        # 当前实现：顺序执行每一个工具调用
        for tool_call in tool_calls:
            raw_arguments = getattr(tool_call.function, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except Exception:
                arguments = {}

            tool_name = getattr(tool_call.function, "name", None)
            if not tool_name or tool_name not in (tool_registry or {}):
                # 未知工具：直接把错误反馈给模型
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"error": f"未知工具: {tool_name}"},
                        ensure_ascii=False,
                    ),
                }
                messages.append(tool_msg)
                continue

            # 调用环境中的工具
            try:
                from gym.core.tool_loader import run_tool_call

                tool_result = run_tool_call(env, tool_name, arguments, tool_call.id)
                result = tool_result.get("result") 
                print("tool execute result",result)
            except Exception as e:
                tool_result = {"status": "error", "error": str(e)}
                result = tool_result

            # 尝试处理「工具返回路径 → 图片 / 文件嵌入」的情况（复用老逻辑）
            # 注意：这里只做最小化处理，不修改原始结果结构
            try:
                if isinstance(result, dict) and "filename" in result:
                    fname = str(result["filename"])
                    resolved = _resolve_existing_path(fname)
                    if resolved:
                        import base64

                        with open(resolved, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("ascii")
                        result["_embedded_file_base64"] = b64
                        result["_embedded_file_name"] = resolved.name
                        result["_generated_file_path"] = str(resolved)
                        _refresh_preview_image(str(resolved))
                elif isinstance(result, str):
                    candidate_path = result.strip().strip('"').strip("'")
                    if _is_supported_image_path(candidate_path):
                        resolved_candidate = _resolve_existing_path(candidate_path)
                        if resolved_candidate and resolved_candidate.exists():
                            import base64

                            with open(resolved_candidate, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("ascii")
                            result = {
                                "original_result": candidate_path,
                                "_embedded_file_base64": b64,
                                "_embedded_file_name": resolved_candidate.name,
                                "_generated_file_path": str(resolved_candidate),
                            }
                            _refresh_preview_image(str(resolved_candidate))
            except Exception:
                # 图片处理失败不应影响主流程
                pass

            tool_content = json.dumps(result, ensure_ascii=False, default=str)

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content,
            }
            messages.append(tool_message)

    if round_count >= max_rounds and not final_answer_text:
        final_answer_text = assistant_text + "\n[对话轮数达到上限，自动终止]"

    return {
        "final_answer": final_answer_text,
        "messages": messages,
        "model_name": current_model,
        "used_tools": True,
        "rounds": round_count,
    }


def simple_test_refine_query(
    query_data: Dict[str, Any],
    model_name: Optional[str] = None,
    use_tools: bool = True,
    max_rounds: int = 50,
) -> Dict[str, Any]:
    """
    精简版精炼数据测试执行入口：
    
    功能：
    - 专门用于测试精炼版数据（refined_versions），这些数据已经通过 load_refined_test_cases_from_dataset 重构
    - 为单个案例加载本地工具与环境
    - 构造多模态用户消息
    - 执行工具调用循环（如果启用）
    - 提取结构化答案并进行过程评分
    
    返回：
        {
          "final_answer": <str>,              # 模型最终自然语言回答（文本）
          "structured_answer": <dict>,       # 提取的结构化答案（用于评分）
          "score": <float>,                  # 评分结果（0.0-1.0）
          "score_summary": <str>,            # 评分摘要
          "score_details": <dict>,           # 评分详细信息
          "messages": [...],                 # 对话轨迹
          "model_name": <str>,
          "used_tools": bool,
        }
    
    说明：
    - 精炼版数据已经通过 load_refined_test_cases_from_dataset 重构，query_data 结构包含：
      - question: refined_question
      - answer: final_answer (字符串)
      - metadata.golden_answer: final_answer (列表格式)
      - metadata.test_type: 'refined'
    - 使用 calculate_answer_score 进行过程评分，需要结构化答案
    """
    current_model = model_name or DEFAULT_MODEL
    client = get_client(current_model)

    # 1. 基于_build_env_and_tools_for_case始终构建环境与（可能为空的）工具注册表
    env, tools_schema, tool_registry = _build_env_and_tools_for_case(query_data)

    # 2. 基于_build_basic_user_message构造用户消息（多模态）
    # 注意：精炼版数据已经通过 load_refined_test_cases_from_dataset 重构，
    # query_data 的结构与普通案例类似，可以直接使用 _build_basic_user_message
    metadata = query_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    test_type = metadata.get("test_type", "refined")  # 精炼版默认为 'refined'
    question_data, user_message = _build_basic_user_message(query_data, test_type)

    # 注入格式模板要求，确保模型输出结构化答案
    user_content_base = question_data.get("text") or ""
    if not user_content_base.strip():
        raise ValueError("问题内容为空，无法构造用户消息。")

    # 精炼版数据需要结构化答案，所以格式要求应该引导模型输出 JSON 格式
    # 如果使用工具，移除格式要求中的推理过程部分
    if use_tools and tools_schema:
        # 查找格式要求的起始位置
        format_start = user_content_base.find("You should strictly respond in this exact format")
        if format_start != -1:
            # 提取格式要求之前的问题部分
            question_part = user_content_base[:format_start].strip()
            # 构建简化的格式要求（只保留Answer部分）
            simplified_format = _get_tool_mode_answer_prompt()
            user_content = question_part + "\n" + simplified_format
            print("✂️ 已移除推理过程格式要求")
        else:
            simplified_format = _get_tool_mode_answer_prompt()
            user_content = user_content_base + "\n" + simplified_format
    else:
        # 纯文本模式：构建包含推理过程的格式要求
        simplified_format = _get_text_mode_reasoning_answer_prompt()
        user_content = user_content_base + "\n" + simplified_format

    # 更新用户消息中的文本部分（保持多模态结构不变）
    if isinstance(user_message.get("content"), list):
        # 多模态：找到最后一个 text 类型片段并替换为带格式的文本
        replaced = False
        for part in reversed(user_message["content"]):
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = user_content
                replaced = True
                break
        if not replaced:
            user_message["content"].append({"type": "text", "text": user_content})
    else:
        # 纯文本：直接替换为带格式的文本
        user_message["content"] = user_content

    # 3. 如果使用工具，添加 ReAct system prompt 以更好地引导工具调用
    messages: List[Dict[str, Any]] = []
    if use_tools and tools_schema:
        messages.append({"role": "system", "content": _get_react_tool_system_prompt()})
    messages.append(user_message)

    # 4. 主循环：有工具 vs 无工具
    if not use_tools or not tools_schema:
        # 纯文本 / 纯多模态单轮调用
        response = client.chat_completions_create(messages=messages)
        assistant_message = _extract_choice_message(response, "simple_refine_text_only")
        final_content = _content_to_text(getattr(assistant_message, "content", None))
        messages.append(
            {
                "role": "assistant",
                "content": getattr(assistant_message, "content", None),
            }
        )
        
        # 提取结构化答案并进行评分
        structured_answer, score, score_summary, score_details = _evaluate_refined_answer(
            final_content, query_data
        )
        
        return {
            "final_answer": final_content,
            "structured_answer": structured_answer,
            "score": score,
            "score_summary": score_summary,
            "score_details": score_details,
            "messages": messages,
            "model_name": current_model,
            "used_tools": False,
        }

    # 5. 工具模式多轮对话
    provider = _detect_provider(current_model)
    round_count = 0
    final_answer_text: str = ""

    while round_count < max_rounds:
        round_count += 1

        response = client.chat_completions_create(
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            parallel_tool_calls=False,
        )

        assistant_message = _extract_choice_message(response, f"simple_refine_round_{round_count}")
        messages.append(assistant_message)

        # 打印 / 记录文本形式，方便调用方调试
        assistant_text = _content_to_text(getattr(assistant_message, "content", None))

        # 尝试读取工具调用
        tool_calls = getattr(assistant_message, "tool_calls", None)

        # 针对 GLM：如果 tool_calls 为空，尝试从文本中解析
        if not tool_calls and provider in ("zhipuai", "glm"):
            print(f"[simple_test_refine_query] GLM 文本格式解析：provider={provider}, 内容长度={len(assistant_text)}")
            if "Action:" in assistant_text:
                print(f"[simple_test_refine_query] 检测到 Action: 模式，开始解析...")
            parsed = _parse_glm_text_tool_calls(assistant_text, provider)
            if parsed:
                print(f"[simple_test_refine_query] ✅ 成功解析 GLM 文本格式工具调用，数量: {len(parsed)}")
                tool_calls = parsed
            else:
                print(f"[simple_test_refine_query] ⚠️ GLM 文本格式解析失败，内容预览: {assistant_text[:300]}")

        # 调试输出：本轮工具调用数量
        if tool_calls:
            print(f"[simple_test_refine_query] 第 {round_count} 轮返回工具调用数: {len(tool_calls)}")
        else:
            print(f"[simple_test_refine_query] 第 {round_count} 轮未返回工具调用，结束对话。")

        # 如果没有工具调用，视为对话结束
        if not tool_calls:
            final_answer_text = assistant_text
            break

        # 当前实现：顺序执行每一个工具调用
        for tool_call in tool_calls:
            raw_arguments = getattr(tool_call.function, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except Exception:
                arguments = {}

            tool_name = getattr(tool_call.function, "name", None)
            if not tool_name or tool_name not in (tool_registry or {}):
                # 未知工具：直接把错误反馈给模型
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        {"error": f"未知工具: {tool_name}"},
                        ensure_ascii=False,
                    ),
                }
                messages.append(tool_msg)
                continue

            # 调用环境中的工具
            try:
                from gym.core.tool_loader import run_tool_call

                tool_result = run_tool_call(env, tool_name, arguments, tool_call.id)
                result = tool_result.get("result")
                print("tool execute result", result)
            except Exception as e:
                tool_result = {"status": "error", "error": str(e)}
                result = tool_result

            # 尝试处理「工具返回路径 → 图片 / 文件嵌入」的情况
            try:
                if isinstance(result, dict) and "filename" in result:
                    fname = str(result["filename"])
                    resolved = _resolve_existing_path(fname)
                    if resolved:
                        import base64

                        with open(resolved, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("ascii")
                        result["_embedded_file_base64"] = b64
                        result["_embedded_file_name"] = resolved.name
                        result["_generated_file_path"] = str(resolved)
                        _refresh_preview_image(str(resolved))
                elif isinstance(result, str):
                    candidate_path = result.strip().strip('"').strip("'")
                    if _is_supported_image_path(candidate_path):
                        resolved_candidate = _resolve_existing_path(candidate_path)
                        if resolved_candidate and resolved_candidate.exists():
                            import base64

                            with open(resolved_candidate, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("ascii")
                            result = {
                                "original_result": candidate_path,
                                "_embedded_file_base64": b64,
                                "_embedded_file_name": resolved_candidate.name,
                                "_generated_file_path": str(resolved_candidate),
                            }
                            _refresh_preview_image(str(resolved_candidate))
            except Exception:
                # 图片处理失败不应影响主流程
                pass

            tool_content = json.dumps(result, ensure_ascii=False, default=str)

            tool_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content,
            }
            messages.append(tool_message)

    if round_count >= max_rounds and not final_answer_text:
        final_answer_text = assistant_text + "\n[对话轮数达到上限，自动终止]"

    # 提取结构化答案并进行评分
    structured_answer, score, score_summary, score_details = _evaluate_refined_answer(
        final_answer_text, query_data
    )

    return {
        "final_answer": final_answer_text,
        "structured_answer": structured_answer,
        "score": score,
        "score_summary": score_summary,
        "score_details": score_details,
        "messages": messages,
        "model_name": current_model,
        "used_tools": True,
        "rounds": round_count,
    }


def _evaluate_refined_answer(
    final_answer_text: str,
    query_data: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], float, str, Dict[str, Any]]:
    """
    评估精炼版答案：提取结构化答案并使用 calculate_answer_score 进行评分。
    
    Args:
        final_answer_text: 模型的最终回答文本
        query_data: 测试案例数据，包含 metadata.golden_answer
    
    Returns:
        Tuple: (structured_answer, score, score_summary, score_details)
        - structured_answer: 提取的结构化答案（字典），如果提取失败则为 None
        - score: 评分（0.0-1.0）
        - score_summary: 评分摘要（字符串）
        - score_details: 评分详细信息（字典）
    """
    from gym.core.data_loader import extract_structured_answer_from_response
    from gym.core.evaluator import calculate_answer_score

    # 提取结构化答案
    structured_answer = extract_structured_answer_from_response(final_answer_text)
    
    if structured_answer is None:
        print("⚠️ 无法从模型回答中提取结构化答案，评分将失败")
        return None, 0.0, "无法提取结构化答案", {}

    # 获取标准答案（golden_answer）
    metadata = query_data.get("metadata", {})
    golden_answer_list = metadata.get("golden_answer", [])
    
    if not golden_answer_list:
        print("⚠️ 未找到标准答案（golden_answer），无法进行评分")
        return structured_answer, 0.0, "未找到标准答案", {}

    # golden_answer 是列表格式，取第一个元素
    golden_answer = golden_answer_list[0] if isinstance(golden_answer_list, list) else golden_answer_list
    
    # 如果 golden_answer 是字典且包含 'final_answer' 键，提取它
    if isinstance(golden_answer, dict) and "final_answer" in golden_answer:
        golden_answer = golden_answer["final_answer"]
    
    # 确保 golden_answer 是字典格式（calculate_answer_score 需要）
    if not isinstance(golden_answer, dict):
        # 尝试将字符串或其他格式转换为字典
        try:
            if isinstance(golden_answer, str):
                golden_answer = json.loads(golden_answer)
            else:
                # 包装为字典
                golden_answer = {"answer": golden_answer}
        except Exception:
            golden_answer = {"answer": golden_answer}

    # 确保 structured_answer 也是字典格式
    if not isinstance(structured_answer, dict):
        structured_answer = {"answer": structured_answer}

    # 使用 calculate_answer_score 进行评分
    try:
        score, score_summary, score_details = calculate_answer_score(
            model_answer=structured_answer,
            golden_standard=golden_answer,
            tolerance=0.05,
        )
        print(f"✅ 评分完成: {score:.2f} - {score_summary}")
        return structured_answer, score, score_summary, score_details
    except Exception as e:
        print(f"❌ 评分过程出错: {e}")
        import traceback
        traceback.print_exc()
        return structured_answer, 0.0, f"评分出错: {e}", {}


def _load_tools_and_build_env_for_case(
    query_data: Dict[str, Any],
    use_tools: bool,
    load_all_topic_tools: bool,
    test_type: str,
    auto_infer_from_metadata: bool = True,
) -> Tuple[
    Dict[str, Any],
    Optional["MinimalSciEnv"],
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, Any]],
    Optional[List[Dict[str, Any]]],
    Optional[Dict[str, Any]],
]:
    """
    统一封装「根据案例加载工具并构建环境」的逻辑。

    职责：
    - 根据 load_all_topic_tools / metadata 决定是否做 topic 级工具聚合
    - 如果 auto_infer_from_metadata=True，根据 metadata 中的 subject/topic 自动推断工具目录
    - 调用 gym.tool_loader.load_tools_for_case / register_tools_to_env
    - 返回更新后的 query_data（可能包含合并后的 usage_tool_protocol）和：
        env, tools_schema, tool_registry, tool_protocols, function_map

    当 use_tools=False 时：
    - 不加载任何工具，仅返回原始 query_data，其余返回 None。
    
    Args:
        query_data: 测试案例数据
        use_tools: 是否使用工具
        load_all_topic_tools: 是否加载相同 topic 的所有工具
        test_type: 测试类型
        auto_infer_from_metadata: 是否根据 metadata 中的 subject/topic 自动推断并加载工具目录
                                   默认为 True，会自动加载 toolkits/{subject}/{topic}/ 下的所有工具
    """
    if not use_tools:
        print("📝 纯文本模式，不使用任何工具")
        return query_data, None, None, None, None, None

    metadata = query_data.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    query_data["metadata"] = metadata

    # 检查是否需要加载相同 topic 的所有工具
    should_load_all_topic_tools = load_all_topic_tools or _coerce_truthy_flag(
        metadata.get("load_all_topic_tools")
    )

    if should_load_all_topic_tools:
        current_topic = metadata.get("topic") or (query_data.get("metadata_summary") or {}).get(
            "topic"
        )

        if current_topic:
            print(f"🔧 启用全topic工具加载模式，topic: {current_topic}")

            # 根据 test_type 选择合适的数据加载函数
            all_cases: List[Dict[str, Any]] = []
            try:
                if test_type == "augmented":
                    all_cases = load_augmented_test_cases_from_dataset()
                elif test_type == "refined":
                    dataset_key = metadata.get("dataset_key")
                    all_cases = load_refined_test_cases_from_dataset(dataset_key=dataset_key)
                else:
                    all_cases = load_test_cases_from_dataset()

                topic_map = group_cases_by_topic(all_cases)
                topic_cases = topic_map.get(current_topic, [])

                if topic_cases:
                    print(f"📚 找到 {len(topic_cases)} 个相同topic的案例")
                    aggregated_protocol = aggregate_usage_tool_protocol_for_cases(topic_cases)

                    if aggregated_protocol:
                        print(f"📦 聚合得到 {len(aggregated_protocol)} 个工具协议")
                        current_protocols = query_data.get("usage_tool_protocol", []) or []
                        merged_protocols = list(current_protocols) + list(aggregated_protocol)

                        temp_case = deepcopy(query_data)
                        temp_case["usage_tool_protocol"] = merged_protocols
                        temp_metadata = temp_case.get("metadata") or {}
                        temp_metadata["topic_protocol_scope"] = current_topic
                        temp_metadata["with_all_tools"] = True
                        temp_case["metadata"] = temp_metadata
                        query_data = temp_case

                        print(f"✅ 已合并工具协议，总计 {len(merged_protocols)} 个工具协议")
                    else:
                        print(
                            f"⚠️ topic {current_topic} 未找到任何工具协议，"
                            f"使用当前案例的工具"
                        )
                else:
                    print(f"⚠️ 未找到topic {current_topic} 的其他案例，使用当前案例的工具")
            except Exception as e:
                print(f"⚠️ 加载全topic工具失败: {e}，将使用当前案例的工具")
                import traceback
                traceback.print_exc()
        else:
            print("⚠️ 当前案例没有topic信息，无法加载全topic工具，使用当前案例的工具")

    # 步骤1: 加载工具协议和函数（从 usage_tool_protocol 字段）
    tool_protocols, function_map = load_tools_for_case(query_data)

    # 步骤2: 注册到环境，同时支持自动从 metadata 推断工具目录
    env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
        tool_protocols,
        function_map,
        query_data=query_data,
        auto_infer_from_metadata=auto_infer_from_metadata,
    )
    tools = tools_schema

    tool_count = len(tools_schema) if tools_schema else 0
    print(
        f"✅ 加载了 {tool_count} 个工具协议，{len(function_map)} 个函数，"
        f"构建环境中的工具数: {len(tool_registry)}"
    )

    if tool_count > 150:
        print(
            f"⚠️  警告：工具数量 ({tool_count}) 较多，可能导致输入超出模型上下文长度限制"
        )
        print(f"   如果遇到 'Input is too long' 错误，请考虑：")
        print(f"   1. 关闭 load_all_topic_tools 选项（仅使用当前案例的工具）")
        print(f"   2. 使用支持更大上下文的模型")
        print(f"   3. 检查是否真的需要加载这么多工具")
    elif tool_count > 100:
        print(
            f"ℹ️  提示：工具数量 ({tool_count}) 较多，"
            f"如果遇到输入长度问题，可考虑关闭 load_all_topic_tools"
        )

    return query_data, env, tools, tool_registry, tool_protocols, function_map


def test_query(
    query_data,
    model_name=None,
    use_tools=True,
    trace_tag: Optional[str] = None,
    mode_name: Optional[str] = None,
    force_retest: bool = False,
    load_all_topic_tools: bool = False,
    auto_infer_from_metadata: bool = True,
):
    """测试单个查询 - 支持选择是否使用工具

    Args:
        query_data: 测试案例数据
        model_name: 指定的模型名称
        use_tools: 是否使用工具，True表示使用工具，False表示纯文本对话
        trace_tag: 轨迹子目录标签
        mode_name: 指定模式目录名称
        force_retest: 是否忽略现有缓存并强制重新调用模型
        load_all_topic_tools: 是否加载相同topic的所有工具（默认False）
        auto_infer_from_metadata: 是否根据 metadata 中的 subject/topic 自动推断并加载工具目录
                                   默认为 True，会自动加载 toolkits/{subject}/{topic}/ 下的所有工具
    """ 
    ## 注册Agent
    test_client = get_client(model_name) if model_name else get_client(DEFAULT_MODEL)
    current_model = model_name or DEFAULT_MODEL

    metadata = query_data.get('metadata') or {}
    if not isinstance(metadata, dict):
        metadata = {}
    query_data['metadata'] = metadata

    effective_mode_name = mode_name
    if effective_mode_name is None:
        meta_override = metadata.get('mode_name') or metadata.get('mode_folder')
        if meta_override:
            effective_mode_name = str(meta_override)
        elif metadata.get('with_all_tools'):
            effective_mode_name = "with_all_tools"
    ## Agent的调用范式
    use_react_prompt = _should_use_react_prompt(use_tools, effective_mode_name, metadata)
    if use_react_prompt and effective_mode_name in (None, "with_tools"):
        suffix = (TOOL_TRACE_SUFFIX or "").strip() or "_react"
        effective_mode_name = f"with_tools{suffix}"

    mode_desc = effective_mode_name or ("使用工具" if use_tools else "不使用工具")
    test_type = metadata.get('test_type', 'normal')

    case_id = (
        query_data.get('id')
        or metadata.get('id')
        or metadata.get('case_id')
        or metadata.get('question_id')
    )
    trace_path = None
    sanitized_tag = None
    if case_id is None:
        print("⚠️ 当前案例缺少可识别的ID，无法进行缓存检查。")
    else:
        # 从 metadata 中获取数据集文件名（如果存在）
        dataset_filename = metadata.get('_dataset_filename') if isinstance(metadata, dict) else None
        
        trace_path, sanitized_tag = _derive_trace_path(
            current_model,
            use_tools if effective_mode_name is None else True,
            case_id,
            trace_tag,
            mode_name=effective_mode_name,
            metadata=metadata,
            dataset_filename=dataset_filename,
        )
        cache_available = trace_path.exists()
        metadata_force_flag = (
            _coerce_truthy_flag(metadata.get("force_retest")) or
            _coerce_truthy_flag(metadata.get("force_reload"))
        )
        force_retest = bool(force_retest or metadata_force_flag)

    ## 开始测试 
    print(f"\nStart:*** 开始测试：\n 测试题目: {query_data['id']} \n(模型: {current_model}, \n 模式: {mode_desc}, \n 类型: {test_type}) ===")

    # 根据测试类型选择合适的模板提取函数
    if test_type == 'augmented':
        answer_template, golden_standard = extract_augmented_answer_template(query_data)
        print(f"📋 增强版测试案例，原始ID: {query_data.get('original_id')}")
    else:
        answer_template, golden_standard = extract_golden_answer_template(query_data)
    
    if answer_template:
        print(f"📋 规范回答模板: {json.dumps(answer_template, ensure_ascii=False, indent=2)}")
        print(f"🎯 标准答案结构: {json.dumps(golden_standard, ensure_ascii=False, indent=2)}")

    # 使用统一的工具加载函数（支持自动从 metadata 推断工具目录）
    query_data, env, tools, tool_registry, tool_protocols, function_map = _load_tools_and_build_env_for_case(
        query_data,
        use_tools,
        load_all_topic_tools,
        test_type,
        auto_infer_from_metadata=auto_infer_from_metadata,
    )

    # 处理可能包含图片的问题（统一通过独立模块构造基础用户消息）
    question_data, user_message = _build_basic_user_message(query_data, test_type)

    # 校验图片加载情况：如果题目需要图片但存在未加载的图片，直接中止
    expected_images = question_data.get('expected_image_count') or 0
    missing_images = question_data.get('missing_images') or []
    if expected_images > 0 and missing_images:
        reason = "图片加载失败"
        extra = {
            "expected_images": expected_images,
            "missing_images": missing_images,
        }
        _record_skip_event(trace_path, case_id, current_model, mode_desc, reason, extra)
        details = {"case_id": case_id}
        details.update(extra)
        raise TestSkipException(reason, details)
    
    # 创建消息 - 在基础用户问题上追加格式要求
    user_content_base = question_data['text']  # 使用清理后的文本

    # 如果使用工具，移除原问题中的旧格式要求部分，仅保留 Answer 模板
    if use_tools:
        # 查找格式要求的起始位置
        format_start = user_content_base.find("You should strictly respond in this exact format")
        if format_start != -1:
            # 提取格式要求之前的问题部分
            question_part = user_content_base[:format_start].strip()

            # 构建简化的格式要求（只保留Answer部分）
            simplified_format = _get_tool_mode_answer_prompt()

            # 组合新的用户内容
            user_content = question_part + "\n" + simplified_format

            print("✂️ 已移除推理过程格式要求")
        else:
            # 构建简化的格式要求（只保留Answer部分）
            simplified_format = _get_tool_mode_answer_prompt()
            user_content = user_content_base + "\n" + simplified_format
    else:
        # 纯文本模式：保留推理过程 + 答案
        simplified_format = _get_text_mode_reasoning_answer_prompt()
        user_content = user_content_base + "\n" + simplified_format

    print(f"📝 {'工具模式' if use_tools else '文本模式'}问题内容:")
    print(f"原始问题长度: {len(user_content_base)}")
    print(f"处理后长度: {len(user_content)}")

    # 检查 user_content 是否为空
    if not user_content or not user_content.strip():
        print("❌ 错误：用户内容为空，无法构造消息")
        return None

    # 在基础消息上更新文本部分，而不是重新构造整个多模态结构
    if isinstance(user_message.get("content"), list):
        # 多模态：找到最后一个 text 类型片段并替换为带格式的文本
        replaced = False
        for part in reversed(user_message["content"]):
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = user_content
                replaced = True
                break
        if not replaced:
            user_message["content"].append({"type": "text", "text": user_content})
    else:
        # 纯文本：直接替换为带格式的文本
        user_message["content"] = user_content

    messages = []
    if use_tools and use_react_prompt:
        messages.append({"role": "system", "content": _get_react_tool_system_prompt()})
    messages.append(user_message)
    print(user_content) 

    # 如果不使用工具，直接进行单轮对话
    if not use_tools:
        print(f"\n--- 纯文本模式对话 ---")

        # 调用模型（不传入tools参数）
        response = test_client.chat_completions_create(
            messages=messages
        )

        assistant_message = _extract_choice_message(response, "纯文本对话")
        # 确保获取完整的内容，避免截断
        final_content = getattr(assistant_message, 'content', None)
        if final_content is None:
            final_content = "模型回答为空"
        else:
            final_content = _content_to_text(final_content)

        print(f"助手回复: {final_content}")
        print(f"\n=== 纯文本对话结束 ===")

        # 将简单的消息轨迹保存
        messages.append({"role": "assistant", "content": final_content})
        round_count = 1

    else:
        # 原有的工具调用逻辑
        # 循环调用机制：每次只处理一个工具调用
        round_count = 0
        max_rounds = 50  # 防止无限循环

        while round_count < max_rounds:
            round_count += 1
            print(f"\n--- 第 {round_count} 轮API调用 ---")

            # 第一轮强制使用工具，后续轮次让模型自由选择
            tool_choice = "auto"

            # 调用模型
            response = test_client.chat_completions_create(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False  # 关闭并行工具调用，确保一次只返回一个工具调用
            )

            assistant_message = _extract_choice_message(response, f"第 {round_count} 轮 API 调用")
            
            # 确保助手消息内容不为空，这对于轨迹保存很重要
            # 但是要保留原始内容，避免截断真实的模型回答
            original_content = getattr(assistant_message, 'content', None)
            if not original_content:
                # 只有在真正为空时才设置默认内容
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                    assistant_message.content = "我将使用工具来解决这个问题。"
                else:
                    assistant_message.content = "没有回复"
            else:
                # 保留原始多模态结构，但日志打印时转为文本
                assistant_message.content = original_content
            
            messages.append(assistant_message)

            # 调试信息
            print(f"助手回复: {_content_to_text(assistant_message.content)}")

            # 检查是否有工具调用
            tool_calls = getattr(assistant_message, 'tool_calls', None)
            
            # 对于 GLM 模型，如果使用 ReAct prompt 但返回的是文本格式的工具调用，需要解析
            # 即使 tool_calls 为 None，也要检查是否是 GLM 的文本格式
            if not tool_calls and use_tools:
                content_text = _content_to_text(assistant_message.content)
                # 获取 provider（从 test_client 或 current_model 配置中）
                provider = _detect_provider(current_model)
                
                # 检查内容中是否包含 Action: 模式（GLM 文本格式工具调用的特征）
                if provider in ("zhipuai", "glm") or ("Action:" in content_text and "<arg_key>" in content_text):
                    print(f"🔍 检测到可能的 GLM 文本格式工具调用 (provider: {provider}, use_react_prompt: {use_react_prompt})")
                    # 检查是否是 GLM 的文本格式工具调用
                    parsed_tool_calls = _parse_glm_text_tool_calls(content_text, provider)
                    if parsed_tool_calls:
                        print(f"✅ 成功解析 GLM 文本格式工具调用，已解析为 {len(parsed_tool_calls)} 个工具调用")
                        # 将解析的工具调用转换为标准格式
                        tool_calls = parsed_tool_calls
                    else:
                        print(f"⚠️  未解析到工具调用（可能格式不匹配）")
            
            if not tool_calls:
                print("没有工具调用，对话结束")
                print(f"最终回复: {assistant_message.content}")
                break

            # 处理所有工具调用
            print(f"工具调用数量: {len(tool_calls)}")

            for i, tool_call in enumerate(tool_calls):
                print(f"--- 执行工具 {i+1}/{len(tool_calls)}: {tool_call.function.name} ---")

                try:
                    # 添加调试信息，显示原始参数字符串
                    raw_arguments = tool_call.function.arguments
                    print(f"原始参数字符串: {repr(raw_arguments)}")
                    
                    # 尝试解析JSON参数
                    arguments = json.loads(raw_arguments)
                    print(f"参数: {arguments}")

                    if tool_call.function.name in (tool_registry or {}):
                        try:
                            # 通过 MinimalSciEnv + run_tool_call 执行工具
                            tool_result = run_tool_call(env, tool_call.function.name, arguments, tool_call.id)
                            print(f"工具执行结果(raw): {tool_result}")
                            result = tool_result.get("result")

                            # Issue #6: 递归扫描 result + arguments 里所有"看起来像刚生成的文件"
                            # 分类：image / database / data / report。
                            # 图像 → 全部 base64 嵌入；其它 → 只暴露路径元信息不 base64。
                            try:
                                artifacts = _detect_generated_artifacts(result, arguments)
                            except Exception as e_scan:
                                print(f"⚠️ 扫描工具产物失败: {e_scan}")
                                artifacts = []

                            image_artifacts = [a for a in artifacts if a["category"] == "image"]
                            other_artifacts = [a for a in artifacts if a["category"] != "image"]

                            # 非图像产物（数据库/数据/报告）：只暴露路径 + 大小元信息
                            if other_artifacts:
                                if result is None:
                                    result = {}
                                elif not isinstance(result, dict):
                                    result = {"original_result": result}
                                result["_generated_artifacts"] = [
                                    {
                                        "path": a["path"],
                                        "category": a["category"],
                                        "size_bytes": a["size_bytes"],
                                        "key_trail": a["key_trail"],
                                        "note": (
                                            f"{a['category']} artifact saved to disk; "
                                            "not embedded (only image artifacts are embedded)"
                                        ),
                                    }
                                    for a in other_artifacts
                                ]

                            # 图像产物：全部读入 base64，准备构造多模态 message
                            if image_artifacts:
                                if result is None:
                                    result = {}
                                elif not isinstance(result, dict):
                                    result = {"original_result": result}
                                result["_embedded_images"] = []
                                for a in image_artifacts:
                                    try:
                                        b64 = _encode_file_to_base64(a["path"])
                                        mime = _mime_from_ext(a["path"])
                                        result["_embedded_images"].append({
                                            "path": a["path"],
                                            "mime_type": mime,
                                            "size_bytes": a["size_bytes"],
                                            "key_trail": a["key_trail"],
                                            "base64": b64,
                                        })
                                        _refresh_preview_image(a["path"])
                                    except OSError as e_read:
                                        print(f"⚠️ 读取图片失败 {a['path']}: {e_read}")

                            embedded_images = result.get("_embedded_images", []) if isinstance(result, dict) else []
                            if embedded_images and not _is_text_only_mode():
                                # 多模态分支：一个 text part + 每张图一个 image part
                                text_result = {
                                    k: v for k, v in result.items()
                                    if not k.startswith("_embedded_")
                                }
                                text_content = json.dumps(text_result, default=str, ensure_ascii=False)
                                if not text_content or text_content.strip() in ("{}", "null"):
                                    text_content = f"工具执行完成，生成了 {len(embedded_images)} 张图像"
                                else:
                                    text_content = f"工具执行结果: {text_content}"

                                content_parts: List[Dict[str, Any]] = [
                                    {"type": "text", "text": text_content}
                                ]
                                provider = getattr(test_client, "provider", "openai")
                                for img in embedded_images:
                                    content_parts.append(
                                        _build_image_content_part(img["base64"], img["mime_type"], provider)
                                    )
                                    print(
                                        f"🖼️ 工具生成图片: {Path(img['path']).name}"
                                        f" (类型: {img['mime_type']}, 从字段: {img['key_trail']})"
                                    )

                                function_message = {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": content_parts,
                                }
                            else:
                                # 纯文本回退（未侦测到图 / text-only 模式）
                                # text-only: 剥离 _embedded_images 里的 base64 巨字段，只保留路径元信息
                                if isinstance(result, dict) and result.get("_embedded_images"):
                                    slim_images = [
                                        {k: v for k, v in img.items() if k != "base64"}
                                        for img in result["_embedded_images"]
                                    ]
                                    result["_embedded_images"] = slim_images
                                stripped_result = _strip_embedded_file_fields(result) if _is_text_only_mode() else result
                                result_content = json.dumps(stripped_result, default=str, ensure_ascii=False)
                                if not result_content or result_content.strip() in ("", "{}", "null"):
                                    result_content = json.dumps(
                                        {"result": "工具执行完成，无返回内容"}, ensure_ascii=False
                                    )

                                function_message = {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result_content,
                                }
                            
                            print(function_message)
                            messages.append(function_message)

                        except Exception as e:
                            print(f"工具执行错误: {e}")
                            error_content = json.dumps({"error": str(e)}, default=str, ensure_ascii=False)
                            if not error_content or error_content.strip() in ['', '{}', 'null']:
                                error_content = json.dumps({"error": "工具执行发生未知错误"}, ensure_ascii=False)
                                
                            function_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_content
                            }
                            messages.append(function_message)
                    else:
                        print(f"未知工具: {tool_call.function.name}")
                        print(f"可用工具: {list((tool_registry or {}).keys())}")
                        error_content = json.dumps({"error": f"未知工具: {tool_call.function.name}"}, default=str, ensure_ascii=False)
                        if not error_content or error_content.strip() in ['', '{}', 'null']:
                            error_content = json.dumps({"error": "工具不存在"}, ensure_ascii=False)
                            
                        function_message = {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_content
                        }
                        messages.append(function_message)

                except Exception as e:
                    print(f"解析工具调用参数失败: {e}")
                    error_content = json.dumps({"error": f"参数解析失败: {str(e)}"}, default=str, ensure_ascii=False)
                    if not error_content or error_content.strip() in ['', '{}', 'null']:
                        error_content = json.dumps({"error": "参数解析失败"}, ensure_ascii=False)
                        
                    function_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_content
                    }
                    messages.append(function_message)

        if round_count >= max_rounds:
            print(f"\n⚠️ 达到最大轮数限制 ({max_rounds})，强制结束对话")
            # 确保获取完整的内容，即使达到轮数限制
            base_content = getattr(assistant_message, 'content', '') or ''
            final_content = base_content + "对话轮数超过限制，自动结束"
        else:
            # 确保获取完整的助手回答内容
            final_content = getattr(assistant_message, 'content', None)
            if final_content is None:
                final_content = "模型回答为空"
            else:
                final_content = _content_to_text(final_content)

        print(f"\n=== 对话结束，共进行了 {round_count} 轮 ===")

    # 在保存前尝试"收敛到严格 JSON"：若期望结构存在但当前无法抽取，则追加一轮只输出JSON的提示
    if answer_template:
        try:
            # parsed_once = extract_structured_answer_from_response(final_content)
            parsed_once = None
        except Exception:
            parsed_once = None
        if not parsed_once:
            finalize_prompt = (
                "现在请仅输出严格 JSON，必须完全匹配以下结构（键名、层级、字段齐全），"
                "所有数值用数字表示（保留2-3位小数），布尔用 true/false，字符串填写说明文字。"
                "不要输出任何解释文字或前后缀，也不要多余字段；若无法计算某值请给出可解析的近似值。\n\n"
                f"{json.dumps(answer_template, ensure_ascii=False, indent=2)}"
            )
            print("⚙️ 尝试进行最终收敛提示，要求模型仅输出 JSON ...")
            # 追加用户指令，不再传 tools
            messages.append({"role": "user", "content": finalize_prompt})
            response = test_client.chat_completions_create(messages=messages)
            final_msg = _extract_choice_message(response, "最终收敛提示")
            # 记录收敛轮的回复
            messages.append({"role": "assistant", "content": getattr(final_msg, 'content', None)})
            # 更新最终文本
            final_content = _content_to_text(getattr(final_msg, 'content', None)) or "模型回答为空"
            # 再尝试解析
            try:
                parsed_once = extract_structured_answer_from_response(final_content)
            except Exception:
                parsed_once = None
            if parsed_once:
                print("✅ 收敛成功，已获得可解析的结构化 JSON。")
            else:
                print("⚠️ 收敛后仍未能抽取到结构化 JSON。")

    # 在每次测试结束后，把整个消息轨迹保存为 JSON 文件，便于后续评分
    try:
        # 从 metadata 中获取数据集文件名（如果存在）
        dataset_filename = metadata.get('_dataset_filename') if isinstance(metadata, dict) else None
        
        trace_path, sanitized_tag = _derive_trace_path(
            current_model,
            use_tools,
            query_data['id'],
            trace_tag,
            mode_name=effective_mode_name,
            metadata=metadata,
            dataset_filename=dataset_filename,
        )
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        # 将 messages 中的对象序列化：若已是 dict/list/str，直接写入；否则转换为字符串
        serializable_messages = []
        for m in messages:
            mm = {}
            # 支持两种消息形式：dict（我们自己创建的）或带属性的对象（例如 SDK 返回的消息对象）
            if isinstance(m, dict):
                role = m.get('role')
                content = m.get('content')
                tool_call_id = m.get('tool_call_id') if 'tool_call_id' in m else None
                tool_calls = m.get('tool_calls')
            else:
                # 安全地尝试从对象读取常用属性
                role = getattr(m, 'role', None)
                # SDK 的 message.content 可能是字符串或对象
                content = getattr(m, 'content', None)
                tool_call_id = getattr(m, 'tool_call_id', None)
                tool_calls = getattr(m, 'tool_calls', None)

            mm['role'] = role

            # 处理content
            if isinstance(content, (str, dict, list)):
                mm['content'] = content
            elif content is None:
                mm['content'] = None
            else:
                # 尝试获取 .content 属性的字符串（如果是 SDK 对象）
                try:
                    mm['content'] = str(content)
                except Exception:
                    mm['content'] = None

            # 处理tool_calls（对于assistant消息）
            if tool_calls is not None:
                if isinstance(tool_calls, list):
                    serialized_tool_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            serialized_tool_calls.append(tc)
                        else:
                            # 从SDK对象提取tool_call信息
                            tc_dict = {
                                'id': getattr(tc, 'id', None),
                                'type': getattr(tc, 'type', 'function'),
                                'function': {
                                    'name': getattr(tc.function, 'name', None) if hasattr(tc, 'function') else None,
                                    'arguments': getattr(tc.function, 'arguments', None) if hasattr(tc, 'function') else None
                                }
                            }
                            serialized_tool_calls.append(tc_dict)
                    mm['tool_calls'] = serialized_tool_calls

            # 对 tool 消消息，content 通常是 JSON 字符串（我们添加的），尝试解析
            if role == 'tool' and isinstance(content, str):
                try:
                    mm['content'] = json.loads(content)
                except Exception:
                    mm['content'] = content

            if tool_call_id is not None:
                mm['tool_call_id'] = tool_call_id

            serializable_messages.append(mm)

        metadata_summary = ensure_metadata_summary(query_data)
        metadata_full = query_data.get('metadata') if isinstance(query_data.get('metadata'), dict) else None
        serialized_query_payload = None
        payload_source = query_data
        if use_tools and tool_protocols:
            try:
                payload_source = deepcopy(query_data)
                payload_source['usage_tool_protocol'] = tool_protocols
            except Exception:
                payload_source = query_data
        try:
            # 深拷贝测试案例，确保写入的JSON可序列化
            serialized_query_payload = json.loads(json.dumps(payload_source, ensure_ascii=False, default=str))
        except Exception:
            # 如果序列化失败，退化为只保留基础字段
            serialized_query_payload = {
                'id': query_data.get('id'),
                'question': query_data.get('question'),
                'answer': query_data.get('answer'),
            }

        with open(trace_path, 'w', encoding='utf-8') as f:
            trace_data = {
                'id': query_data['id'],
                'query': query_data['question'],
                'query_data': serialized_query_payload,
                'model': current_model,
                'use_tools': use_tools,  # 新增：标记是否使用工具
                'mode': "tool_mode" if use_tools else "text_mode",  # 新增：测试模式
                'rounds': round_count if use_tools else 1,  # 添加轮数信息
                'messages': serializable_messages
            }

            if metadata_full:
                trace_data['metadata'] = metadata_full
            if metadata_summary:
                trace_data['metadata_summary'] = metadata_summary

            # 添加规范答案信息用于评分
            if answer_template and golden_standard:
                trace_data['answer_template'] = answer_template
                trace_data['golden_standard'] = golden_standard
                trace_data['expected_format'] = "structured_json"

                # 尝试从最终回答中提取结构化数据
                try:
                    final_answer_structured = extract_structured_answer_from_response(final_content)
                    if final_answer_structured:
                        trace_data['model_structured_answer'] = final_answer_structured
                        trace_data['answer_extraction_success'] = True
                        # 无论提取是否成功，都保存原始回答文本；并额外保存最后一条 assistant 的原始多模态结构（若有）
                        trace_data['model_raw_answer'] = final_content
                        # 把最后一条 assistant 消息附加保存，便于后续精确回退
                        try:
                            last_assist_msg = None
                            for m in reversed(serializable_messages):
                                if m.get('role') == 'assistant':
                                    last_assist_msg = m
                                    break
                            if last_assist_msg is not None:
                                trace_data['model_last_assistant_message'] = last_assist_msg
                        except Exception:
                            pass

                        # 立即进行评分
                        from gym.core.evaluator import calculate_answer_score
                        score, summary, details = calculate_answer_score(final_answer_structured, golden_standard)
                        trace_data['evaluation_score'] = score
                        trace_data['evaluation_summary'] = summary
                        trace_data['evaluation_details'] = details

                        print(f"📊 自动评分结果: {summary}")
                        if score < 0.8:  # 如果分数较低，显示详细信息
                            print(f"📋 评分详情: {json.dumps(details, ensure_ascii=False, indent=2)}")
                    else:
                        trace_data['answer_extraction_success'] = False
                        trace_data['model_raw_answer'] = final_content
                        try:
                            last_assist_msg = None
                            for m in reversed(serializable_messages):
                                if m.get('role') == 'assistant':
                                    last_assist_msg = m
                                    break
                            if last_assist_msg is not None:
                                trace_data['model_last_assistant_message'] = last_assist_msg
                        except Exception:
                            pass
                        print("⚠️ 未能从回答中提取结构化数据，无法自动评分")
                except Exception as e:
                    trace_data['answer_extraction_success'] = False
                    trace_data['answer_extraction_error'] = str(e)
                    trace_data['model_raw_answer'] = final_content
                    try:
                        last_assist_msg = None
                        for m in reversed(serializable_messages):
                            if m.get('role') == 'assistant':
                                last_assist_msg = m
                                break
                        if last_assist_msg is not None:
                            trace_data['model_last_assistant_message'] = last_assist_msg
                    except Exception:
                        pass
                    print(f"❌ 答案提取错误: {e}")
            else:
                trace_data['expected_format'] = "free_text"
                trace_data['model_raw_answer'] = final_content
                try:
                    last_assist_msg = None
                    for m in reversed(serializable_messages):
                        if m.get('role') == 'assistant':
                            last_assist_msg = m
                            break
                    if last_assist_msg is not None:
                        trace_data['model_last_assistant_message'] = last_assist_msg
                except Exception:
                    pass

            trace_data['mode_folder'] = trace_path.parent.name
            if sanitized_tag:
                trace_data['trace_tag'] = sanitized_tag

            json.dump(trace_data, f, ensure_ascii=False, indent=2)
        print(f"对话轨迹已保存为: {trace_path}")
    except Exception as e_save:
        print(f"保存轨迹失败: {e_save}")

    return final_content


def debug_simple_test_query_with_first_refined_case(
    model_name: Optional[str] = None,
    use_tools: bool = True,
    dataset_key: Optional[str] = None,
) -> Optional[str]:
    """
    最小单元测试函数：
    - 从 dataset_config 中解析出对应的 JSON 文件（默认 multi）
    - 取第一个 case，调用 simple_test_query
    - 打印并返回最终回答文本

    使用方式（在项目根目录）：
        python -c "from gym.test_executor import debug_simple_test_query_with_first_refined_case as f; f('gpt-4o')"
        # 或指定 single 数据集：
        python -c "from gym.test_executor import debug_simple_test_query_with_first_refined_case as f; f('gpt-4o', dataset_key='single')"
    """
    import json
    from gym.config.dataset_config import get_dataset_entry

    entry = get_dataset_entry(dataset_key)
    dataset_path = entry.dataset_path
    if not dataset_path.exists():
        print(f"警告：数据集文件不存在: {dataset_path}")
        return None

    try:
        with dataset_path.open("r", encoding="utf-8") as f:
            cases = json.load(f)
    except Exception as e:
        print(f"LOAD ERROR  加载数据集失败: {e}")
        return None

    if not isinstance(cases, list) or not cases:
        print("TYPE ERROR  数据集内容为空或格式不是列表")
        return None

    case = cases[0]
    case_id = case.get("id", "unknown")
    print(f"📝 使用第一个案例进行 simple_test_query 调试，ID: {case_id}")

    result = simple_test_query(
        case,
        model_name=model_name,
        use_tools=use_tools,
    )
    final_answer = result.get("final_answer", "")
    print("\n=== 模型最终回答 ===")
    print(final_answer)
    print("====================\n")
    
    # 添加 evaluation：提取 boxed answer 并进行判题
    from gym.core.evaluator import extract_boxed_answer, is_answer_correct
    
    print("=== 开始评估 ===")
    
    # 提取 boxed answer
    boxed_answer = extract_boxed_answer(final_answer)
    
    # 读取标准答案
    standard_answer = case.get("answer")
    if not standard_answer:
        # 尝试从 metadata 中获取
        metadata = case.get("metadata", {})
        standard_answer = metadata.get("golden_answer") or metadata.get("answer")
    
    # 处理标准答案可能是列表或字典的情况
    standard_answer_str = None
    if standard_answer:
        if isinstance(standard_answer, list) and standard_answer:
            standard_answer_str = str(standard_answer[0])
        elif isinstance(standard_answer, dict):
            standard_answer_str = json.dumps(standard_answer, ensure_ascii=False)
        else:
            standard_answer_str = str(standard_answer)
    
    # 获取问题文本
    question_text = case.get("question", "")
    
    # 进行判题（如果有 boxed answer 和标准答案）
    is_correct = None
    if boxed_answer and standard_answer_str:
        try:
            print(f"📋 标准答案: {standard_answer_str}")
            print(f"📋 模型答案: {boxed_answer}")
            print(f"📋 问题: {question_text[:100]}..." if len(question_text) > 100 else f"📋 问题: {question_text}")
            
            is_correct = is_answer_correct(question_text, boxed_answer, standard_answer_str, case_id)
            
            if is_correct:
                print("判题结果✅ : 正确")
            else:
                print("判题结果❌ : 错误")
                
        except Exception as e:
            print(f"判题过程出错❌ : {e}")
            import traceback
            traceback.print_exc()
    else:
        if not boxed_answer:
            print("⚠️ 未能从回答中提取 boxed answer，跳过判题")
        if not standard_answer_str:
            print("⚠️ 未找到标准答案，跳过判题")
    
    print("====================\n")
    
    # 保存 trace 文件
    try:
        from gym.core.data_loader import ensure_metadata_summary
        
        # 获取 metadata 和 dataset_key
        metadata = case.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # 从数据集文件名推断 dataset_key（如果 metadata 中没有）
        if not metadata.get("dataset_key"):
            dataset_filename = dataset_path.name
            if "single" in dataset_filename.lower():
                metadata["dataset_key"] = "merged_single_questions"
            else:
                metadata["dataset_key"] = "merged_questions"
        
        # 确保 metadata 中有 _dataset_filename（用于路径判断）
        if "_dataset_filename" not in metadata:
            metadata["_dataset_filename"] = dataset_path.name
        
        # 确定 trace 路径（使用本文件内的精简版推导逻辑，避免导入 gym.test_executor）
        mode_name = "with_tools_react" if use_tools else "without_tools"
        trace_path = _derive_trace_path_for_debug(
            model_name=model_name or DEFAULT_MODEL,
            use_tools=use_tools,
            case_id=case_id,
            mode_name=mode_name,
            metadata=metadata,
            dataset_filename=dataset_path.name,
        )
        
        # 序列化 messages（简化版，因为 simple_test_query 返回的 messages 已经是 dict 格式）
        serializable_messages = []
        for m in result.get("messages", []):
            if isinstance(m, dict):
                # 已经是 dict，直接使用，但需要处理 tool 消息的 content（可能是 JSON 字符串）
                mm = m.copy()
                if m.get("role") == "tool" and isinstance(m.get("content"), str):
                    try:
                        mm["content"] = json.loads(m["content"])
                    except Exception:
                        pass
                serializable_messages.append(mm)
            else:
                # 如果是对象，转换为 dict（简化处理）
                serializable_messages.append({
                    "role": getattr(m, "role", None),
                    "content": getattr(m, "content", None),
                })
        
        # 构建 trace 数据
        metadata_summary = ensure_metadata_summary(case)
        trace_data = {
            "id": case_id,
            "query": case.get("question", ""),
            "query_data": case,  # 保存完整的 case 数据
            "model": model_name or DEFAULT_MODEL,
            "use_tools": use_tools,
            "mode": "tool_mode" if use_tools else "text_mode",
            "rounds": result.get("rounds", 1),
            "messages": serializable_messages,
            "model_raw_answer": final_answer,
        }
        
        if metadata:
            trace_data["metadata"] = metadata
        if metadata_summary:
            trace_data["metadata_summary"] = metadata_summary
        
        # 添加 evaluation 结果
        if boxed_answer is not None:
            trace_data["model_boxed_answer"] = boxed_answer
            trace_data["boxed_extraction_success"] = True
        else:
            trace_data["boxed_extraction_success"] = False
        
        if is_correct is not None:
            trace_data["boxed_answer_evaluation"] = {
                "is_correct": is_correct,
                "standard_answer": standard_answer_str,
                "model_answer": boxed_answer,
                "evaluation_method": "gpt4.1_judge",
                "evaluation_success": True,
            }
        
        # 保存 trace 文件
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Trace 文件已保存: {trace_path}")
        
    except Exception as e:
        print(f"⚠️ 保存 trace 文件失败: {e}")
        import traceback
        traceback.print_exc()
    
    return final_answer


__all__ = [
    "simple_test_query",
    "simple_test_refine_query",
    "test_query",
    "debug_simple_test_query_with_first_refined_case",
]

if __name__ == "__main__": 
    debug_simple_test_query_with_first_refined_case('glm-4.6v', use_tools=True)