"""
批量多题调试脚本（benchmark 入口）

功能：
- 对 ``dataset/refine_merged_multi_questions.json`` 或
  ``dataset/refine_merged_single_questions.json`` 中的所有案例逐一测试。
- 对每个案例：
  - 调用模型（可选使用工具）
  - 提取 boxed 答案或结构化答案
  - 使用 LLM 判题或结构化评分（对/错/分数）
  - 将完整对话轨迹和评测结果落盘到统一的 traces 目录

落盘目录结构（根据数据集是否为单模态 / 多模态自动选择）：

- 单模态：data_analysis/tracetoanalyze/tracesmerged_single_questions/<model>/<mode>/<id>_trace.json
- 多模态：data_analysis/tracetoanalyze/tracesmerged_questions/<model>/<mode>/<id>_trace.json

运行方式（在项目根目录）：

    # 跑多模态数据集全部案例（默认）
    python gym/test_querys.py

    # 跑单模态数据集
    python gym/test_querys.py --dataset single

    # 指定模型 + 关闭工具（纯 LLM baseline）
    python gym/test_querys.py --model gpt-4o --no-tools

    # 只跑指定 case
    python gym/test_querys.py --case-id 5

    # 纯文本模式（Issue #6）：不把图像回传给模型，
    # 适用于不支持 image_url 消息的 OpenAI 兼容 API。
    python gym/test_querys.py --text-only

    # single 和 multi 全跑
    python gym/test_querys.py --dataset both

参数总览：见 `python gym/test_querys.py --help` 或本文件 `_build_arg_parser()`。

注意：
- 本文件依赖 `gym.test_executor` 模块
- 支持 test_type="normal"（boxed 答案评估）和 test_type="refine"（结构化过程评估）
"""

from __future__ import annotations
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent
for _module_name in list(sys.modules.keys()):
    if _module_name == 'gym' or _module_name.startswith('gym.'):
        del sys.modules[_module_name]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
from typing import Any, Dict, List, Optional, Tuple
from gym.agent import DEFAULT_MODEL
from gym.test_executor import (
    simple_test_query,
    test_query,
    _resolve_dataset_folder,
    _resolve_mode_folder,
)
from gym.core.evaluator import extract_boxed_answer, is_answer_correct
from gym.core.data_loader import ensure_metadata_summary, normalize_image_path
from gym.core.exceptions import TestSkipException
def _derive_trace_path_for_multi(
    model_name: str,
    use_tools: bool,
    case_id: Any,
    mode_name: str,
    dataset_filename: str,
) -> Path:
    """
    多题批量测试使用的简化版 trace 路径推导逻辑。

    根据数据集文件名是否包含 "single"，决定单模态 / 多模态的父目录：

        data_analysis/tracetoanalyze/tracesmerged_single_questions
        data_analysis/tracetoanalyze/tracesmerged_questions

    其下按如下结构组织：

        [父目录] / model_name / mode_name / {case_id}_trace.json
    """
    is_single = "single" in dataset_filename.lower()

    project_root = Path(__file__).resolve().parents[1]
    traces_root = project_root / "data_analysis" / "tracetoanalyze"

    if is_single:
        base_root = traces_root / "tracesmerged_single_questions"
    else:
        base_root = traces_root / "tracesmerged_questions"

    model_dir = model_name or DEFAULT_MODEL
    base_dir = base_root / model_dir / mode_name
    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir / f"{case_id}_trace.json"

def _evaluate_refine_from_trace(
    case: Dict[str, Any],
    model_name: str,
    use_tools: bool,
    dataset_path: Path,
) -> Tuple[Optional[float], Optional[str], Optional[Path]]:
    """
    从 trace 文件中读取 refine 类型案例的评估信息。
    
    返回: (score, score_summary, trace_path)
    """
    case_id = case.get("id", "unknown")
    
    try:
        from gym.config.dataset_config import get_trace_root
        
        metadata = case.get("metadata") or {}
        dataset_folder = _resolve_dataset_folder(metadata)
        mode_folder = _resolve_mode_folder(use_tools, None)
        
        # 判断是 single 还是 multi
        dataset_filename = dataset_path.name
        if "single" in dataset_filename.lower():
            data_type_folder = "orignal_data_single"
        else:
            if "single" in dataset_folder.lower():
                data_type_folder = "orignal_data_single"
            else:
                data_type_folder = "orignal_data_multi"
        
        model_name_actual = model_name or DEFAULT_MODEL
        trace_root = get_trace_root(metadata.get("dataset_key"))
        trace_path = trace_root / model_name_actual / data_type_folder / mode_folder / f"{case_id}_trace.json"
        
        if trace_path.exists():
            with open(trace_path, "r", encoding="utf-8") as f:
                trace_data = json.load(f)
            
            # 提取评估信息
            evaluation_score = trace_data.get("evaluation_score")
            evaluation_summary = trace_data.get("evaluation_summary", "未评估")
            
            if evaluation_score is not None:
                return float(evaluation_score), evaluation_summary, trace_path
    
    except Exception as e:
        print(f"  ⚠️ 无法读取评估信息: {e}")
        import traceback
        traceback.print_exc()
    
    return None, None, None


def normalize_case_image_paths(case: Dict[str, Any]) -> None:
    """统一处理案例中的图片路径，将旧路径转换为新的统一路径格式
    
    修改 case 中的 metadata.image_path，将路径从：
    - "failed_question_images/xxx.jpg"
    - "filtered_images/xxx.jpg"
    - "/sfe_images/xxx.png"
    - "/r_bench/images/xxx.png"
    
    转换为：
    - "gym/test_images/failed_question_images/xxx.png"
    - "gym/test_images/filtered_images/xxx.png"
    - "gym/test_images/sfe_images/xxx.png"
    - "gym/test_images/r_bench/images/xxx.png"
    
    Args:
        case: 测试案例字典，会被原地修改
    """
    if not isinstance(case, dict):
        return
    
    metadata = case.get("metadata")
    if not isinstance(metadata, dict):
        return
    
    image_paths = metadata.get("image_path")
    if not image_paths:
        return
    
    # 处理单个路径字符串
    if isinstance(image_paths, str):
        normalized = normalize_image_path(image_paths)
        metadata["image_path"] = normalized
        return
    
    # 处理路径列表
    if isinstance(image_paths, (list, tuple)):
        normalized_paths = []
        for path in image_paths:
            if isinstance(path, str):
                normalized_paths.append(normalize_image_path(path))
            else:
                normalized_paths.append(path)
        metadata["image_path"] = normalized_paths


def _evaluate_and_save_trace(
    case: Dict[str, Any],
    result: Dict[str, Any],
    final_answer: str,
    model_name: str,
    use_tools: bool,
    dataset_path: Path,
    test_type: str = "normal",
) -> Tuple[Optional[bool], Optional[Path]]:
    """
    对单个案例执行：
    - 如果 test_type="normal": boxed 答案提取 + LLM 判题 + trace 落盘
    - 如果 test_type="refine": 从 trace 文件读取评估信息（test_query 已处理）

    返回：(is_correct, trace_path)
    """
    case_id = case.get("id", "unknown")

    # refine 类型：评估信息已在 test_query 中处理并保存在 trace 文件中
    if test_type == "refine":
        print("=== 读取 refine 类型评估信息 ===")
        score, score_summary, trace_path = _evaluate_refine_from_trace(
            case=case,
            model_name=model_name,
            use_tools=use_tools,
            dataset_path=dataset_path,
        )
        
        if score is not None:
            print(f"✅ 评分: {score:.2f}")
            print(f"✅ 摘要: {score_summary}")
            # 将 score 转换为 is_correct（>0.8 认为正确，可根据需要调整阈值）
            is_correct = score >= 0.8
            return is_correct, trace_path
        else:
            print("⚠️ 未能从 trace 文件读取评估信息")
            return None, trace_path

    # normal 类型：使用原有的评估逻辑
    print("=== 开始评估并保存 trace ===")

    # 1. 提取 boxed answer
    boxed_answer = extract_boxed_answer(final_answer)

    # 2. 读取标准答案
    standard_answer = case.get("answer")
    if not standard_answer:
        metadata = case.get("metadata", {})
        if isinstance(metadata, dict):
            standard_answer = (
                metadata.get("golden_answer")
                or metadata.get("answer")
            )

    standard_answer_str: Optional[str] = None
    if standard_answer:
        if isinstance(standard_answer, list) and standard_answer:
            standard_answer_str = str(standard_answer[0])
        elif isinstance(standard_answer, dict):
            standard_answer_str = json.dumps(standard_answer, ensure_ascii=False)
        else:
            standard_answer_str = str(standard_answer)

    question_text = case.get("question", "")

    # 3. 判题（如果有 boxed answer 且有标准答案）
    is_correct: Optional[bool] = None
    if boxed_answer and standard_answer_str:
        try:
            print(f"📋 标准答案: {standard_answer_str}")
            print(f"📋 模型答案: {boxed_answer}")
            if question_text:
                preview = (
                    f"{question_text[:100]}..."
                    if len(question_text) > 100
                    else question_text
                )
                print(f"📋 问题: {preview}")

            is_correct = is_answer_correct(
                question_text, boxed_answer, standard_answer_str, case_id
            )

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

    # 4. 保存 trace 文件
    trace_path: Optional[Path] = None
    try:
        # 元数据与 dataset_key
        metadata = case.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        dataset_filename = dataset_path.name

        # 推导 mode_name（与主执行器对齐：with_tools_react / without_tools）
        mode_name = "with_tools_react" if use_tools else "without_tools"

        trace_path = _derive_trace_path_for_multi(
            model_name=model_name,
            use_tools=use_tools,
            case_id=case_id,
            mode_name=mode_name,
            dataset_filename=dataset_filename,
        )

        # 序列化 messages（simple_test_query 返回的 messages 已经大部分是 dict）
        serializable_messages: List[Dict[str, Any]] = []
        for m in result.get("messages", []):
            if isinstance(m, dict):
                mm = m.copy()
                if mm.get("role") == "tool" and isinstance(mm.get("content"), str):
                    try:
                        mm["content"] = json.loads(mm["content"])
                    except Exception:
                        pass
                serializable_messages.append(mm)
            else:
                serializable_messages.append(
                    {
                        "role": getattr(m, "role", None),
                        "content": getattr(m, "content", None),
                    }
                )

        metadata_summary = ensure_metadata_summary(case)

        trace_data: Dict[str, Any] = {
            "id": case_id,
            "query": question_text,
            "query_data": case,
            "model": model_name,
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
        trace_data["boxed_extraction_success"] = boxed_answer is not None
        if boxed_answer is not None:
            trace_data["model_boxed_answer"] = boxed_answer

        if is_correct is not None:
            trace_data["boxed_answer_evaluation"] = {
                "is_correct": is_correct,
                "standard_answer": standard_answer_str,
                "model_answer": boxed_answer,
                "evaluation_method": "gpt4.1_judge",
                "evaluation_success": True,
            }

        with trace_path.open("w", encoding="utf-8") as f:
            json.dump(trace_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Trace 文件已保存: {trace_path}")

    except Exception as e:
        print(f"⚠️ 保存 trace 文件失败: {e}")
        import traceback

        traceback.print_exc()

    print("====================\n")

    return is_correct, trace_path


def run_all_refined_cases(
    model_name: Optional[str] = None,
    use_tools: bool = True,
    test_type: str = "normal",
    force_retest: bool = False,
    load_all_topic_tools: bool = False,
    auto_infer_from_metadata: bool = True,
    dataset_key: Optional[str] = None,
    dataset_path: Optional[Path] = None,
    case_ids: Optional[List[Any]] = None,
    text_only: bool = False,
) -> None:
    """
    对指定 dataset 里的所有案例逐一执行测试 + evaluation + trace 落盘。

    Args:
        model_name: 模型名称
        use_tools: 是否使用工具
        test_type: 测试类型
            - "normal": 使用 simple_test_query + extract_boxed_answer + is_answer_correct
            - "refine": 使用 test_query + calculate_answer_score（结构化评估）
        force_retest: 是否强制重新测试（忽略缓存）
        load_all_topic_tools: 是否加载相同topic的所有工具（仅对 refine 类型有效）
        auto_infer_from_metadata: 是否根据 metadata 中的 subject/topic 自动推断并加载工具目录
                                   默认为 True，会自动加载 toolkits/{subject}/{topic}/ 下的所有工具
        dataset_key: dataset 键名（"multi" / "single" / 完整键），传入时优先级高于默认
        dataset_path: 直接给定 dataset JSON 路径，优先级最高
        case_ids: 只跑这些 id（str/int 均可）。为 None 时跑全部
        text_only: True 时禁用图像回传（题面图 + 工具生成图都跳过），适合不支持
                   image_url 消息的 OpenAI 兼容 API
    """
    import os

    from gym.config.dataset_config import (
        get_dataset_entry,
        set_current_dataset_key,
    )

    # Issue #6: text-only 模式通过环境变量向 gym.test_executor 透传，
    # 让 tool-message 图像回传路径也能拿到开关。
    if text_only:
        os.environ["SCIAGENT_TEXT_ONLY"] = "1"

    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        if dataset_key:
            set_current_dataset_key(dataset_key)
    else:
        entry = get_dataset_entry(dataset_key)
        set_current_dataset_key(entry.key)
        dataset_path = entry.dataset_path

    if not dataset_path.exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return

    # 根据 test_type 选择不同的数据加载方式
    try:
        if test_type == "refine":
            # refine 类型：使用 load_refined_test_cases_from_dataset 加载精炼版案例
            from gym.core.data_loader import load_refined_test_cases_from_dataset
            cases = load_refined_test_cases_from_dataset(dataset_path=str(dataset_path))
            print(f"✅ 使用 load_refined_test_cases_from_dataset 加载了 {len(cases)} 个精炼版案例")
        else:
            # normal 类型：直接从 JSON 文件加载原始案例
            with dataset_path.open("r", encoding="utf-8") as f:
                cases = json.load(f)
            print(f"✅ 直接从 JSON 文件加载了 {len(cases)} 个案例")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if not isinstance(cases, list) or not cases:
        print("❌ 数据集内容为空或格式不是列表")
        return

    current_model = model_name or DEFAULT_MODEL
    total = len(cases)
    success = 0
    correct = 0
    evaluated = 0
    skipped = 0

    test_type_label = "refine (结构化评估)" if test_type == "refine" else "normal (boxed评估)"
    modality_label = "text-only" if text_only else "multimodal"
    print(f"\n=== 开始批量测试 {dataset_path.name} ({total} 个案例) ===")
    print(
        f"模型: {current_model} | "
        f"模式: {'with_tools_react' if use_tools else 'without_tools'} | "
        f"类型: {test_type_label} | "
        f"输入形态: {modality_label}"
    )
    if case_ids:
        wanted = {str(c) for c in case_ids}
        print(f"仅执行指定 case_id: {sorted(wanted)}")
    else:
        wanted = None

    for idx, case in enumerate(cases, start=1):
        case_id = case.get("id", f"case_{idx}")
        if wanted is not None and str(case_id) not in wanted:
            continue
        # 纯文本模式下，主动去掉 metadata.image_path，避免图像被打包进 message
        if text_only:
            metadata = case.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop("image_path", None)
        else:
            # 统一处理图片路径（仅在多模态模式下）
            normalize_case_image_paths(case)
        
        # refine 类型：显示原始 ID 和精炼索引
        if test_type == "refine":
            original_id = case.get("original_id", "unknown")
            refined_index = case.get("refined_index", "unknown")
            print(f"\n--- [{idx}/{total}] 测试精炼版案例 ID: {case_id} (原始: {original_id}, 精炼索引: {refined_index}) ---")
        else:
            print(f"\n--- [{idx}/{total}] 测试案例 ID: {case_id} ---") 

        try:
            # 根据 test_type 选择不同的测试函数
            if test_type == "refine":
                # refine 类型：使用 test_query（内部已处理结构化评估）
                # 确保 metadata 中有 _dataset_filename（用于路径判断）
                if isinstance(case.get("metadata"), dict):
                    case["metadata"]["_dataset_filename"] = dataset_path.name
                else:
                    case["metadata"] = {"_dataset_filename": dataset_path.name}
                
                final_answer = test_query(
                    case,
                    model_name=current_model,
                    use_tools=use_tools,
                    force_retest=force_retest,
                    load_all_topic_tools=load_all_topic_tools,
                    auto_infer_from_metadata=auto_infer_from_metadata,
                )
                
                # test_query 返回的是字符串，评估信息已在 trace 文件中
                # 创建一个结果字典以便统一处理
                result = {"final_answer": final_answer}
                
            else:
                # normal 类型：使用 simple_test_query
                result = simple_test_query(
                    case,
                    model_name=current_model,
                    use_tools=use_tools,
                    auto_infer_from_metadata=auto_infer_from_metadata,
                )
                final_answer = result.get("final_answer", "")
            
            success += 1

            # 评估和保存 trace
            is_correct, _ = _evaluate_and_save_trace(
                case=case,
                result=result,
                final_answer=final_answer,
                model_name=current_model,
                use_tools=use_tools,
                dataset_path=dataset_path,
                test_type=test_type,
            )

            if is_correct is not None:
                evaluated += 1
                if is_correct:
                    correct += 1

        except TestSkipException as skip_exc:
            print(f"⏭️ 案例 {case_id} 被跳过，原因：{skip_exc.reason}")
            if skip_exc.details:
                print(f"   详情: {json.dumps(skip_exc.details, ensure_ascii=False)}")
            skipped += 1
            continue
        except Exception as e:
            print(f"❌ 案例 {case_id} 测试过程出错: {e}")
            import traceback

            traceback.print_exc()
            continue 
     

    print("\n=== 批量测试完成 ===")
    print(f"总案例数      : {total}")
    print(f"成功执行      : {success}")
    print(f"跳过案例      : {skipped}")
    print(f"已参与判题案例: {evaluated}")
    print(f"判定为正确    : {correct}")
    if evaluated:
        acc = correct / evaluated * 100.0
        print(f"判题准确率    : {acc:.2f}%")
    else:
        print("判题准确率    : 无有效评测结果")


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python gym/test_querys.py",
        description="SciAgentGYM benchmark 入口：批量跑 refined dataset 里的所有案例。",
    )
    p.add_argument(
        "--dataset",
        default="multi",
        choices=["multi", "single", "both",
                 "refine_merged_multi_questions", "refine_merged_single_questions"],
        help="要跑的数据集：multi=多模态，single=单模态，both=两个都跑。默认 multi。",
    )
    p.add_argument(
        "--model",
        default=None,
        help="模型名（见 gym/config/config.py 里的 SUPPORTED_MODELS）。留空则使用 DEFAULT_MODEL。",
    )
    p.add_argument(
        "--no-tools",
        action="store_true",
        help="关闭工具（纯 LLM baseline）。默认开启工具。",
    )
    p.add_argument(
        "--test-type",
        default="normal",
        choices=["normal", "refine"],
        help="normal=boxed 答案评估；refine=结构化过程评估。默认 normal。",
    )
    p.add_argument(
        "--force-retest",
        action="store_true",
        help="忽略已存在的 trace 文件，强制重跑。",
    )
    p.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="只跑指定 id 的 case，可多次传（如 --case-id 5 --case-id 12）。",
    )
    p.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "纯文本模式（Issue #6）：不把 metadata.image_path 或工具生成的 base64 图像"
            "回传给模型，适合不支持 image_url 消息的 OpenAI 兼容 API / 纯文本模型。"
            "也可通过环境变量 SCIAGENT_TEXT_ONLY=1 开启。"
        ),
    )
    p.add_argument(
        "--load-all-topic-tools",
        action="store_true",
        help="除本 case 的工具外，同时加载同 topic 下的所有工具（仅 refine 类型有效）。",
    )
    p.add_argument(
        "--no-auto-infer",
        action="store_true",
        help="关闭「从 metadata.subject/topic 自动推断工具目录」的行为。",
    )
    return p


def _resolve_dataset_keys(spec: str) -> List[str]:
    if spec == "both":
        return ["refine_merged_multi_questions", "refine_merged_single_questions"]
    if spec == "multi":
        return ["refine_merged_multi_questions"]
    if spec == "single":
        return ["refine_merged_single_questions"]
    return [spec]


if __name__ == "__main__":
    import os

    args = _build_arg_parser().parse_args()

    text_only = args.text_only or os.environ.get("SCIAGENT_TEXT_ONLY", "").strip() in {"1", "true", "TRUE", "yes"}

    for dataset_key in _resolve_dataset_keys(args.dataset):
        print(f"\n########## 数据集: {dataset_key} ##########")
        run_all_refined_cases(
            model_name=args.model,
            use_tools=not args.no_tools,
            test_type=args.test_type,
            force_retest=args.force_retest,
            load_all_topic_tools=args.load_all_topic_tools,
            auto_infer_from_metadata=not args.no_auto_infer,
            dataset_key=dataset_key,
            case_ids=args.case_id,
            text_only=text_only,
        )


