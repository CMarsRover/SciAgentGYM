"""
核心评分算法模块

这个模块包含：
- 答案提取：extract_boxed_answer 从模型回答中抽取 \\boxed{} 内的最终答案
- 答案判断：is_answer_correct 基于 LLM 的对/错判题接口
- 评分算法：calculate_answer_score 等核心评分函数

实现高内聚低耦合的设计原则。
"""
import json
import math
import re
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Any, Iterable, List, Optional

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gym.config.dataset_config import get_trace_root


# ========== 答案提取与判断 ==========

def _extract_balanced_braces(text: str, start_pos: int) -> Optional[str]:
    """
    从给定位置开始，提取与第一个 '{' 匹配的成对大括号中的内容。
    使用手动栈匹配算法，支持任意嵌套级别的大括号。
    """
    if not text or start_pos < 0 or start_pos >= len(text):
        return None

    if text[start_pos] != "{":
        return None

    stack = 0
    result_chars = []

    for idx in range(start_pos, len(text)):
        ch = text[idx]
        if ch == "{":
            stack += 1
            # 第一个左括号本身不计入结果，只记录内部内容
            if stack > 1:
                result_chars.append(ch)
        elif ch == "}":
            stack -= 1
            if stack == 0:
                # 匹配结束
                return "".join(result_chars)
            elif stack < 0:
                # 不正常的括号结构
                return None
            else:
                result_chars.append(ch)
        else:
            if stack >= 1:
                result_chars.append(ch)

    # 没有正常闭合
    return None


def extract_boxed_answer(response_content: Optional[str]) -> Optional[str]:
    """从模型回答中提取 \\boxed{} 中的内容。

    使用手动括号匹配算法，支持任意嵌套级别的大括号。

    Args:
        response_content: 模型的回答文本

    Returns:
        提取的 boxed 中的内容，如果没有找到则返回 None
    """
    if not response_content or not isinstance(response_content, str):
        return None

    try:
        # 查找所有可能的 boxed 开始位置
        boxed_patterns = [
            r"\\boxed\{",
            r"\$\\boxed\{",
            r"boxed\{",
        ]

        all_matches = []
        for pattern in boxed_patterns:
            for match in re.finditer(pattern, response_content):
                # match.end() 指向 '{' 后的第一个字符，因此减 1 回到 '{'
                start_pos = match.end() - 1
                content = _extract_balanced_braces(response_content, start_pos)
                if content is not None:
                    all_matches.append(content)

        if all_matches:
            # 返回最后一个匹配（通常是最终答案）
            answer = all_matches[-1].strip()
            print(f"🎯 提取到boxed答案: {answer}")
            return answer

        print("⚠️ 未找到boxed格式的答案")
        return None

    except Exception as e:
        print(f"提取boxed答案失败: {e}")
        return None


def is_answer_correct(question_text, model_answer, standard_answer, unique_id):
    """判断模型的答案是否正确

    评判要求：
    1. 忽略答案中的格式差异。
    2. 忽略无关的前缀或文字修饰。
    3. 如果标准答案是一个表达式或数值，只要模型答案在逻辑、计算或数值上等价，也视为正确。
    4. 若模型答案与标准答案**核心内容一致**，则判断为"正确"。
    """
    prompt = f"""这是一个问题、一个标准答案，以及一个由AI模型生成的答案。请你判断AI模型的答案是否正确。
    评判要求：
    1. 忽略答案中的格式差异。
    2. 忽略无关的前缀或文字修饰。
    3. 如果标准答案是一个表达式或数值，只要模型答案在逻辑、计算或数值上等价，也视为正确。
    4. 若模型答案与标准答案**核心内容一致**，则判断为"正确"。

    请只输出：`正确` 或 `错误`。

    ---
    问题：
    {question_text}
    ---
    标准答案：
    {standard_answer}
    ---
    AI模型的答案：
    {model_answer}
    ---
    """
    
    try:
        # 使用本项目 gym/config/config.py 中定义的判题模型
        from gym.config.config import JUDGE_MODEL
        from gym.agent import get_client

        evaluation_model = JUDGE_MODEL  # 通常为 "gpt-4.1"

        client = get_client(evaluation_model)
        response = client.chat_completions_create(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
        )

        result = response.choices[0].message.content.strip()
        is_correct = result == "正确"
        
        print(f"  判断结果: {result} ({'✅' if is_correct else '❌'})")
        return is_correct
        
    except Exception as e:
        print(f"  ❌ 答案判断失败: {e}")
        raise


# ========== 核心评分算法 ==========


def _count_fields_recursive(obj, path=""):
    """递归计算金标准中的叶子字段数量"""
    if isinstance(obj, dict):
        count = 0
        for key, value in obj.items():
            key_path = f"{path}.{key}" if path else key
            if isinstance(value, (dict, list)):
                count += _count_fields_recursive(value, key_path)
            else:
                count += 1
        return count
    if isinstance(obj, list):
        count = 0
        for i, item in enumerate(obj):
            item_path = f"{path}[{i}]" if path else f"[{i}]"
            if isinstance(item, (dict, list)):
                count += _count_fields_recursive(item, item_path)
            else:
                count += 1
        return count
    return 1


def _iter_expected_leaf_paths(expected, path=""):
    """按顺序遍历金标准中的叶子字段路径"""
    if isinstance(expected, dict):
        for key, value in expected.items():
            key_path = f"{path}.{key}" if path else key
            yield from _iter_expected_leaf_paths(value, key_path)
    elif isinstance(expected, list):
        for idx, item in enumerate(expected):
            key_path = f"{path}[{idx}]" if path else f"[{idx}]"
            yield from _iter_expected_leaf_paths(item, key_path)
    else:
        yield path


def _count_matches_from_details(details: Dict[str, Dict]) -> int:
    """从评测细节中统计匹配数量"""
    return sum(1 for detail in details.values() if detail.get("status") == "match")


def build_match_sequence(golden_standard: Dict, field_details: Dict[str, Dict]) -> List[bool]:
    """
    构建与金标准顺序一致的匹配布尔序列
    """
    sequence: List[bool] = []
    for path in _iter_expected_leaf_paths(golden_standard):
        detail = field_details.get(path)
        sequence.append(bool(detail and detail.get("status") == "match"))
    return sequence


def compute_segment_scores_from_details(
    golden_standard: Dict,
    field_details: Dict[str, Dict],
    segments: Iterable[int] = (25, 50, 75, 100),
) -> Dict[str, float]:
    """
    根据匹配序列计算不同覆盖范围的分数（例如前25%、50%、75%、100%）
    """
    match_sequence = build_match_sequence(golden_standard, field_details)
    total = len(match_sequence)
    if total == 0:
        return {}

    segment_scores: Dict[str, float] = {}
    for segment in segments:
        if segment <= 0:
            continue
        coverage_count = min(total, max(1, math.ceil(total * (segment / 100))))
        covered_matches = sum(match_sequence[:coverage_count])
        segment_scores[f"0_{segment}"] = covered_matches / coverage_count

    return segment_scores


def is_answer_correct_with_llm(question_text, model_answer, standard_answer, unique_id):
    """使用LLM判断模型答案是否正确"""
    prompt = f"""这是一个问题、一个标准答案，以及一个由AI模型生成的答案。请你判断AI模型的答案是否正确。
    请严格根据标准答案进行评判。如果AI的答案在逻辑、计算、概念上与标准答案一致，即便措辞不同，也应判断为正确。
    请只回答 '正确' 或 '错误'。

    ---
    问题：
    {question_text}
    ---
    标准答案：
    {standard_answer}
    ---
    AI模型的答案：
    {model_answer}
    ---
    """

    try:
        # 使用配置文件中定义的模型进行评判
        from gym.config.config import JUDGE_MODEL
        from gym.utils.client_manager import get_client

        # 使用配置文件中的默认模型
        evaluation_model = JUDGE_MODEL

        payload = {
            "model": evaluation_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 5
        }

        client = get_client(evaluation_model)

        response = client.chat_completions_create(
            messages=payload["messages"],
            max_tokens=payload["max_tokens"]
        )

        result = response.choices[0].message.content.strip()
        is_correct = result == "正确"

        print(f"  LLM判断结果: {result} ({'✅' if is_correct else '❌'})")
        return is_correct

    except Exception as e:
        print(f"  ❌ LLM答案判断失败: {e}")
        return False


def evaluate_pass_at_k_results(
    model_name: Optional[str] = None,
    use_tools: Optional[Any] = None,
    k: Optional[int] = None,
    trace_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """评估 pass@k 汇总文件，统计最佳结果准确率。"""

    from gym.config.config import PASS_AT_K

    if k is None:
        k = PASS_AT_K

    if trace_dir is not None:
        pass_root = Path(trace_dir)
    else:
        base_root = get_trace_root()
        pass_root = base_root / f"pass@{k}"
        if model_name:
            pass_root = pass_root / model_name
            if use_tools == 'with_all_tools':
                pass_root = pass_root / 'with_all_tools'
            elif use_tools is True:
                pass_root = pass_root / 'with_tools'
            elif use_tools is False:
                pass_root = pass_root / 'without_tools'

    if not pass_root.exists():
        print(f"❌ pass@{k} 目录不存在: {pass_root}")
        return {}

    summary_files: List[Path] = []
    for root, _, files in os.walk(pass_root):
        for name in files:
            if name.endswith('_trace.json'):
                summary_files.append(Path(root) / name)

    if not summary_files:
        print(f"⚠️ pass@{k} 未找到汇总文件")
        return {}

    total_cases = 0
    evaluated_cases = 0
    correct_cases = 0

    for summary_path in summary_files:
        try:
            with summary_path.open('r', encoding='utf-8') as f:
                summary_data = json.load(f)
        except Exception as exc:
            print(f"⚠️ 无法读取 {summary_path}: {exc}")
            continue

        total_cases += 1
        best_is_correct = summary_data.get('best_is_correct')
        if best_is_correct is None:
            continue
        evaluated_cases += 1
        if best_is_correct:
            correct_cases += 1

    accuracy = (correct_cases / evaluated_cases) if evaluated_cases else 0.0

    print("\n" + "=" * 60)
    print(f"📊 pass@{k} 汇总评估")
    print(f"   路径: {pass_root}")
    print(f"   总文件: {len(summary_files)}")
    print(f"   统计案例: {total_cases}")
    print(f"   已评估: {evaluated_cases}")
    print(f"   正确: {correct_cases}")
    print(f"   准确率: {accuracy:.2%}" if evaluated_cases else "   准确率: 无评估结果")
    print("=" * 60)

    return {
        'pass_at_k': k,
        'path': str(pass_root),
        'total_cases': total_cases,
        'evaluated_cases': evaluated_cases,
        'correct_cases': correct_cases,
        'accuracy': accuracy,
    }


def template_match_with_llm(actual_value, expected_value, path=""):
    """
    对于长度超过7个字符的字符串，使用LLM进行模版匹配

    Args:
        actual_value: 实际值
        expected_value: 期望值
        path: 字段路径

    Returns:
        bool: 是否匹配
    """
    # 只对字符串进行LLM匹配
    if not (isinstance(actual_value, str) and isinstance(expected_value, str)):
        return None

    # 只对长度超过7个字符的字符串进行LLM匹配
    if len(actual_value) <= 7 and len(expected_value) <= 7:
        return None

    prompt = f"""请判断以下两个答案是否在含义上等价或匹配。
    即使表达方式不同，只要核心含义、数值、逻辑关系一致就应该判断为匹配。
    请只回答 '匹配' 或 '不匹配'。

    ---
    标准答案：
    {expected_value}
    ---
    模型答案：
    {actual_value}
    ---
    """

    try:
        from gym.config.config import JUDGE_MODEL
        from gym.utils.client_manager import get_client

        evaluation_model = JUDGE_MODEL

        payload = {
            "model": evaluation_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 5
        }

        client = get_client(evaluation_model)

        response = client.chat_completions_create(
            messages=payload["messages"],
            max_tokens=payload["max_tokens"]
        )

        result = response.choices[0].message.content.strip()
        is_match = result == "匹配"

        print(f"  LLM模版匹配 [{path}]: {result} ({'✅' if is_match else '❌'})")
        return is_match

    except Exception as e:
        print(f"  ❌ LLM模版匹配失败 [{path}]: {e}")
        return None


def secondary_verification_with_llm(actual_value, expected_value, path=""):
    """
    二次验证：当人工匹配结果为未匹配时，使用LLM进行最终验证
    支持所有类型的值（数值、布尔、字符串等）

    Args:
        actual_value: 实际值
        expected_value: 期望值
        path: 字段路径

    Returns:
        bool: 是否匹配，None表示LLM验证失败
    """
    # 将值转换为字符串用于LLM比较
    actual_str = str(actual_value)
    expected_str = str(expected_value)
    

    prompt = f"""请判断以下两个答案是否在含义上等价或匹配。
    即使表达方式不同，只要核心含义、数值、逻辑关系一致就应该判断为匹配。
    请只回答 '匹配' 或 '不匹配'。

    ---
    标准答案：{expected_str} 
    ---
    模型答案：{actual_str} 
    ---
    """

    try:
        from gym.config.config import JUDGE_MODEL
        from gym.utils.client_manager import get_client

        evaluation_model = JUDGE_MODEL

        payload = {
            "model": evaluation_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 5
        }

        client = get_client(evaluation_model)

        response = client.chat_completions_create(
            messages=payload["messages"],
            max_tokens=payload["max_tokens"]
        )

        result = response.choices[0].message.content.strip()
        is_match = result == "匹配"

        print(f"  二次LLM验证 [{path}]: {result} ({'✅' if is_match else '❌'})")
        return is_match

    except Exception as e:
        print(f"  ❌ 二次LLM验证失败 [{path}]: {e}")
        return None


def calculate_answer_score(model_answer: Dict, golden_standard: Dict, tolerance: float = 0.05) -> Tuple[float, str, Dict]:
    """
    计算模型答案与标准答案的匹配分数

    这是一个纯函数，只负责评分算法，不涉及任何外部依赖。

    Args:
        model_answer: 模型的结构化答案
        golden_standard: 标准答案
        tolerance: 数值比较的容差

    Returns:
        Tuple[分数, 摘要, 详细信息]
    """

    def compare_values_recursive(actual, expected, path="", tolerance=0.05):
        """递归比较值，返回 (是否匹配, 详细信息字典)"""
        current_path = path

        # 定义数值转换函数
        def try_convert_to_number(value):
            """尝试将值转换为数值，如果不能转换则返回原值"""
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                # 去除前后空格
                value = value.strip()
                try:
                    # 先尝试转换为int
                    if '.' not in value and 'e' not in value.lower() and 'E' not in value:
                        return int(value)
                    # 再尝试转换为float
                    return float(value)
                except ValueError:
                    return value
            return value

        # 尝试数值转换
        expected_converted = try_convert_to_number(expected)
        actual_converted = try_convert_to_number(actual)

        # 字典比较
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False, {
                    current_path: {
                        "status": "type_mismatch",
                        "path": current_path,
                        "expected": expected,
                        "actual": actual
                    }
                }

            all_match = True
            all_details = {}

            for key, expected_value in expected.items():
                key_path = f"{current_path}.{key}" if current_path else key

                if key not in actual:
                    all_match = False
                    all_details[key_path] = {
                        "status": "missing",
                        "path": key_path,
                        "expected": expected_value,
                        "actual": None
                    }
                    continue

                match, details = compare_values_recursive(actual[key], expected_value, key_path, tolerance)
                if not match:
                    all_match = False

                # 合并子节点的详细信息
                all_details.update(details)

            # 检查是否有多余的键
            for key in actual:
                if key not in expected:
                    key_path = f"{current_path}.{key}" if current_path else key
                    all_details[key_path] = {
                        "status": "unexpected",
                        "path": key_path,
                        "expected": None,
                        "actual": actual[key]
                    }

            return all_match, all_details

        # 列表比较
        elif isinstance(expected, list):
            if not isinstance(actual, list):
                return False, {
                    current_path: {
                        "status": "type_mismatch",
                        "path": current_path,
                        "expected": expected,
                        "actual": actual
                    }
                }

            if len(actual) != len(expected):
                return False, {
                    current_path: {
                        "status": "length_mismatch",
                        "path": current_path,
                        "expected_length": len(expected),
                        "actual_length": len(actual),
                        "expected": expected,
                        "actual": actual
                    }
                }

            all_match = True
            all_details = {}

            for i, (actual_item, expected_item) in enumerate(zip(actual, expected)):
                item_path = f"{current_path}[{i}]" if current_path else f"[{i}]"
                match, details = compare_values_recursive(actual_item, expected_item, item_path, tolerance)
                if not match:
                    all_match = False

                # 合并子节点的详细信息
                all_details.update(details)

            return all_match, all_details

        # 叶子节点比较（数值、布尔、字符串等）
        else:
            def apply_secondary_verification_if_needed(manual_match_result, match_type, result_detail):
                """如果人工匹配失败，应用二次LLM验证"""
                if manual_match_result:
                    # 人工匹配成功，直接返回
                    result_detail["status"] = "match"
                    result_detail["match_type"] = match_type
                    return True, {current_path: result_detail}
                else:
                    # 人工匹配失败，进行二次LLM验证
                    secondary_match_result = secondary_verification_with_llm(actual, expected, current_path)
                    if secondary_match_result is True:
                        result_detail["status"] = "match"
                        result_detail["match_type"] = "secondary_llm_verification"
                        return True, {current_path: result_detail}
                    elif secondary_match_result is False:
                        result_detail["status"] = "mismatch"
                        result_detail["match_type"] = "secondary_llm_failed"
                        return False, {current_path: result_detail}
                    else:
                        # 二次LLM验证失败或不适用，最终标记为不匹配
                        result_detail["status"] = "mismatch"
                        result_detail["match_type"] = "manual_and_llm_failed"
                        return False, {current_path: result_detail}

            result_detail = {"path": current_path, "expected": expected, "actual": actual}

            # 使用之前转换的数值
            expected_num = expected_converted
            actual_num = actual_converted

            # 数值比较（包括字符串形式的数字）
            if isinstance(expected_num, (int, float)) and isinstance(actual_num, (int, float)):
                expected_val = float(expected_num)
                actual_val = float(actual_num)

                result_detail["expected_num"] = expected_val
                result_detail["actual_num"] = actual_val

                # 使用更宽松的比较策略
                # 1. 先尝试直接相等比较
                if expected_val == actual_val:
                    return apply_secondary_verification_if_needed(True, "exact", result_detail)

                # 2. 保留两位小数进行比较
                expected_rounded = round(expected_val, 2)
                actual_rounded = round(actual_val, 2)
                result_detail["expected_rounded"] = expected_rounded
                result_detail["actual_rounded"] = actual_rounded

                if expected_rounded == actual_rounded:
                    result_detail["error"] = abs(actual_val - expected_val)
                    return apply_secondary_verification_if_needed(True, "rounded", result_detail)

                # 3. 使用相对误差比较（对于较大的数值）
                if abs(expected_val) > 1:
                    relative_error = abs(actual_val - expected_val) / abs(expected_val)
                    result_detail["relative_error"] = relative_error
                    if relative_error <= tolerance:
                        return apply_secondary_verification_if_needed(True, "tolerance", result_detail)

                # 4. 使用绝对误差比较（对于较小的数值）
                else:
                    absolute_error = abs(actual_val - expected_val)
                    result_detail["absolute_error"] = absolute_error
                    if absolute_error <= tolerance:
                        return apply_secondary_verification_if_needed(True, "tolerance", result_detail)

                # 数值不匹配，记录误差信息并进行二次验证
                if abs(expected_val) > 0:
                    result_detail["relative_error"] = abs(actual_val - expected_val) / abs(expected_val)
                result_detail["absolute_error"] = abs(actual_val - expected_val)
                return apply_secondary_verification_if_needed(False, "numeric_mismatch", result_detail)

            # 布尔值比较
            elif isinstance(expected, bool) and isinstance(actual, bool):
                manual_match = (expected == actual)
                return apply_secondary_verification_if_needed(manual_match, "exact" if manual_match else "boolean_mismatch", result_detail)

            # 字符串比较
            elif isinstance(expected, str) and isinstance(actual, str):
                # 先进行基本的人工匹配（忽略大小写和前后空格）
                expected_clean = expected.strip().lower()
                actual_clean = actual.strip().lower()
                manual_match = (expected_clean == actual_clean)
                return apply_secondary_verification_if_needed(manual_match, "exact" if manual_match else "string_mismatch", result_detail)

            # 其他类型的直接比较
            else:
                manual_match = (expected == actual)
                return apply_secondary_verification_if_needed(manual_match, "exact" if manual_match else "other_type_mismatch", result_detail)

    if not model_answer or not golden_standard:
        return 0.0, "缺少答案数据", {}

    if not isinstance(model_answer, dict) or not isinstance(golden_standard, dict):
        return 0.0, "答案格式不匹配", {}

    # 进行递归比较
    overall_match, field_details = compare_values_recursive(model_answer, golden_standard, "", tolerance)

    # 计算分数
    total_fields = _count_fields_recursive(golden_standard)
    matched_fields = _count_matches_from_details(field_details)

    if total_fields == 0:
        return 0.0, "标准答案为空", {}

    score = matched_fields / total_fields
    summary = f"匹配字段: {matched_fields}/{total_fields} (得分: {score:.2%})"

    return score, summary, field_details


# 为了保持向后兼容性，保留这个函数
def evaluate_saved_traces(model_name=None, trace_dir=None, new_evaluate=False, lenient_mode=True):
    """
    向后兼容的函数，现在委托给新的评估服务

    建议直接使用 evaluation_service 模块中的函数
    """
    print("⚠️ 注意: evaluate_saved_traces 函数已被弃用，请使用 evaluation_service 模块")

    from gym.core.evaluation_service import evaluate_traces
    result = evaluate_traces(
        trace_dir=trace_dir,
        model_name=model_name,
        force_reextract=new_evaluate,
        lenient_mode=lenient_mode,
    )

    if result['success']:
        return result
    else:
        print(f"❌ 评估失败: {result.get('error', 'Unknown error')}")
        return None


if __name__ == "__main__":
    """使用示例 - 现在委托给新的评估服务"""
    print("🔄 重定向到新的评估服务...")

    import sys
    from gym.core.evaluation_service import evaluate_traces

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"📊 评估模型: {model_name}")
        result = evaluate_traces(model_name=model_name, force_reextract=True, lenient_mode=True)
    else:
        print("📊 评估所有模型")
        result = evaluate_traces(force_reextract=True, lenient_mode=True)
