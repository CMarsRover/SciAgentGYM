"""
评估服务模块 - 独立的模型回答评估功能

这个模块专门负责对已保存的模型回答进行评估，与测试执行完全分离。
实现高内聚低耦合的设计原则。
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path so we can import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gym.core.data_loader import extract_structured_answer_from_response, _post_process_extracted_answer
from gym.core.evaluator import calculate_answer_score, compute_segment_scores_from_details
from gym.config.dataset_config import get_trace_root, list_available_datasets

TRACE_ROOT_NAMES = {entry.trace_root.name for entry in list_available_datasets()}


class EvaluationService:
    """评估服务类 - 负责对模型回答进行结构化提取和评分"""

    def __init__(self, lenient_mode: bool = True):
        """
        初始化评估服务

        Args:
            lenient_mode: 是否使用宽松模式（对数值比较更宽松）
        """
        self.lenient_mode = lenient_mode
        self.tolerance = 0.05 if lenient_mode else 0.01

    def extract_answer_from_trace(self, trace_data: Dict) -> Optional[Dict]:
        """
        从轨迹数据中提取结构化答案

        Args:
            trace_data: 完整的轨迹数据字典

        Returns:
            提取的结构化答案，如果提取失败返回None
        """
        model_answer = None

        # 1. 尝试重新提取答案（如果有原始回答）
        raw_answer = trace_data.get('model_raw_answer')
        if not raw_answer or not isinstance(raw_answer, str) or not raw_answer.strip():
            # 如果没有原始回答，尝试从messages中提取最后一条assistant消息
            raw_answer = self._extract_last_assistant_text(trace_data)

        if raw_answer and isinstance(raw_answer, str) and raw_answer.strip():
            # 重新提取结构化答案
            extracted = extract_structured_answer_from_response(raw_answer)
            if extracted is not None:
                model_answer = extracted

        # 2. 如果重新提取失败，使用已有的结构化答案
        if model_answer is None:
            model_answer = trace_data.get('model_structured_answer')

        # 3. 对答案进行后处理，修复常见问题
        if model_answer is not None:
            processed_answer = _post_process_extracted_answer(model_answer)
            model_answer = processed_answer

        return model_answer

    def evaluate_single_trace(self, trace_file_path: str, force_reextract: bool = False) -> Dict:
        """
        评估单个轨迹文件

        Args:
            trace_file_path: 轨迹文件路径
            force_reextract: 是否强制重新提取答案

        Returns:
            评估结果字典，包含分数、摘要和详细信息
        """
        try:
            with open(trace_file_path, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
        except Exception as e:
            return {
                'success': False,
                'error': f'读取轨迹文件失败: {e}',
                'trace_file': trace_file_path
            }

        # 提取模型答案
        if force_reextract or 'model_structured_answer' not in trace_data:
            model_answer = self.extract_answer_from_trace(trace_data)
            # 更新轨迹文件中的结构化答案
            if model_answer is not None:
                trace_data['model_structured_answer'] = model_answer
        else:
            model_answer = trace_data.get('model_structured_answer')
            # 对现有答案也应用后处理
            if model_answer is not None:
                processed_answer = _post_process_extracted_answer(model_answer)
                if processed_answer != model_answer:
                    trace_data['model_structured_answer'] = processed_answer
                    model_answer = processed_answer

        # 检查是否有标准答案
        golden_standard = trace_data.get('golden_standard')
        if not golden_standard:
            return {
                'success': False,
                'error': '缺少标准答案(golden_standard)',
                'trace_file': trace_file_path
            }

        # 进行评分
        if model_answer is not None:
            score, summary, details = self._calculate_answer_score(model_answer, golden_standard)

            # 更新轨迹文件
            trace_data['evaluation_score'] = score
            trace_data['evaluation_summary'] = summary
            trace_data['evaluation_details'] = details
            segment_scores = compute_segment_scores_from_details(golden_standard, details)
            if segment_scores:
                trace_data['evaluation_score_segments'] = segment_scores
            if 'evaluation_failure_reason' in trace_data:
                del trace_data['evaluation_failure_reason']  # 移除失败标记

            # 保存更新后的轨迹文件
            try:
                with open(trace_file_path, 'w', encoding='utf-8') as f:
                    json.dump(trace_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ 保存轨迹文件失败 {trace_file_path}: {e}")

            return {
                'success': True,
                'score': score,
                'summary': summary,
                'details': details,
                'trace_file': trace_file_path
            }
        else:
            # 提取失败
            score = 0.0
            summary = "缺少或无法提取结构化答案"
            details = {"error": "model_structured_answer is missing, null, or could not be extracted from raw answer"}

            # 更新轨迹文件
            trace_data['evaluation_score'] = score
            trace_data['evaluation_summary'] = summary
            trace_data['evaluation_details'] = details
            segment_scores = compute_segment_scores_from_details(golden_standard, details)
            if segment_scores:
                trace_data['evaluation_score_segments'] = segment_scores
            trace_data['evaluation_failure_reason'] = 'extraction_failed_or_missing'

            # 保存更新后的轨迹文件
            try:
                with open(trace_file_path, 'w', encoding='utf-8') as f:
                    json.dump(trace_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"⚠️ 保存轨迹文件失败 {trace_file_path}: {e}")

            return {
                'success': True,
                'score': score,
                'summary': summary,
                'details': details,
                'trace_file': trace_file_path
            }

    def evaluate_trace_directory(
        self,
        trace_dir: str,
        model_name: Optional[str] = None,
        force_reextract: bool = False,
        match_type_filter: Optional[str] = None,
        include_missing_only: bool = False,
    ) -> Dict:
        """
        评估轨迹目录中的所有文件

        Args:
            trace_dir: 轨迹目录路径
            model_name: 指定模型名称，None表示所有模型
            force_reextract: 是否强制重新提取答案
            match_type_filter: 仅重新评测 evaluation_details 中包含指定 match_type 的轨迹文件
            include_missing_only: 是否只重新评测 evaluation_score 缺失或评估失败的轨迹

        Returns:
            评估统计结果
        """
        if not os.path.exists(trace_dir):
            return {
                'success': False,
                'error': f'轨迹目录不存在: {trace_dir}'
            }

        print(f"📊 开始评估轨迹文件: {trace_dir}")

        # 收集轨迹文件
        trace_files = self._collect_trace_files(trace_dir, model_name)

        # 可选：仅保留包含指定 match_type 的轨迹
        if match_type_filter:
            original_count = len(trace_files)
            trace_files = self._filter_trace_files_by_match_type(trace_files, match_type_filter)
            print(f"🎯 过滤 match_type={match_type_filter} 的轨迹: {len(trace_files)}/{original_count} 个")

        if not trace_files:
            return {
                'success': False,
                'error': '未找到任何轨迹文件' if not match_type_filter else f'未找到包含 match_type={match_type_filter} 的轨迹文件'
            }

        print(f"找到 {len(trace_files)} 个轨迹文件")

        # 评估统计
        total_score = 0
        evaluated_count = 0
        model_stats = {}
        evaluation_results = []

        def is_missing_or_failed(trace_path: str) -> bool:
            """检查轨迹是否缺少评分或标记为失败。"""
            try:
                with open(trace_path, 'r', encoding='utf-8') as f_in:
                    data = json.load(f_in)
            except Exception:
                return True

            if 'evaluation_score' not in data:
                return True

            failure_reason = data.get('evaluation_failure_reason')
            if failure_reason:
                return True

            return False

        filtered_trace_files = []
        for trace_file in trace_files:
            if include_missing_only and not is_missing_or_failed(trace_file):
                continue
            filtered_trace_files.append(trace_file)

        if not filtered_trace_files:
            return {
                'success': False,
                'error': '未找到需要重新评估的轨迹文件'
            }

        trace_files_to_process = filtered_trace_files

        for trace_file in trace_files_to_process:
            # 从文件路径中提取模型和案例信息
            model, case_id, mode = self._extract_trace_info(trace_file)
            display_id = f"{model} - 案例{case_id}"
            if mode != 'unknown':
                display_id += f" ({mode})"

            # 评估单个轨迹
            result = self.evaluate_single_trace(trace_file, force_reextract)

            if result['success']:
                score = result['score']
                summary = result['summary']

                previously_scored = not is_missing_or_failed(trace_file)
                if force_reextract or not previously_scored:
                    print(f"🔄 {display_id}: {score:.2%} (已重新评估)")
                else:
                    print(f"✅ {display_id}: {score:.2%} (已评分)")

                # 统计
                stats_key = f"{model}"
                if mode != 'unknown':
                    stats_key += f" ({mode})"

                if stats_key not in model_stats:
                    model_stats[stats_key] = {'scores': [], 'count': 0}
                model_stats[stats_key]['scores'].append(score)
                model_stats[stats_key]['count'] += 1

                total_score += score
                evaluated_count += 1

                evaluation_results.append({
                    'model': model,
                    'case_id': case_id,
                    'mode': mode,
                    'score': score,
                    'summary': summary
                })
            else:
                print(f"❌ 处理轨迹文件失败 {trace_file}: {result.get('error', 'Unknown error')}")

        # 输出统计结果
        if evaluated_count > 0:
            overall_avg = total_score / evaluated_count
            print(f"\n📈 评估统计:")
            print(f"总体平均分: {overall_avg:.2%} ({evaluated_count}个案例)")

            # 按模型分组显示统计
            for stats_key, stats in sorted(model_stats.items()):
                if stats['count'] > 0:
                    model_avg = sum(stats['scores']) / stats['count']
                    print(f"  {stats_key}: {model_avg:.2%} ({stats['count']}个案例)")
        else:
            print("没有可评估的案例")

        return {
            'success': True,
            'total_evaluated': evaluated_count,
            'overall_average': total_score / evaluated_count if evaluated_count > 0 else 0,
            'model_stats': model_stats,
            'evaluation_results': evaluation_results
        }

    def _extract_last_assistant_text(self, trace_data: Dict) -> Optional[str]:
        """从trace的messages中提取最后一条assistant文本内容"""
        try:
            # 优先使用单独保存的最后一条assistant消息（如果存在）
            last_assist = trace_data.get('model_last_assistant_message')
            if isinstance(last_assist, dict):
                content = last_assist.get('content')
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        try:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                t = part.get('text', '')
                                if isinstance(t, str) and t.strip():
                                    parts.append(t.strip())
                        except Exception:
                            continue
                    if parts:
                        return "\n\n".join(parts)

            messages = trace_data.get('messages', []) or []
            # 反向遍历，找到最后一条assistant消息
            for msg in reversed(messages):
                if not isinstance(msg, dict):
                    continue
                if msg.get('role') != 'assistant':
                    continue
                content = msg.get('content')
                # 直接字符串
                if isinstance(content, str):
                    txt = content.strip()
                    if txt:
                        return txt
                # 多模态列表
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        try:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                t = part.get('text', '')
                                if isinstance(t, str) and t.strip():
                                    parts.append(t.strip())
                        except Exception:
                            continue
                    if parts:
                        return "\n\n".join(parts)
        except Exception:
            pass
        return None

    def _collect_trace_files(self, directory: str, model_name: Optional[str] = None) -> List[str]:
        """递归收集轨迹文件，兼容 with_tools_react / with_all_tools 等后缀"""
        files: List[str] = []
        allowed_prefixes = ("with_tools", "without_tools", "with_all_tools")

        try:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)

                if os.path.isfile(item_path) and item.endswith('_trace.json'):
                    files.append(item_path)
                elif os.path.isdir(item_path):
                    # 兼容 with_tools_react / without_tools_react / with_all_tools_react 等目录名
                    if item.startswith(allowed_prefixes):
                        files.extend(self._collect_trace_files(item_path, model_name))
                    else:
                        # 始终向下递归，这样模型目录下的其它层级也能被扫描到
                        files.extend(self._collect_trace_files(item_path, model_name if model_name else item))
        except PermissionError:
            print(f"⚠️ 无法访问目录: {directory}")
        except Exception as e:
            print(f"⚠️ 读取目录时出错 {directory}: {e}")

        return files

    def _extract_trace_info(self, trace_file: str) -> Tuple[str, str, str]:
        """从轨迹文件路径中提取模型名称、案例ID和模式"""
        try:
            with open(trace_file, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)

            model = trace_data.get('model', 'unknown')
            case_id = trace_data.get('id', 'unknown')
            mode = trace_data.get('mode', 'unknown')

            if model == 'unknown':
                # 尝试从文件路径推断模型名称
                path_parts = trace_file.replace(os.sep, '/').split('/')
                for i, part in enumerate(path_parts):
                    if part in TRACE_ROOT_NAMES and i + 1 < len(path_parts):
                        model = path_parts[i + 1]
                        break

            return model, case_id, mode
        except Exception:
            return 'unknown', 'unknown', 'unknown'

    def _calculate_answer_score(self, model_answer: Dict, golden_standard: Dict) -> Tuple[float, str, Dict]:
        """计算模型答案与标准答案的匹配分数"""

        return calculate_answer_score(model_answer, golden_standard, self.tolerance)

    def _filter_trace_files_by_match_type(self, trace_files: List[str], match_type: str) -> List[str]:
        """过滤出 evaluation_details（及其他嵌套结构）中包含指定 match_type 的轨迹文件"""

        def contains_match_type(obj: Any) -> bool:
            if isinstance(obj, dict):
                if obj.get('match_type') == match_type:
                    return True
                for value in obj.values():
                    if contains_match_type(value):
                        return True
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if contains_match_type(item):
                        return True
            return False

        filtered_files: List[str] = []
        for file_path in trace_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    trace_data = json.load(f)
            except Exception as exc:
                print(f"⚠️ 无法读取 {file_path} 进行过滤: {exc}")
                continue

            if contains_match_type(trace_data.get('evaluation_details')) or contains_match_type(trace_data.get('attempts')):
                filtered_files.append(file_path)
        return filtered_files


# 便捷函数
def evaluate_traces(trace_dir: str = None, model_name: str = None,
                   force_reextract: bool = False, lenient_mode: bool = True,
                   match_type_filter: Optional[str] = None,
                   include_missing_only: bool = False) -> Dict:
    """
    便捷函数：评估轨迹文件

    Args:
        trace_dir: 轨迹目录，默认为当前数据集的 trace 目录
        model_name: 指定模型名称，None表示所有模型
        force_reextract: 是否强制重新提取答案
        lenient_mode: 是否使用宽松模式
        match_type_filter: 仅重新评测包含指定 match_type 的轨迹
        include_missing_only: 是否仅评测 evaluation_score 缺失或存在评估失败原因的轨迹

    Returns:
        评估统计结果
    """
    if trace_dir is None:
        trace_root = get_trace_root()
        if model_name:
            trace_dir = str(trace_root / model_name)
        else:
            trace_dir = str(trace_root)

    service = EvaluationService(lenient_mode=lenient_mode)
    return service.evaluate_trace_directory(
        trace_dir,
        model_name,
        force_reextract,
        match_type_filter=match_type_filter,
        include_missing_only=include_missing_only,
    )


def evaluate_single_trace_file(trace_file: str, force_reextract: bool = False,
                              lenient_mode: bool = True) -> Dict:
    """
    便捷函数：评估单个轨迹文件

    Args:
        trace_file: 轨迹文件路径
        force_reextract: 是否强制重新提取答案
        lenient_mode: 是否使用宽松模式

    Returns:
        评估结果
    """
    service = EvaluationService(lenient_mode=lenient_mode)
    return service.evaluate_single_trace(trace_file, force_reextract)


if __name__ == "__main__":
    """使用示例"""
    import sys

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        print(f"📊 评估模型: {model_name}")
        result = evaluate_traces(model_name=model_name, force_reextract=True, lenient_mode=True)
    else:
        print("📊 评估所有模型")
        result = evaluate_traces(force_reextract=True, lenient_mode=True)

    if result['success']:
        print(f"\n✅ 评估完成，总计评估 {result['total_evaluated']} 个案例")
    else:
        print(f"\n❌ 评估失败: {result.get('error', 'Unknown error')}")
