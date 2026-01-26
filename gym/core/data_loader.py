"""
数据加载和处理模块
"""
import json
import re
import base64
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from gym.config.dataset_config import (
    get_dataset_entry,
    get_current_dataset_key,
    get_trace_root,
)

def ensure_metadata_summary(case_data, default_test_type=None):
    """Ensure cases carry a compact metadata summary for downstream stats/traces."""
    if not isinstance(case_data, dict):
        return {}

    metadata = case_data.get('metadata')
    if not isinstance(metadata, dict):
        metadata = {}

    # Copy existing summary if present so we don't drop precomputed values
    summary = dict(case_data.get('metadata_summary') or {})

    subject = metadata.get('subject')
    if subject:
        summary['subject'] = subject

    topic = metadata.get('topic')
    if topic:
        summary['topic'] = topic

    test_type = metadata.get('test_type') or default_test_type or summary.get('test_type')
    if test_type:
        metadata['test_type'] = test_type
        summary['test_type'] = test_type

    original_id = metadata.get('original_question_id') or case_data.get('original_id')
    if original_id:
        original_id = str(original_id)
        metadata['original_question_id'] = original_id
        summary['original_question_id'] = original_id

    # Provide a stable composite key that is handy for group-by statistics
    if subject and topic:
        summary['subject_topic_key'] = f"{subject}::{topic}"

    # Remove empty values
    summary = {k: v for k, v in summary.items() if v not in (None, '', [], {})}

    case_data['metadata'] = metadata
    if summary:
        case_data['metadata_summary'] = summary
    else:
        case_data.pop('metadata_summary', None)

    return case_data.get('metadata_summary', {})


def load_test_cases_from_dataset():
    """从dataset/data_toolusage.json加载测试案例"""
    try:
        # 使用绝对路径
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'dataset' / 'merged_questions_augmented_generated.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)

        for case in cases:
            ensure_metadata_summary(case, default_test_type='normal')
        return cases
    except Exception as e:
        print(f"加载测试案例失败: {e}")
        return []


def group_cases_by_subject(cases: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """按照 metadata.subject 对测试案例进行分组"""
    subject_map: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        metadata = case.get('metadata') or {}
        subject = metadata.get('subject') or '未知科目'
        subject_map.setdefault(subject, []).append(case)
    return subject_map


def group_cases_by_topic(
    cases: List[Dict[str, Any]],
    fallback: str = '其他',
) -> Dict[str, List[Dict[str, Any]]]:
    """按照 metadata.topic 对测试案例进行分组，缺失字段归为 fallback"""
    topic_map: Dict[str, List[Dict[str, Any]]] = {}
    for case in cases:
        metadata = case.get('metadata') or {}
        if not isinstance(metadata, dict):
            metadata = {}

        topic_raw = metadata.get('topic')
        if not topic_raw:
            summary = case.get('metadata_summary') or {}
            if isinstance(summary, dict):
                topic_raw = summary.get('topic')

        topic = str(topic_raw).strip() if topic_raw else ''
        key = topic or fallback
        topic_map.setdefault(key, []).append(case)
    return topic_map


def deduplicate_usage_tool_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """确保每个工具名称唯一，同时清理类内重复工具"""
    global_seen = set()
    deduped_entries: List[Dict[str, Any]] = []
    duplicates_removed = 0

    for entry in entries:
        if not isinstance(entry, dict):
            deduped_entries.append(entry)
            continue

        if 'class_name' in entry and isinstance(entry.get('tools'), list):
            class_tools = []
            for tool in entry.get('tools', []):
                if not isinstance(tool, dict):
                    class_tools.append(tool)
                    continue
                func_meta = tool.get('function') or {}
                name = func_meta.get('name')
                if name and name in global_seen:
                    duplicates_removed += 1
                    continue
                if name:
                    global_seen.add(name)
                class_tools.append(tool)

            if class_tools:
                class_entry = deepcopy(entry)
                class_entry['tools'] = class_tools
                deduped_entries.append(class_entry)
            else:
                duplicates_removed += 1
            continue

        func_meta = entry.get('function') or {}
        name = func_meta.get('name')
        if name and name in global_seen:
            duplicates_removed += 1
            continue
        if name:
            global_seen.add(name)
        deduped_entries.append(entry)

    if duplicates_removed:
        print(f"⚠️ 工具列表去除了 {duplicates_removed} 个重复工具名称")

    return deduped_entries


def aggregate_usage_tool_protocol_for_cases(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """聚合同一批案例的 usage_tool_protocol，去重并保留原始结构"""
    aggregated: List[Dict[str, Any]] = []
    seen_keys = set()
    class_entries: Dict[str, Dict[str, Any]] = {}

    for case in cases:
        protocols = case.get('usage_tool_protocol') or []
        for proto in protocols:
            if not isinstance(proto, dict):
                continue

            func_block = proto.get('function')
            if isinstance(func_block, dict) and func_block.get('name'):
                func_name = func_block.get('name')
                addl = proto.get('additionalProperties') or {}
                func_path = addl.get('function_path')
                key = ('function', func_name, func_path)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                aggregated.append(deepcopy(proto))
                continue

            if 'class_name' in proto and isinstance(proto.get('tools'), list):
                class_name = proto.get('class_name') or 'anonymous_class'
                key = ('class', class_name)
                if key not in seen_keys:
                    class_entry = deepcopy(proto)
                    if not isinstance(class_entry.get('tools'), list):
                        class_entry['tools'] = []
                    aggregated.append(class_entry)
                    class_entries[class_name] = class_entry
                    seen_keys.add(key)
                else:
                    class_entry = class_entries.get(class_name)
                    if class_entry is None:
                        continue

                existing_tools = class_entry.get('tools') or []
                existing_keys = {
                    (
                        (tool.get('function') or {}).get('name'),
                        ((tool.get('additionalProperties') or {}).get('function_path'))
                    )
                    for tool in existing_tools
                    if isinstance(tool, dict)
                }
                for nested_tool in proto.get('tools', []):
                    if not isinstance(nested_tool, dict):
                        continue
                    nested_func = nested_tool.get('function') or {}
                    nested_name = nested_func.get('name')
                    nested_path = (nested_tool.get('additionalProperties') or {}).get('function_path')
                    nested_key = (nested_name, nested_path)
                    if nested_key in existing_keys:
                        continue
                    existing_keys.add(nested_key)
                    existing_tools.append(deepcopy(nested_tool))
                class_entry['tools'] = existing_tools
                continue

            serialized = json.dumps(proto, sort_keys=True, ensure_ascii=False)
            key = ('raw', serialized)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            aggregated.append(deepcopy(proto))

    return deduplicate_usage_tool_entries(aggregated)


def load_augmented_test_cases_from_dataset():
    """从dataset/merged_questions_augmented.json加载增强版测试案例（反向测试）
    
    这个函数将augmented_versions中的内容转换为独立的测试案例，
    使用augmented_question作为问题，final_answer作为标准答案
    
    Returns:
        list: 增强版测试案例列表，每个案例包含：
        - id: 原始案例ID + 增强版本索引（如 "1_aug_0"）
        - question: augmented_question
        - answer: final_answer
        - metadata: 从原始案例继承，但使用final_answer作为golden_answer
        - original_id: 原始案例ID
        - augmented_index: 增强版本索引
    """
    try:
        # 使用绝对路径
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'dataset' / 'merged_questions_augmented_generated.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            original_cases = json.load(f)
        
        augmented_cases = []
        
        for case in original_cases:
            original_id = case.get('id')
            augmented_versions = case.get('augmented_versions', [])
            
            # 如果没有增强版本，跳过
            if not augmented_versions:
                continue
            
            # 为每个增强版本创建独立的测试案例
            for aug_index, aug_version in enumerate(augmented_versions):
                augmented_question = aug_version.get('augmented_question')
                final_answer = aug_version.get('final_answer')
                
                # 确保必要字段存在
                if not augmented_question or not final_answer:
                    print(f"跳过案例 {original_id} 的增强版本 {aug_index}：缺少必要字段")
                    continue
                
                # 创建新的测试案例
                augmented_case = {
                    'id': f"{original_id}_aug_{aug_index}",
                    'original_id': original_id,
                    'augmented_index': aug_index,
                    'question': augmented_question,
                    'answer': str(final_answer),  # 将final_answer转换为字符串，保持与原始格式一致
                    'metadata': {
                        # 继承原始案例的元数据
                        'subject': case.get('metadata', {}).get('subject', ''),
                        'topic': case.get('metadata', {}).get('topic', ''),
                        'image_path': case.get('metadata', {}).get('image_path', []),
                        'solution_steps': aug_version.get('solution_outline', []),  # 使用增强版本的解决步骤
                        'tool_expected': case.get('metadata', {}).get('tool_expected', []),
                        # 使用final_answer作为golden_answer
                        'golden_answer': [final_answer] if isinstance(final_answer, dict) else [{'final_answer': final_answer}],
                        'original_question_id': str(original_id),
                        'test_type': 'augmented',  # 标记为增强版测试
                        'verification': aug_version.get('verification', ''),  # 添加验证信息
                    },
                    # 保留原始的工具协议
                    'usage_tool_protocol': case.get('usage_tool_protocol', []),
                    # 保存完整的增强版本信息
                    'augmented_version_data': aug_version
                }

                ensure_metadata_summary(augmented_case, default_test_type='augmented')
                augmented_cases.append(augmented_case)
        
        print(f"成功加载了 {len(augmented_cases)} 个增强版测试案例")
        return augmented_cases
        
    except Exception as e:
        print(f"加载增强版测试案例失败: {e}")
        return []


def load_refined_test_cases_from_dataset(
    dataset_key: Optional[str] = None,
    dataset_path: Optional[str] = None,
):
    """从配置的数据集中加载精炼版测试案例
    
    这个函数将refined_versions中的内容转换为独立的测试案例，
    使用refined_question作为问题，final_answer作为标准答案
    
    Returns:
        list: 精炼版测试案例列表，每个案例包含：
        - id: 原始案例ID + 精炼版本索引（如 "1_ref_0"）
        - question: refined_question
        - answer: final_answer
        - metadata: 从原始案例继承，但使用final_answer作为golden_answer
        - original_id: 原始案例ID
        - refined_index: 精炼版本索引
    """
    try:
        # 使用绝对路径
        project_root = Path(__file__).parent.parent.parent
        if dataset_path:
            data_path = Path(dataset_path)
            # 如果是相对路径，转换为绝对路径
            if not data_path.is_absolute():
                # 尝试相对于项目根目录
                abs_path = project_root / data_path
                if abs_path.exists():
                    data_path = abs_path
                else:
                    # 尝试相对于 gym 目录
                    gym_dir = Path(__file__).parent.parent
                    abs_path = gym_dir / data_path
                    if abs_path.exists():
                        data_path = abs_path
                    else:
                        # 最后尝试直接使用传入的路径（可能是绝对路径的字符串）
                        data_path = Path(dataset_path).resolve()
            resolved_dataset_key = dataset_key or get_current_dataset_key()
        else:
            entry = get_dataset_entry(dataset_key)
            data_path = entry.dataset_path
            resolved_dataset_key = entry.key
        trace_root = get_trace_root(resolved_dataset_key)
        with open(data_path, 'r', encoding='utf-8') as f:
            original_cases = json.load(f)
        
        refined_cases = []
        
        for case in original_cases:
            original_id = case.get('id')
            refined_versions = case.get('refined_versions', [])
            
            # 如果没有精炼版本，跳过
            if not refined_versions:
                continue
            
            # 为每个精炼版本创建独立的测试案例
            for ref_index, ref_version in enumerate(refined_versions):
                refined_question = ref_version.get('refined_question')
                final_answer = ref_version.get('final_answer')
                
                # 确保必要字段存在
                if not refined_question or not final_answer:
                    print(f"跳过案例 {original_id} 的精炼版本 {ref_index}：缺少必要字段")
                    continue
                
                # 创建新的测试案例
                refined_case = {
                    'id': f"{original_id}_ref_{ref_index}",
                    'original_id': original_id,
                    'refined_index': ref_index,
                    'question': refined_question,
                    'answer': str(final_answer),  # 将final_answer转换为字符串，保持与原始格式一致
                    'metadata': {
                        # 继承原始案例的元数据
                        'subject': case.get('metadata', {}).get('subject', ''),
                        'topic': case.get('metadata', {}).get('topic', ''),
                        'image_path': case.get('metadata', {}).get('image_path', []),
                        'solution_steps': case.get('metadata', {}).get('solution_steps', []),
                        'tool_expected': case.get('metadata', {}).get('tool_expected', []),
                        # 使用final_answer作为golden_answer
                        'golden_answer': [final_answer] if isinstance(final_answer, dict) else [{'final_answer': final_answer}],
                        'original_question_id': str(original_id),
                        'test_type': 'refined',  # 标记为精炼版测试
                        'dataset_key': resolved_dataset_key,
                        'trace_root': str(trace_root),
                    },
                    # 保留原始的工具协议
                    'usage_tool_protocol': case.get('usage_tool_protocol', []),
                    # 保存完整的精炼版本信息
                    'refined_version_data': ref_version
                }

                ensure_metadata_summary(refined_case, default_test_type='refined')
                refined_cases.append(refined_case)
        
        print(f"成功加载了 {len(refined_cases)} 个精炼版测试案例 (数据集: {resolved_dataset_key})")
        return refined_cases
        
    except Exception as e:
        print(f"加载精炼版测试案例失败: {e}")
        return []


def extract_image_paths(question_text):
    """从问题文本中提取图片路径
    
    Args:
        question_text: 问题文本，可能包含 <images/filename.ext> 格式的图片路径
        
    Returns:
        tuple: (clean_question_text, image_paths_list)
    """
    if not question_text:
        return question_text, []
    
    # 匹配 <images/filename.ext> 格式的图片路径
    image_pattern = r'<images/([^>]+)>'
    image_matches = re.findall(image_pattern, question_text)
    
    # 从问题文本中移除图片标签
    clean_question = re.sub(image_pattern, '', question_text).strip()
    
    return clean_question, image_matches


def normalize_image_path(image_path_str: str) -> str:
    """将旧的图片路径转换为新的统一路径格式
    
    将以下格式的路径：
    - "failed_question_images/xxx.jpg"
    - "filtered_images/xxx.jpg"
    - "/sfe_images/xxx.png" 或 "sfe_images/xxx.png"
    - "/r_bench/images/xxx.png" 或 "r_bench/images/xxx.png"
    
    转换为：
    - "gym/test_images/xxx.png" (直接保存在 test_images 目录下，不保留子目录结构)
    
    Args:
        image_path_str: 原始图片路径
        
    Returns:
        str: 转换后的统一路径格式
    """
    if not image_path_str or not isinstance(image_path_str, str):
        return image_path_str
    
    # 移除开头的 / 和 ./
    path_str = image_path_str.lstrip('/').lstrip('./')
    
    # 如果已经是新格式，直接返回
    if path_str.startswith('gym/test_images/'):
        return path_str
    
    # 提取文件名（不包含目录结构）
    path_obj = Path(path_str)
    filename = path_obj.name
    
    # 统一转换为 .png 扩展名
    if path_obj.suffix:
        filename = path_obj.stem + '.png'
    else:
        # 如果没有扩展名，尝试从原路径获取
        original_ext = Path(image_path_str).suffix
        if original_ext:
            filename = path_obj.name.rsplit('.', 1)[0] + '.png'
        else:
            filename = path_obj.name + '.png'
    
    # 直接保存在 gym/test_images/ 目录下，不保留子目录结构
    new_path = f"gym/test_images/{filename}"
    
    return new_path


def load_image_as_base64(image_path_or_filename):
    """加载图片文件并转换为base64编码
    
    Args:
        image_path_or_filename: 图片文件路径或文件名
        支持以下格式：
        - "filename.jpg" (在images目录下查找)
        - "failed_question_images/filename.jpg" (相对路径)
        - "gym/test_images/failed_question_images/filename.png" (新统一路径)
        - "/absolute/path/to/image.jpg" (绝对路径)
        
    Returns:
        tuple: (base64_string, mime_type) 或 (None, None) 如果加载失败
    """
    try:
        project_root = Path(__file__).parent.parent.parent

        image_path_str = image_path_or_filename
        if image_path_str.startswith('/'):
            # 如果路径以 / 开头，先移除它，然后当作项目根目录下的相对路径
            image_path_str = image_path_str.lstrip('/')

        # 首先尝试从新的统一路径加载
        if image_path_str.startswith('gym/test_images/'):
            image_path = project_root / image_path_str
            if image_path.exists():
                # 根据文件扩展名确定MIME类型
                ext = image_path.suffix.lower()
                mime_type_map = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp'
                }
                mime_type = mime_type_map.get(ext, 'image/jpeg')
                
                # 读取图片文件并编码为base64
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    base64_string = base64.b64encode(image_data).decode('utf-8')
                    
                return base64_string, mime_type
            else:
                # 新路径不存在，直接返回失败
                # 注意：旧目录的回退逻辑已移除，所有图片应统一使用 gym/test_images/ 路径
                print(f"图片文件不存在（新路径）: {image_path}")
                return None, None

        if '/' in image_path_str:
            # 相对路径（如 "failed_question_images/filename.jpg" 或 "sfe_images/..."）
            image_path = project_root / image_path_str
        else:
            # 仅文件名（在images目录下查找）
            image_path = project_root / 'images' / image_path_str

        # 兼容老数据：部分图片实际放在 gym/ 下的子目录中
        if not image_path.exists():
            # 尝试在 gym/ 目录下查找
            alt_path = project_root / "gym" / image_path_str
            if alt_path.exists():
                image_path = alt_path

        if not image_path.exists():
            print(f"图片文件不存在: {image_path}")
            return None, None
        
        # 根据文件扩展名确定MIME类型
        ext = image_path.suffix.lower()
        mime_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_type_map.get(ext, 'image/jpeg')
        
        # 读取图片文件并编码为base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
            
        return base64_string, mime_type
        
    except Exception as e:
        print(f"加载图片失败 {image_path_or_filename}: {e}")
        return None, None


def process_question_with_images(question_text):
    """处理包含图片的问题文本
    
    Args:
        question_text: 原始问题文本
        
    Returns:
        dict: 包含处理后的问题和图片信息的字典
        {
            'text': str,  # 清理后的问题文本
            'images': [   # 图片信息列表
                {
                    'filename': str,
                    'base64': str,
                    'mime_type': str
                },
                ...
            ],
            'expected_image_count': int,
            'missing_images': [str]
        }
    """
    clean_text, image_filenames = extract_image_paths(question_text)
    
    result = {
        'text': clean_text,
        'images': [],
        'expected_image_count': len(image_filenames),
        'missing_images': []
    }
    
    # 加载所有图片
    for filename in image_filenames:
        base64_data, mime_type = load_image_as_base64(filename)
        if base64_data and mime_type:
            result['images'].append({
                'filename': filename,
                'base64': base64_data,
                'mime_type': mime_type
            })
        else:
            print(f"跳过无法加载的图片: {filename}")
            result['missing_images'].append(filename)
    
    return result


def process_question_with_images_from_metadata(query_data):
    """处理查询数据，从 metadata 中的 image_path 加载图片
    
    这个函数专门用于增强测试，从原始案例的 metadata.image_path 中加载图片
    
    Args:
        query_data: 测试案例数据，包含 metadata.image_path
        
    Returns:
        dict: 包含处理后的问题和图片信息的字典
        {
            'text': str,  # 问题文本
            'images': [   # 图片信息列表
                {
                    'path': str,
                    'base64': str,
                    'mime_type': str
                },
                ...
            ],
            'expected_image_count': int,
            'missing_images': [str]
        }
    """
    result = {
        'text': query_data.get('question', ''),
        'images': [],
        'expected_image_count': 0,
        'missing_images': []
    }
    
    # 从 metadata.image_path 中获取图片路径
    image_paths = query_data.get('metadata', {}).get('image_path', [])
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    elif not isinstance(image_paths, (list, tuple)):
        image_paths = []
    
    result['expected_image_count'] = len(image_paths)
    
    if image_paths:
        print(f"📸 发现 {len(image_paths)} 个图片路径")
        for image_path in image_paths:
            print(f"   正在处理图片: {image_path}")
            base64_data, mime_type = load_image_as_base64(image_path)
            if base64_data and mime_type:
                result['images'].append({
                    'path': image_path,
                    'base64': base64_data,
                    'mime_type': mime_type
                })
                print(f"   ✅ 成功加载图片: {image_path}")
            else:
                print(f"   ❌ 跳过无法加载的图片: {image_path}")
                result['missing_images'].append(image_path)
    
    return result


def extract_golden_answer_template(query_data):
    """从golden_answer中提取第一个元素作为规范回答模板
    
    支持复杂的嵌套结构，包括：
    - 数值矩阵 (如质量矩阵、雅可比矩阵)
    - 方程字符串数组 (如约束方程、运动方程)
    - 符号表达式矩阵 (如符号雅可比矩阵)
    - 嵌套字典结构 (如评估点参数)
    
    Args:
        query_data: 包含metadata.golden_answer的数据字典
        
    Returns:
        tuple: (template, original_data) 或 (None, None)
    """

    def create_template_recursive(obj, key_name=""):
        """递归创建模板，保留嵌套结构
        
        智能识别不同类型的数据结构：
        - 数值矩阵：显示维度信息
        - 符号表达式矩阵：标注为符号表达式
        - 方程字符串：根据上下文标注
        - 参数值：根据名称推断类型
        
        Args:
            obj: 要模板化的对象
            key_name: 当前对象的键名，用于上下文判断
            
        Returns:
            模板化后的对象结构
        """
        if isinstance(obj, dict):
            template = {}
            for key, value in obj.items():
                template[key] = create_template_recursive(value, key)
            return template
        elif isinstance(obj, list):
            if len(obj) > 0:
                first_item = obj[0]
                # 检查是否是数值矩阵/数组
                if isinstance(first_item, list):
                    # 检查是否是数值矩阵
                    if all(isinstance(x, (int, float)) for x in first_item):
                        return f"[{len(obj)}x{len(first_item)} 数值矩阵]"
                    # 检查是否是符号表达式矩阵（可能包含字符串和数值）
                    elif key_name == "symbolic":
                        return f"[{len(obj)}x{len(first_item)} 符号表达式矩阵]"
                    else:
                        # 对于其他混合类型数组，保留结构
                        return [create_template_recursive(first_item, f"{key_name}_item")]
                elif isinstance(first_item, (int, float)):
                    return f"[{len(obj)}个数值的数组]"
                elif isinstance(first_item, str):
                    # 特殊处理约束方程等字符串数组
                    if key_name in ["constraint_equations", "motion_equations"]:
                        return f"[{len(obj)}个方程字符串]"
                    else:
                        return f"[{len(obj)}个字符串的数组]"
                else:
                    # 对于复杂对象数组，使用第一个元素的模板
                    return [create_template_recursive(first_item, f"{key_name}_item")]
            else:
                return []
        elif isinstance(obj, bool):
            return "[布尔值]"
        elif isinstance(obj, (int, float)):
            # 根据上下文提供更具体的描述
            if key_name in ["theta", "beta", "S"]:
                return "[角度/位置参数]"
            elif "matrix" in key_name.lower():
                return "[矩阵元素]"
            else:
                return "[数值]"
        elif isinstance(obj, str):
            # 根据内容提供更具体的描述
            if "=" in str(obj) and any(op in str(obj) for op in ["*", "+", "-", "**"]):
                return "[数学方程字符串]"
            elif key_name == "symbolic":
                return "[符号表达式]"
            else:
                return "[字符串]"
        else:
            return "[待填充]"

    try:
        golden_answers = query_data.get('metadata', {}).get('golden_answer', [])
        if golden_answers and len(golden_answers) > 0:
            first_answer = golden_answers[0]
            if isinstance(first_answer, dict):
                # 递归创建保留结构的模板
                template = create_template_recursive(first_answer, "root")
                return template, first_answer
        return None, None
    except Exception as e:
        print(f"提取golden_answer模板失败: {e}")
        return None, None


def extract_augmented_answer_template(query_data):
    """从增强版测试案例中提取final_answer作为规范回答模板
    
    专门用于处理augmented_versions中的final_answer结构
    
    Args:
        query_data: 包含augmented_version_data.final_answer的数据字典
        
    Returns:
        tuple: (template, original_data) 或 (None, None)
    """
    def create_template_recursive(obj, key_name=""):
        """递归创建模板，保留嵌套结构"""
        if isinstance(obj, dict):
            template = {}
            for key, value in obj.items():
                template[key] = create_template_recursive(value, key)
            return template
        elif isinstance(obj, list):
            if len(obj) > 0:
                first_item = obj[0]
                if isinstance(first_item, list):
                    # 数值矩阵
                    if all(isinstance(x, (int, float)) for x in first_item):
                        return f"[{len(obj)}x{len(first_item)} 数值矩阵]"
                    else:
                        return [create_template_recursive(first_item, f"{key_name}_item")]
                elif isinstance(first_item, (int, float)):
                    return f"[{len(obj)}个数值的数组]"
                elif isinstance(first_item, str):
                    return f"[{len(obj)}个字符串的数组]"
                else:
                    return [create_template_recursive(first_item, f"{key_name}_item")]
            else:
                return []
        elif isinstance(obj, bool):
            return "[布尔值]"
        elif isinstance(obj, (int, float)):
            return "[数值]"
        elif isinstance(obj, str):
            return "[字符串]"
        else:
            return "[待填充]"

    try:
        # 从增强版数据中获取final_answer
        augmented_data = query_data.get('augmented_version_data', {})
        final_answer = augmented_data.get('final_answer')
        
        if final_answer:
            if isinstance(final_answer, dict):
                # 为复杂结构创建模板
                template = create_template_recursive(final_answer, "root")
                return template, final_answer
            else:
                # 对于简单类型，创建基本模板
                template = create_template_recursive(final_answer, "root")
                return template, final_answer
        
        # 如果没有找到final_answer，尝试从golden_answer中获取
        return extract_golden_answer_template(query_data)
        
    except Exception as e:
        print(f"提取增强版答案模板失败: {e}")
        return None, None


def extract_refined_answer_template(query_data):
    """从精炼版测试案例中提取final_answer作为规范回答模板
    
    专门用于处理refined_versions中的final_answer结构
    
    Args:
        query_data: 包含refined_version_data.final_answer的数据字典
        
    Returns:
        tuple: (template, original_data) 或 (None, None)
    """
    def create_template_recursive(obj, key_name=""):
        """递归创建模板，保留嵌套结构"""
        if isinstance(obj, dict):
            template = {}
            for key, value in obj.items():
                template[key] = create_template_recursive(value, key)
            return template
        elif isinstance(obj, list):
            if len(obj) > 0:
                first_item = obj[0]
                if isinstance(first_item, list):
                    # 数值矩阵
                    if all(isinstance(x, (int, float)) for x in first_item):
                        return f"[{len(obj)}x{len(first_item)} 数值矩阵]"
                    else:
                        return [create_template_recursive(first_item, f"{key_name}_item")]
                elif isinstance(first_item, (int, float)):
                    return f"[{len(obj)}个数值的数组]"
                elif isinstance(first_item, str):
                    return f"[{len(obj)}个字符串的数组]"
                else:
                    return [create_template_recursive(first_item, f"{key_name}_item")]
            else:
                return []
        elif isinstance(obj, bool):
            return "[布尔值]"
        elif isinstance(obj, (int, float)):
            return "[数值]"
        elif isinstance(obj, str):
            return "[字符串]"
        else:
            return "[待填充]"

    try:
        # 从精炼版数据中获取final_answer
        refined_data = query_data.get('refined_version_data', {})
        final_answer = refined_data.get('final_answer')
        
        if final_answer:
            if isinstance(final_answer, dict):
                # 为复杂结构创建模板
                template = create_template_recursive(final_answer, "root")
                return template, final_answer
            else:
                # 对于简单类型，创建基本模板
                template = create_template_recursive(final_answer, "root")
                return template, final_answer
        
        # 如果没有找到final_answer，尝试从golden_answer中获取
        return extract_golden_answer_template(query_data)
        
    except Exception as e:
        print(f"提取精炼版答案模板失败: {e}")
        return None, None


def extract_structured_answer_from_response(response_content):
    """从模型回答中提取结构化的JSON答案"""
    if not response_content:
        return None

    try:
        # 预处理：移除LaTeX包装和其他格式（不丢内容）
        processed_content = _preprocess_response_content(response_content)

        candidates = []  # 收集所有可解析JSON候选

        def _try_parse_and_collect(json_str: str, source_tag: str):
            if not json_str:
                return
            try:
                # 先尝试直接解析，不做清洗，避免破坏本就合法的JSON
                try:
                    parsed = json.loads(json_str)
                    cleaned_len_ref = len(json_str)
                except json.JSONDecodeError:
                    cleaned_json = _clean_json_string(json_str)
                    parsed = json.loads(cleaned_json)
                    cleaned_len_ref = len(cleaned_json)
                # 为候选打分：优先字典，其次列表；更长的字符串、更多键、嵌套更深者优先
                score = 0
                try:
                    score += cleaned_len_ref  # 长度
                except Exception:
                    pass
                if isinstance(parsed, dict):
                    score += 5000
                    try:
                        score += len(parsed) * 50
                    except Exception:
                        pass
                elif isinstance(parsed, list):
                    score += 3000
                    try:
                        score += len(parsed) * 10
                    except Exception:
                        pass
                candidates.append((score, parsed, source_tag))
            except json.JSONDecodeError:
                pass

        # 方法1：优先解析代码块中的完整内容（而不是第一个最小花括号）
        codeblock_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        code_blocks = re.findall(codeblock_pattern, processed_content, re.IGNORECASE)
        for block in code_blocks:
            # 直接尝试把整个代码块作为JSON解析
            _try_parse_and_collect(block, 'codeblock_full')
            # 如果失败，从代码块内部提取可能的JSON对象（按配对花括号）
            potential_in_block = _extract_potential_json_objects(block)
            potential_in_block.sort(key=len, reverse=True)
            for js in potential_in_block:
                _try_parse_and_collect(js, 'codeblock_potential')

        # 方法2：LaTeX boxed 包裹
        boxed_pattern = r'\$\\boxed\{([\s\S]*?)\}\$'
        boxed_matches = re.findall(boxed_pattern, processed_content)
        for match in boxed_matches:
            cleaned_match = _clean_latex_json(match)
            _try_parse_and_collect(cleaned_match, 'latex_boxed')
            # 再从内部提取可能的JSON
            potential_in_box = _extract_potential_json_objects(cleaned_match)
            potential_in_box.sort(key=len, reverse=True)
            for js in potential_in_box:
                _try_parse_and_collect(js, 'latex_potential')

        # 方法3：从全文中按花括号配对提取候选
        potential_jsons = _extract_potential_json_objects(processed_content)
        potential_jsons.sort(key=len, reverse=True)
        for json_str in potential_jsons:
            _try_parse_and_collect(json_str, 'fulltext_potential')

        # 选择最佳候选
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            raw_result = candidates[0][1]
            # 对提取的答案进行后处理，修复常见问题
            return _post_process_extracted_answer(raw_result)

        return None

    except Exception as e:
        print(f"提取结构化答案失败: {e}")
        return None


def _post_process_extracted_answer(answer):
    """对提取的答案进行后处理，修复常见的结构和类型问题"""
    if not isinstance(answer, dict):
        return answer

    # 修复缺失的顶层键问题
    # 如果答案看起来像是直接的内容而不是包装在顶层键中，尝试包装它
    processed_answer = _fix_missing_top_level_key(answer)

    # 修复字符串数值和布尔值问题
    processed_answer = _normalize_data_types(processed_answer)

    return processed_answer


def _fix_missing_top_level_key(answer):
    """修复缺失顶层键的问题

    某些模型回答可能缺少预期的顶层键，如"关键量"等
    """
    if not isinstance(answer, dict):
        return answer

    # 检测是否缺少常见的顶层键
    common_top_keys = ["关键量", "answer", "result", "solution"]

    # 如果答案已经包含这些顶层键之一，则不需要修复
    for key in common_top_keys:
        if key in answer:
            return answer

    # 检查是否答案内容直接包含了预期结构的内部键
    expected_inner_keys = ["解析结果", "代入数值", "四状态(P,V,T)", "功与热(取过程方向为正)", "一致性核查清单"]

    # 如果发现内部键，说明可能缺少"关键量"包装
    found_inner_keys = sum(1 for key in expected_inner_keys if key in answer)
    if found_inner_keys >= 2:  # 如果发现2个或更多内部键，认为需要包装
        return {"关键量": answer}

    return answer


def _normalize_data_types(obj):
    """递归地规范化数据类型，将字符串数值转换为数值，字符串布尔值转换为布尔值"""
    if isinstance(obj, dict):
        return {k: _normalize_data_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_data_types(item) for item in obj]
    elif isinstance(obj, str):
        # 尝试转换字符串数值
        normalized = _try_convert_string_to_number_or_bool(obj)
        return normalized
    else:
        return obj


def _try_convert_string_to_number_or_bool(value):
    """尝试将字符串转换为合适的数值或布尔类型"""
    if not isinstance(value, str):
        return value

    value_stripped = value.strip()

    # 转换布尔值
    if value_stripped.lower() == 'true':
        return True
    elif value_stripped.lower() == 'false':
        return False

    # 转换数值
    try:
        # 处理科学记数法，如 "2.494e-2"
        if 'e' in value_stripped.lower():
            return float(value_stripped)
        # 处理整数
        if '.' not in value_stripped:
            # 但要排除明显不是数字的字符串
            if value_stripped.isdigit() or (value_stripped.startswith('-') and value_stripped[1:].isdigit()):
                return int(value_stripped)
        # 处理浮点数
        else:
            return float(value_stripped)
    except (ValueError, TypeError):
        pass

    # 如果无法转换，返回原字符串
    return value


def _preprocess_response_content(content):
    """预处理响应内容，移除一些常见的格式包装但保留JSON内容"""
    if not content:
        return content
    
    # 不移除任何实际内容，只是清理，保留完整的响应
    # 这样我们可以在各个方法中处理不同的格式
    return content.strip()


def _clean_latex_json(latex_json):
    """清理LaTeX格式的JSON字符串"""
    if not latex_json:
        return latex_json
    
    # 移除LaTeX转义字符
    cleaned = latex_json.replace('\\{', '{').replace('\\}', '}')
    cleaned = cleaned.replace('\\"', '"')
    cleaned = cleaned.replace('\\\\', '\\')
    
    return cleaned


def _clean_json_string(json_str):
    """清理JSON字符串，移除多余的空白和格式字符，并处理无效的表达式"""
    if not json_str:
        return json_str
    
    # 移除开头和结尾的空白字符
    cleaned = json_str.strip()
    
    # 移除可能的引号包装
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    
    # 处理JavaScript表达式（将其转换为字符串）
    cleaned = _fix_javascript_expressions(cleaned)
    
    return cleaned


def _fix_javascript_expressions(json_str):
    """修复JSON中的JavaScript表达式，将其转换为字符串"""
    import re
    
    # 匹配常见的数学表达式模式（如 0.5 * p_0）
    # 查找在JSON值位置的表达式（冒号后面，逗号前面或括号前面）
    patterns = [
        # 数字乘法表达式 (如: 0.5 * p_0)
        (r':\s*([0-9.]+\s*\*\s*[a-zA-Z_][a-zA-Z0-9_]*)', r': "\1"'),
        # 变量表达式 (如: p_0)  
        (r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r': "\1"\2'),
        # 复杂数学表达式
        (r':\s*([^",{}[\]]+\s*[*/+-]\s*[^",{}[\]]+)', r': "\1"'),
    ]
    
    for pattern, replacement in patterns:
        json_str = re.sub(pattern, replacement, json_str)
    
    return json_str


def _extract_potential_json_objects(content):
    """从内容中提取潜在的JSON对象"""
    potential_jsons = []

    # Find all { } pairs that could be complete JSON objects
    start_positions = [i for i, char in enumerate(content) if char == '{']

    for start_pos in start_positions:
        brace_count = 0
        for i in range(start_pos, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = content[start_pos:i+1]
                    # 简单过滤：避免把明显的代码片段（如 function(){}, if(){}) 当作JSON
                    if 'function' in candidate or '=>{' in candidate:
                        break
                    potential_jsons.append(candidate)
                    break

        # 处理不完整的JSON（没有匹配的结束花括号）
        if brace_count > 0:
            incomplete_json = content[start_pos:]
            # 尝试修复不完整的JSON
            repaired_json = _try_repair_incomplete_json(incomplete_json)
            if repaired_json:
                potential_jsons.append(repaired_json)

    return potential_jsons


def _try_repair_incomplete_json(incomplete_json):
    """尝试修复不完整的JSON字符串"""
    try:
        import re
        import json

        # 移除末尾的不完整内容（如 "等等", "..."等）
        cleaned = re.sub(r'\s*(等等|\.\.\.|\.\.\.|…).*$', '', incomplete_json, flags=re.DOTALL)

        # 处理不完整的键值对，特别是类似 "key":  的情况
        # 先检查是否有不完整的键值对
        lines = cleaned.split('\n')
        valid_content = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                valid_content.append(line)
                continue

            # 检查是否是不完整的键值对
            if ':' in stripped:
                # 如果行以 : 结尾或者值部分为空/不完整，移除这行
                if re.search(r':\s*$', stripped) or re.search(r':\s*"[^"]*$', stripped):
                    # 移除这个不完整的键值对，但要检查前面是否有逗号需要移除
                    if valid_content and valid_content[-1].strip().endswith(','):
                        # 移除前一行末尾的逗号
                        valid_content[-1] = valid_content[-1].rstrip().rstrip(',')
                    break

            valid_content.append(line)

        # 重新组合内容
        repaired = '\n'.join(valid_content).strip()

        # 移除末尾可能的多余逗号
        repaired = re.sub(r',(\s*[\]}])', r'\1', repaired)
        repaired = re.sub(r',\s*$', '', repaired)

        # 计算缺失的闭合括号数量
        brace_count = repaired.count('{') - repaired.count('}')
        bracket_count = repaired.count('[') - repaired.count(']')

        # 添加缺失的闭合括号
        repaired += ']' * bracket_count + '}' * brace_count

        # 尝试解析修复后的JSON
        parsed = json.loads(repaired)
        return repaired

    except Exception as e:
        # 如果修复失败，尝试另一种策略：找到最后一个完整的对象/数组
        try:
            import re
            import json

            # 查找所有可能的截断点
            lines = incomplete_json.split('\n')
            for i in range(len(lines) - 1, -1, -1):
                truncated = '\n'.join(lines[:i+1])

                # 尝试平衡括号
                brace_count = truncated.count('{') - truncated.count('}')
                bracket_count = truncated.count('[') - truncated.count(']')

                if brace_count >= 0 and bracket_count >= 0:
                    # 移除末尾的逗号
                    truncated = re.sub(r',\s*$', '', truncated.strip())
                    test_json = truncated + ']' * bracket_count + '}' * brace_count

                    try:
                        json.loads(test_json)
                        return test_json
                    except:
                        continue

            return None

        except Exception:
            return None
