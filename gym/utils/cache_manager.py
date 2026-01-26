"""
缓存管理模块
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from gym.config.config import TOOL_TRACE_SUFFIX
from gym.config.dataset_config import get_trace_root


def _resolve_mode_folder(use_tools: bool = True, mode_name: Optional[str] = None) -> str:
    base = mode_name or ("with_tools" if use_tools else "without_tools")
    if use_tools:
        suffix = (TOOL_TRACE_SUFFIX or "").strip()
        if suffix and not base.endswith(suffix):
            base = f"{base}{suffix}"
    return base


MODE_VARIANTS = [
    ("with_tools", True, None),
    ("without_tools", False, None),
    ("with_all_tools", True, "with_all_tools"),
]


def check_trace_cache(case_id, model_name: str, use_tools: bool = True, mode_name: Optional[str] = None) -> bool:
    """检查指定案例和模型的trace文件是否已存在"""
    mode_folder = _resolve_mode_folder(use_tools, mode_name)
    trace_path = get_trace_root() / model_name / mode_folder / f"{case_id}_trace.json"
    return trace_path.exists()


def get_cache_status_for_model(model_name: str, test_cases: list, use_tools: bool = True, mode_name: Optional[str] = None) -> dict:
    """获取指定模型的缓存状态"""
    cached_cases = []
    missing_cases = []

    for case in test_cases:
        case_id = case['id']
        if check_trace_cache(case_id, model_name, use_tools, mode_name=mode_name):
            cached_cases.append(case_id)
        else:
            missing_cases.append(case_id)

    return {
        'cached': cached_cases,
        'missing': missing_cases,
        'cached_count': len(cached_cases),
        'missing_count': len(missing_cases),
        'total_count': len(test_cases),
        'cache_rate': len(cached_cases) / len(test_cases) if test_cases else 0
    }


def get_cache_statistics():
    """获取缓存统计信息"""
    trace_root = get_trace_root()
    if not trace_root.exists():
        return {}

    cache_stats = {}
    total_traces = 0

    for model_dir in trace_root.iterdir():
        if not model_dir.is_dir():
            continue

        model_stats = {key: 0 for key, _, _ in MODE_VARIANTS}
        model_stats['total'] = 0

        for key, flag, override in MODE_VARIANTS:
            mode_folder = _resolve_mode_folder(flag, override)
            mode_path = model_dir / mode_folder
            if mode_path.exists() and mode_path.is_dir():
                trace_files = [f for f in mode_path.iterdir() if f.is_file() and f.name.endswith('_trace.json')]
                model_stats[key] = len(trace_files)
                model_stats['total'] += len(trace_files)
                total_traces += len(trace_files)

        legacy_files = [
            f for f in model_dir.iterdir()
            if f.is_file() and (f.name.endswith('_trace.json') or f.name.endswith('_notool_trace.json'))
        ]
        if legacy_files:
            model_stats['legacy'] = len(legacy_files)
            model_stats['total'] += len(legacy_files)
            total_traces += len(legacy_files)

        cache_stats[model_dir.name] = model_stats

    cache_stats['_total'] = total_traces
    return cache_stats


def clear_cache(model_name=None, case_ids=None, use_tools=None):
    """清除缓存"""
    trace_root = get_trace_root()
    if not trace_root.exists():
        print(f"没有找到{trace_root}目录")
        return

    deleted_count = 0

    if model_name is None:
        if case_ids is None:
            shutil.rmtree(trace_root, ignore_errors=True)
            trace_root.mkdir(parents=True, exist_ok=True)
            print("✅ 已清除所有缓存")
        else:
            for model_dir in trace_root.iterdir():
                if model_dir.is_dir():
                    deleted_count += _clear_cases_in_model(model_dir, case_ids, use_tools)
            print(f"✅ 已清除 {deleted_count} 个指定案例的缓存文件")
    else:
        model_path = trace_root / model_name
        if not model_path.exists():
            print(f"模型 {model_name} 没有缓存")
            return

        if case_ids is None:
            if use_tools is None:
                shutil.rmtree(model_path, ignore_errors=True)
                print(f"✅ 已清除模型 {model_name} 的所有缓存")
            else:
                mode_folder = _resolve_mode_folder(use_tools)
                mode_path = model_path / mode_folder
                if mode_path.exists():
                    shutil.rmtree(mode_path, ignore_errors=True)
                    print(f"✅ 已清除模型 {model_name} 的{'使用工具' if use_tools else '不使用工具'}模式缓存")
                else:
                    print(f"模型 {model_name} 没有{'使用工具' if use_tools else '不使用工具'}模式的缓存")
        else:
            deleted_count = _clear_cases_in_model(model_path, case_ids, use_tools)
            mode_desc = "所有模式" if use_tools is None else ("使用工具" if use_tools else "不使用工具")
            print(f"✅ 已清除模型 {model_name} 的 {deleted_count} 个指定案例缓存 ({mode_desc})")


def _clear_cases_in_model(model_path: Path, case_ids, use_tools) -> int:
    """清除指定模型路径下的指定案例缓存"""
    deleted_count = 0
    model_path = Path(model_path)

    if use_tools is None:
        for key, flag, override in MODE_VARIANTS:
            mode_folder = _resolve_mode_folder(flag, override)
            mode_path = model_path / mode_folder
            if not mode_path.exists():
                continue
            for case_id in case_ids:
                trace_file = mode_path / f"{case_id}_trace.json"
                if trace_file.exists():
                    trace_file.unlink()
                    deleted_count += 1

        for case_id in case_ids:
            for suffix in ['_trace.json', '_notool_trace.json']:
                trace_file = model_path / suffix
                if trace_file.exists():
                    trace_file.unlink()
                    deleted_count += 1
    else:
        mode_path = model_path / _resolve_mode_folder(use_tools)
        if mode_path.exists():
            for case_id in case_ids:
                trace_file = mode_path / f"{case_id}_trace.json"
                if trace_file.exists():
                    trace_file.unlink()
                    deleted_count += 1

    return deleted_count


def cache_management_menu(test_cases):
    """缓存管理菜单"""
    while True:
        print("\n=== 缓存管理 ===")

        cache_stats = get_cache_statistics()
        if cache_stats:
            print("📊 当前缓存统计:")
            for model, stats in cache_stats.items():
                if model == '_total':
                    continue
                if isinstance(stats, dict):
                    total = stats.get('total', 0)
                    with_tools = stats.get('with_tools', 0)
                    without_tools = stats.get('without_tools', 0)
                    with_all_tools = stats.get('with_all_tools', 0)
                    legacy = stats.get('legacy', 0)

                    print(f"  {model}: {total} 个缓存文件")
                    if with_tools:
                        print(f"    ├─ 使用工具: {with_tools} 个")
                    if without_tools:
                        print(f"    ├─ 不使用工具: {without_tools} 个")
                    if with_all_tools:
                        print(f"    ├─ 学科聚合工具: {with_all_tools} 个")
                    if legacy:
                        print(f"    └─ 旧格式: {legacy} 个")
                else:
                    print(f"  {model}: {stats} 个缓存文件")
            print(f"  总计: {cache_stats.get('_total', 0)} 个缓存文件")
        else:
            print("📦 暂无缓存文件")

        print("\n缓存管理选项:")
        print("1. 查看详细缓存信息")
        print("2. 清除所有缓存")
        print("3. 清除指定模型的缓存")
        print("4. 清除指定案例的缓存")
        print("5. 返回主菜单")

        choice = input("请选择操作 (1-5): ").strip()

        if choice == "1":
            show_detailed_cache_info()
        elif choice == "2":
            confirm = input("⚠️ 确认清除所有缓存？(y/N): ").strip().lower()
            if confirm == 'y':
                clear_cache()
        elif choice == "3":
            from gym.utils.client_manager import list_models
            models = list_models()
            print(f"\n可用模型: {models}")
            model_name = input("请输入要清除缓存的模型名称: ").strip()
            if model_name:
                confirm = input(f"⚠️ 确认清除模型 {model_name} 的所有缓存？(y/N): ").strip().lower()
                if confirm == 'y':
                    clear_cache(model_name)
        elif choice == "4":
            print(f"\n可用案例ID: {[case['id'] for case in test_cases]}")
            case_ids_input = input("请输入要清除的案例ID（多个用逗号分隔）: ").strip()
            try:
                case_ids = [int(x.strip()) for x in case_ids_input.split(',') if x.strip()]
                if case_ids:
                    model_name = input("请输入模型名称（留空表示所有模型）: ").strip() or None
                    confirm = input(f"⚠️ 确认清除案例 {case_ids} 的缓存？(y/N): ").strip().lower()
                    if confirm == 'y':
                        clear_cache(model_name, case_ids)
            except ValueError:
                print("❌ 案例ID格式错误")
        elif choice == "5":
            break
        else:
            print("❌ 无效选择")


def show_detailed_cache_info():
    """显示详细的缓存信息"""
    trace_root = get_trace_root()
    if not trace_root.exists():
        print(f"没有找到{trace_root}目录")
        return

    print("\n📋 详细缓存信息:")

    for model_dir in trace_root.iterdir():
        if not model_dir.is_dir():
            continue

        print(f"\n🤖 模型: {model_dir.name}")

        has_new_structure = False
        for key, flag, override in MODE_VARIANTS:
            mode_folder = _resolve_mode_folder(flag, override)
            mode_path = model_dir / mode_folder
            if not mode_path.exists() or not mode_path.is_dir():
                continue
            has_new_structure = True
            trace_files = [f for f in mode_path.iterdir() if f.is_file() and f.name.endswith('_trace.json')]
            mode_name = {
                "with_tools": "使用工具",
                "without_tools": "不使用工具",
                "with_all_tools": "with_all_tools",
            }.get(key, key)
            print(f"  📁 {mode_name} ({mode_folder}):")

            if not trace_files:
                print("    (无缓存文件)")
                continue

            for trace_path in sorted(trace_files, key=lambda p: p.name):
                _show_trace_file_info(trace_path, trace_path.name, "    ")

        legacy_files = [
            f for f in model_dir.iterdir()
            if f.is_file() and (f.name.endswith('_trace.json') or f.name.endswith('_notool_trace.json'))
        ]
        if legacy_files:
            if has_new_structure:
                print("  📁 旧格式文件:")
            for trace_path in sorted(legacy_files, key=lambda p: p.name):
                prefix = "    " if has_new_structure else "  "
                _show_trace_file_info(trace_path, trace_path.name, prefix)

        if not has_new_structure and not legacy_files:
            print("  (无缓存文件)")


def _show_trace_file_info(trace_path: Path, trace_file: str, prefix: str = ""):
    """显示单个轨迹文件的信息"""
    try:
        stat = trace_path.stat()
        file_size = stat.st_size
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        case_id = trace_file.replace('_trace.json', '').replace('_notool_trace.json', '')
        try:
            with trace_path.open('r', encoding='utf-8') as f:
                trace_data = json.load(f)
            rounds = trace_data.get('rounds', '?')
            has_answer = 'model_structured_answer' in trace_data
            mode = trace_data.get('mode', '未知模式')
            print(f"{prefix}📄 案例{case_id} ({mode}): {file_size/1024:.1f}KB, {rounds}轮, {'有结构化答案' if has_answer else '无结构化答案'}, {mod_time}")
        except Exception:
            print(f"{prefix}📄 案例{case_id}: {file_size/1024:.1f}KB, 无法读取内容, {mod_time}")
    except Exception as e:
        print(f"{prefix}❌ 文件错误 {trace_file}: {e}")

