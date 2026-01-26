#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具注册测试模块

测试所有数据集文件中的工具注册情况，验证工具加载和注册流程的正确性。
"""
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gym.core.tool_loader import load_tools_for_case, register_tools_to_env


def test_all_tools_registration(
    dataset_dir: str = "gym/dataset",
    output_file: str = "gym/dataset/tool_registration_test_results.json"
):
    """
    测试所有数据集文件中的工具注册情况。
    
    参数:
        dataset_dir: 数据集目录路径
        output_file: 输出结果文件路径
    
    返回:
        dict: 测试结果统计
    """
    dataset_path = project_root / dataset_dir
    output_path = project_root / output_file
    
    if not dataset_path.exists():
        print(f"❌ 数据集目录不存在: {dataset_path}")
        return
    
    # 查找所有 JSON 文件
    json_files = list(dataset_path.glob("*.json"))
    if not json_files:
        print(f"❌ 在 {dataset_path} 中未找到 JSON 文件")
        return
    
    print(f"📋 找到 {len(json_files)} 个数据集文件")
    print("=" * 80)
    
    all_results = {
        "test_timestamp": str(Path(__file__).stat().st_mtime),
        "dataset_dir": str(dataset_dir),
        "total_files": len(json_files),
        "files": []
    }
    
    total_cases = 0
    total_success = 0
    total_failed = 0
    
    for json_file in json_files:
        print(f"\n📂 处理文件: {json_file.name}")
        file_result = {
            "filename": json_file.name,
            "total_cases": 0,
            "success_cases": 0,
            "failed_cases": 0,
            "failed_details": []
        }
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                cases = json.load(f)
            
            if not isinstance(cases, list):
                print(f"  ⚠️ 文件格式错误：顶层不是列表")
                file_result["error"] = "文件格式错误：顶层不是列表"
                all_results["files"].append(file_result)
                continue
            
            file_result["total_cases"] = len(cases)
            total_cases += len(cases)
            
            print(f"  📊 共 {len(cases)} 个测试案例")
            
            for idx, test_case in enumerate(cases):
                case_id = test_case.get("id") or test_case.get("metadata", {}).get("case_id") or idx + 1
                
                try:
                    # 尝试加载工具
                    tool_protocols, function_map = load_tools_for_case(test_case)
                    
                    if not tool_protocols:
                        raise ValueError("未找到任何工具协议")
                    
                    if not function_map:
                        raise ValueError("未找到任何工具函数")
                    
                    # 尝试注册到环境
                    env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
                        tool_protocols,
                        function_map
                    )
                    
                    if not tool_instances:
                        raise ValueError("工具注册失败：未创建任何工具实例")
                    
                    total_success += 1
                    file_result["success_cases"] += 1
                    
                    if (idx + 1) % 100 == 0:
                        print(f"    ✓ 已处理 {idx + 1}/{len(cases)} 个案例")
                
                except Exception as e:
                    total_failed += 1
                    file_result["failed_cases"] += 1
                    
                    # 记录失败详情
                    metadata = test_case.get("metadata") or {}
                    failed_detail = {
                        "case_id": case_id,
                        "subject": metadata.get("subject"),
                        "topic": metadata.get("topic"),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "usage_tool_protocol_count": len(test_case.get("usage_tool_protocol", [])),
                    }
                    
                    # 记录工具路径信息
                    tool_paths = []
                    for tool in test_case.get("usage_tool_protocol", []):
                        if isinstance(tool, dict):
                            addl = tool.get("additionalProperties") or {}
                            tool_path = addl.get("function_path")
                            if tool_path:
                                tool_paths.append(tool_path)
                    failed_detail["tool_paths"] = tool_paths
                    
                    file_result["failed_details"].append(failed_detail)
                    
                    if file_result["failed_cases"] <= 5:  # 只打印前5个失败案例
                        print(f"    ❌ 案例 {case_id} 失败: {str(e)[:100]}")
            
            print(f"  ✅ 成功: {file_result['success_cases']}, ❌ 失败: {file_result['failed_cases']}")
            
        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")
            file_result["error"] = str(e)
        
        all_results["files"].append(file_result)
    
    # 汇总统计
    all_results["summary"] = {
        "total_cases": total_cases,
        "total_success": total_success,
        "total_failed": total_failed,
        "success_rate": f"{(total_success / total_cases * 100):.2f}%" if total_cases > 0 else "0%"
    }
    
    # 保存结果到 JSON 文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "=" * 80)
    print("📊 测试汇总:")
    print(f"  总案例数: {total_cases}")
    print(f"  ✅ 成功: {total_success}")
    print(f"  ❌ 失败: {total_failed}")
    print(f"  成功率: {all_results['summary']['success_rate']}")
    print(f"  结果已保存到: {output_path}")
    print("=" * 80)
    
    return all_results


def test_single_case(dataset_file: str = None):
    """
    测试单个案例的工具注册。
    
    参数:
        dataset_file: 数据集文件路径，默认使用 refine_merged_questions_augmented.json
    """
    if dataset_file is None:
        dataset_path = project_root / "gym" / "dataset" / "refine_merged_questions_augmented.json"
    else:
        dataset_path = Path(dataset_file)
    
    if not dataset_path.exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
    
    if not test_cases:
        print("❌ 数据集为空")
        return
    
    print(f"📋 测试单个案例 (ID: {test_cases[0].get('id', 'unknown')})")
    
    # 步骤1: 加载工具协议和函数
    tool_protocols, function_map = load_tools_for_case(test_cases[0])
    
    # 步骤2: 注册到环境
    env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
        tool_protocols,
        function_map
    )
    
    print(f"✅ 成功注册 {len(tool_instances)} 个工具")
    print(f"📝 tool_instances: {tool_instances}")
    print(f"📝 tools_schema: {tools_schema}")
    print(f"📝 工具列表: {list(tool_registry.keys())}")
    
    return env, tool_instances, tools_schema, tool_registry


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="工具注册测试")
    parser.add_argument("--test-all", action="store_true", help="测试所有数据集文件")
    parser.add_argument("--dataset-dir", default="gym/dataset", help="数据集目录")
    parser.add_argument("--output", default="gym/dataset/tool_registration_test_results.json", help="输出文件路径")
    parser.add_argument("--single", action="store_true", help="测试单个案例")
    parser.add_argument("--file", default=None, help="指定数据集文件（用于 --single）")
    
    args = parser.parse_args()
    
    if args.test_all:
        # 运行完整测试
        test_all_tools_registration(dataset_dir=args.dataset_dir, output_file=args.output)
    elif args.single:
        # 单个案例测试
        test_single_case(dataset_file=args.file)
    else:
        # 默认运行单个案例测试
        test_single_case()
