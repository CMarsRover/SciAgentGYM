#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试自动推断工具目录功能

验证 register_tools_to_env 的 auto_infer_from_metadata 参数
是否能根据 metadata 中的 subject 和 topic 自动加载对应的工具
"""

import json
import sys
from pathlib import Path

# 确保使用本地 gym 模块而非 OpenAI Gym
# 必须在任何其他导入之前执行
# 文件位于 gym/test/，向上两级到项目根目录
project_root = Path(__file__).resolve().parent.parent.parent
gym_path = Path(__file__).resolve().parent.parent

# 从 sys.modules 中移除可能存在的 OpenAI gym
for module_name in list(sys.modules.keys()):
    if module_name == 'gym' or module_name.startswith('gym.'):
        del sys.modules[module_name]

# 将项目根目录插入到 sys.path 最前面
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 确保 gym_path 的父目录在 sys.path 中
if str(gym_path.parent) not in sys.path:
    sys.path.insert(0, str(gym_path.parent))


def test_auto_infer_tools():
    """测试自动推断工具目录功能"""
    # 使用显式的模块导入
    import importlib.util
    
    # 直接加载 tool_loader 模块
    tool_loader_path = gym_path / "core" / "tool_loader.py"
    spec = importlib.util.spec_from_file_location("gym.core.tool_loader", tool_loader_path)
    tool_loader = importlib.util.module_from_spec(spec)
    sys.modules["gym.core.tool_loader"] = tool_loader
    spec.loader.exec_module(tool_loader)
    
    # 从加载的模块中获取函数
    infer_toolkits_path_from_metadata = tool_loader.infer_toolkits_path_from_metadata
    load_all_tools_from_directory = tool_loader.load_all_tools_from_directory
    register_tools_to_env = tool_loader.register_tools_to_env
    get_toolkits_index = tool_loader.get_toolkits_index
    
    # 1. 加载测试数据集
    dataset_path = Path(__file__).parent / "dataset" / "refine_merged_questions_augmented.json"
    
    if not dataset_path.exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    print(f"✅ 加载了 {len(cases)} 个测试案例")
    
    # 2. 打印 toolkits 索引
    print("\n=== Toolkits 索引 ===")
    index = get_toolkits_index()
    for subject, topics in index.items():
        print(f"  {subject}/")
        for topic in topics:
            print(f"    └── {topic}/")
    
    # 3. 测试几个具体案例的自动推断
    print("\n=== 测试自动推断工具目录 ===")
    
    # 找出不同 subject/topic 的案例
    seen_combinations = set()
    test_cases = []
    
    for case in cases:
        metadata = case.get("metadata", {})
        subject = metadata.get("subject", "")
        topic = metadata.get("topic", "")
        
        if not subject or not topic:
            continue
        
        combo = (subject, topic)
        if combo not in seen_combinations:
            seen_combinations.add(combo)
            test_cases.append(case)
        
        if len(test_cases) >= 5:  # 只测试前 5 个不同组合
            break
    
    for i, case in enumerate(test_cases, 1):
        metadata = case.get("metadata", {})
        subject = metadata.get("subject", "")
        topic = metadata.get("topic", "")
        case_id = case.get("id", "unknown")
        
        print(f"\n--- 案例 {i}: ID={case_id} ---")
        print(f"  Subject: {subject}")
        print(f"  Topic: {topic}")
        
        # 测试路径推断
        inferred_path = infer_toolkits_path_from_metadata(metadata)
        if inferred_path:
            print(f"  ✅ 推断路径: {inferred_path}")
            
            # 测试加载工具
            protocols, functions = load_all_tools_from_directory(inferred_path)
            print(f"  ✅ 加载了 {len(protocols)} 个工具协议, {len(functions)} 个函数")
            
            if protocols:
                tool_names = [p.get("function", {}).get("name", "?") for p in protocols[:5]]
                print(f"  工具示例: {', '.join(tool_names)}")
        else:
            print(f"  ⚠️ 无法推断路径")
    
    # 4. 测试完整的 register_tools_to_env 流程
    print("\n=== 测试完整注册流程 (auto_infer_from_metadata=True) ===")
    
    # 选择一个 Physics/Optics 的案例
    optics_case = None
    for case in cases:
        metadata = case.get("metadata", {})
        if metadata.get("subject") == "Physics" and metadata.get("topic") == "Optics":
            optics_case = case
            break
    
    if optics_case:
        case_id = optics_case.get("id", "unknown")
        print(f"\n使用案例 ID={case_id} 测试")
        print(f"  Question: {optics_case.get('question', '')[:80]}...")
        
        # 使用 auto_infer_from_metadata=True
        env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
            [], {},  # 空的 protocols 和 function_map
            query_data=optics_case,
            auto_infer_from_metadata=True,
        )
        
        print(f"\n  ✅ 环境创建成功")
        print(f"  ✅ 注册了 {len(tool_registry)} 个工具")
        
        if tool_registry:
            print(f"\n  已注册的工具:")
            for name in list(tool_registry.keys())[:10]:
                print(f"    - {name}")
            if len(tool_registry) > 10:
                print(f"    ... 还有 {len(tool_registry) - 10} 个工具")
        
        # 打印 tools_schema 示例
        if tools_schema:
            print(f"\n  Tools Schema 示例 (第一个工具):")
            print(json.dumps(tools_schema[0], ensure_ascii=False, indent=4))
    else:
        print("⚠️ 未找到 Physics/Optics 案例")
    
    print("\n=== 测试完成 ===")


def test_specific_case(case_id: int = 5):
    """测试特定案例"""
    # 使用显式的模块导入
    import importlib.util
    
    tool_loader_path = gym_path / "core" / "tool_loader.py"
    spec = importlib.util.spec_from_file_location("gym.core.tool_loader", tool_loader_path)
    tool_loader = importlib.util.module_from_spec(spec)
    sys.modules["gym.core.tool_loader"] = tool_loader
    spec.loader.exec_module(tool_loader)
    
    register_tools_to_env = tool_loader.register_tools_to_env
    
    # 加载数据集
    dataset_path = Path(__file__).parent / "dataset" / "refine_merged_questions_augmented.json"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    
    # 查找指定 ID 的案例
    target_case = None
    for case in cases:
        if case.get("id") == case_id:
            target_case = case
            break
    
    if not target_case:
        print(f"❌ 未找到 ID={case_id} 的案例")
        return
    
    metadata = target_case.get("metadata", {})
    print(f"=== 测试案例 ID={case_id} ===")
    print(f"Subject: {metadata.get('subject')}")
    print(f"Topic: {metadata.get('topic')}")
    print(f"Question: {target_case.get('question', '')[:100]}...")
    
    # 使用自动推断
    print("\n正在自动推断并加载工具...")
    env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
        [], {},
        query_data=target_case,
        auto_infer_from_metadata=True,
    )
    
    print(f"\n✅ 成功加载 {len(tool_registry)} 个工具:")
    for name in tool_registry.keys():
        print(f"  - {name}")
    
    return env, tool_instances, tools_schema, tool_registry


if __name__ == "__main__":
    # 运行完整测试
    test_auto_infer_tools()
    
    # 也可以单独测试某个案例
    # test_specific_case(5)  # 测试 ID=5 的 Optics 案例
