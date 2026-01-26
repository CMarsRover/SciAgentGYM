"""
测试环境注册功能

验证从数据集中加载案例时，环境是否能正确：
1. 提取 case_id 和 domain
2. 初始化环境文件系统
3. 正确组织目录结构
"""

import sys
import json
from pathlib import Path

# 添加项目根目录到路径（文件位于 gym/test/）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from gym.core.tool_loader import load_tools_for_case, register_tools_to_env
from gym.core.environment_fs import get_environment_fs


def test_environment_registration_from_dataset():
    """测试从数据集加载案例并注册环境"""
    
    print("=" * 70)
    print("环境注册功能测试")
    print("=" * 70)
    print()
    
    # 加载数据集
    core_dir = Path(__file__).resolve().parent
    dataset_path = core_dir / "dataset" / "refine_merged_questions_augmented.json"
    
    if not dataset_path.exists():
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return False
    
    print(f"📂 加载数据集: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    
    if not isinstance(cases, list) or not cases:
        print("❌ 数据集内容为空或格式不是列表")
        return False
    
    print(f"✅ 成功加载数据集，共 {len(cases)} 个案例\n")
    
    # 测试统计
    stats = {
        'total_tested': 0,
        'successful_registrations': 0,
        'failed_registrations': 0,
        'cases_with_tools': 0,
        'cases_without_tools': 0,
        'file_system_initialized': 0,
        'details': []
    }
    
    # 测试前10个有工具的案例
    test_count = 0
    max_test = 10
    
    print("开始测试环境注册...\n")
    
    for idx, case in enumerate(cases, start=1):
        if test_count >= max_test:
            break
        
        case_id = case.get("id", f"case_{idx}")
        metadata = case.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        
        # 提取信息
        subject = metadata.get("subject", "")
        topic = metadata.get("topic", "")
        
        print(f"📋 测试案例 {idx}: ID={case_id}, Subject={subject}, Topic={topic}")
        
        # 尝试加载工具
        try:
            tool_protocols, function_map = load_tools_for_case(case)
            
            if not tool_protocols:
                print(f"   ⚠️  案例没有工具，跳过")
                stats['cases_without_tools'] += 1
                continue
            
            stats['cases_with_tools'] += 1
            stats['total_tested'] += 1
            test_count += 1
            
            print(f"   ✅ 找到 {len(tool_protocols)} 个工具")
            
            # 注册环境
            try:
                env, tool_instances, tools_schema, tool_registry = register_tools_to_env(
                    tool_protocols,
                    function_map,
                    query_data=case  # 传递 query_data 自动提取信息
                )
                
                if env is None:
                    print(f"   ❌ 环境创建失败")
                    stats['failed_registrations'] += 1
                    stats['details'].append({
                        'case_id': case_id,
                        'status': 'failed',
                        'reason': '环境创建失败'
                    })
                    continue
                
                print(f"   ✅ 环境注册成功")
                print(f"      注册工具数: {len(tool_instances)}")
                
                # 检查文件系统
                if hasattr(env, 'file_system') and env.file_system is not None:
                    print(f"   ✅ 文件系统已初始化")
                    stats['file_system_initialized'] += 1
                    
                    # 检查 case_id 和 domain
                    env_case_id = getattr(env, '_case_id', None)
                    env_domain = getattr(env, '_domain', None)
                    
                    print(f"      环境 case_id: {env_case_id}")
                    print(f"      环境 domain: {env_domain}")
                    
                    # 测试获取目录
                    if env_domain:
                        mid_result_dir = env.get_mid_result_dir()
                        print(f"      中间结果目录: {mid_result_dir}")
                        
                        # 验证目录是否存在
                        fs = get_environment_fs()
                        domain_dir = fs.get_domain_dir(env_domain, env_case_id)
                        if domain_dir.exists():
                            print(f"   ✅ 目录已创建: {domain_dir}")
                        else:
                            print(f"   ⚠️  目录不存在: {domain_dir}")
                    else:
                        print(f"   ⚠️  未提取到 domain")
                else:
                    print(f"   ❌ 文件系统未初始化")
                
                stats['successful_registrations'] += 1
                stats['details'].append({
                    'case_id': case_id,
                    'status': 'success',
                    'tools_count': len(tool_instances),
                    'env_case_id': getattr(env, '_case_id', None),
                    'env_domain': getattr(env, '_domain', None)
                })
                
            except Exception as e:
                print(f"   ❌ 环境注册失败: {e}")
                import traceback
                traceback.print_exc()
                stats['failed_registrations'] += 1
                stats['details'].append({
                    'case_id': case_id,
                    'status': 'failed',
                    'reason': str(e)
                })
        
        except Exception as e:
            print(f"   ❌ 加载工具失败: {e}")
            stats['failed_registrations'] += 1
            stats['details'].append({
                'case_id': case_id,
                'status': 'failed',
                'reason': f'加载工具失败: {str(e)}'
            })
        
        print()
    
    # 打印统计结果
    print("=" * 70)
    print("测试结果统计")
    print("=" * 70)
    print(f"总测试案例数: {stats['total_tested']}")
    print(f"有工具的案例: {stats['cases_with_tools']}")
    print(f"无工具的案例: {stats['cases_without_tools']}")
    print(f"成功注册: {stats['successful_registrations']}")
    print(f"注册失败: {stats['failed_registrations']}")
    print(f"文件系统初始化: {stats['file_system_initialized']}")
    
    # 显示详细信息
    if stats['details']:
        print("\n详细信息:")
        print("-" * 70)
        for detail in stats['details']:
            if detail['status'] == 'success':
                print(f"✅ {detail['case_id']}: {detail['tools_count']} 个工具, "
                      f"domain={detail.get('env_domain')}, case_id={detail.get('env_case_id')}")
            else:
                print(f"❌ {detail['case_id']}: {detail.get('reason', '未知错误')}")
    
    print("\n" + "=" * 70)
    if stats['failed_registrations'] == 0 and stats['successful_registrations'] > 0:
        print("✅ 所有测试通过！环境注册功能正常工作")
        return True
    else:
        print(f"⚠️  部分测试失败，请查看上述详情")
        return False


def test_file_system_integration():
    """测试文件系统集成功能"""
    
    print("\n" + "=" * 70)
    print("文件系统集成测试")
    print("=" * 70)
    print()
    
    # 加载一个测试案例
    core_dir = Path(__file__).resolve().parent
    dataset_path = core_dir / "dataset" / "refine_merged_questions_augmented.json"
    
    with dataset_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    
    # 找一个有工具的案例
    test_case = None
    for case in cases:
        try:
            tool_protocols, _ = load_tools_for_case(case)
            if tool_protocols:
                test_case = case
                break
        except:
            continue
    
    if not test_case:
        print("❌ 未找到有工具的测试案例")
        return False
    
    print(f"📋 使用测试案例: ID={test_case.get('id')}")
    
    # 注册环境
    tool_protocols, function_map = load_tools_for_case(test_case)
    env, _, _, _ = register_tools_to_env(
        tool_protocols,
        function_map,
        query_data=test_case
    )
    
    if not env:
        print("❌ 环境创建失败")
        return False
    
    # 测试文件系统功能
    print("\n测试文件系统功能:")
    print("-" * 70)
    
    # 1. 测试保存
    test_data = {"test": "data", "timestamp": "2024-01-01"}
    save_result = env.file_system.save_result(
        domain=env._domain or "test",
        filename="test_file",
        data=test_data,
        case_id=env._case_id
    )
    
    if save_result["success"]:
        print(f"✅ 保存成功: {save_result['filepath']}")
    else:
        print(f"❌ 保存失败: {save_result['error']}")
        return False
    
    # 2. 测试加载
    load_result = env.file_system.load_result(
        domain=env._domain or "test",
        filename="test_file",
        case_id=env._case_id
    )
    
    if load_result["success"]:
        print(f"✅ 加载成功: {load_result['data']}")
        if load_result["data"] == test_data:
            print("✅ 数据验证通过")
        else:
            print("❌ 数据不匹配")
            return False
    else:
        print(f"❌ 加载失败: {load_result['error']}")
        return False
    
    # 3. 测试目录结构
    domain_dir = env.file_system.get_domain_dir(
        env._domain or "test",
        env._case_id
    )
    print(f"✅ 目录路径: {domain_dir}")
    
    # 4. 清理测试文件
    delete_result = env.file_system.delete_result(
        domain=env._domain or "test",
        filename="test_file",
        case_id=env._case_id
    )
    if delete_result["success"]:
        print(f"✅ 清理成功")
    
    print("\n" + "=" * 70)
    print("✅ 文件系统集成测试通过！")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        # 测试环境注册
        registration_success = test_environment_registration_from_dataset()
        
        # 测试文件系统集成
        fs_success = test_file_system_integration()
        
        if registration_success and fs_success:
            print("\n✅ 所有测试通过！")
            sys.exit(0)
        else:
            print("\n⚠️  部分测试失败")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
