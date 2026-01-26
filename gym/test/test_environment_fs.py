"""
测试环境文件系统管理模块
"""

import sys
from pathlib import Path

# 添加项目根目录到路径（文件位于 gym/test/）
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from gym.core.environment_fs import (
    EnvironmentFileSystem,
    get_environment_fs,
    save_mid_result,
    load_mid_result,
    get_mid_result_path
)


def test_environment_fs():
    """测试环境文件系统功能"""
    
    print("=" * 70)
    print("环境文件系统测试")
    print("=" * 70)
    print()
    
    # 测试 1: 基本功能
    print("1. 测试基本保存和加载功能")
    print("-" * 70)
    
    test_data = {
        "test_key": "test_value",
        "number": 42,
        "list": [1, 2, 3]
    }
    
    # 保存
    result = save_mid_result(
        domain="structural_biology",
        filename="test_result",
        data=test_data,
        case_id="test_case_001"
    )
    
    if result["success"]:
        print(f"✅ 保存成功: {result['filepath']}")
        print(f"   文件大小: {result['size']} 字节")
    else:
        print(f"❌ 保存失败: {result['error']}")
        return False
    
    # 加载
    load_result = load_mid_result(
        domain="structural_biology",
        filename="test_result",
        case_id="test_case_001"
    )
    
    if load_result["success"]:
        print(f"✅ 加载成功: {load_result['filepath']}")
        print(f"   数据: {load_result['data']}")
        assert load_result["data"] == test_data, "数据不匹配"
        print("✅ 数据验证通过")
    else:
        print(f"❌ 加载失败: {load_result['error']}")
        return False
    
    print()
    
    # 测试 2: 不同格式
    print("2. 测试不同文件格式")
    print("-" * 70)
    
    # JSON
    json_result = save_mid_result(
        domain="physics",
        filename="test_json",
        data={"format": "json"},
        format="json"
    )
    print(f"JSON: {'✅' if json_result['success'] else '❌'} {json_result.get('filepath', json_result.get('error'))}")
    
    # Pickle
    pickle_result = save_mid_result(
        domain="physics",
        filename="test_pickle",
        data={"format": "pickle", "complex": [1, 2, {"nested": True}]},
        format="pickle"
    )
    print(f"Pickle: {'✅' if pickle_result['success'] else '❌'} {pickle_result.get('filepath', pickle_result.get('error'))}")
    
    # Text
    txt_result = save_mid_result(
        domain="physics",
        filename="test_txt",
        data="This is a text file",
        format="txt"
    )
    print(f"Text: {'✅' if txt_result['success'] else '❌'} {txt_result.get('filepath', txt_result.get('error'))}")
    
    print()
    
    # 测试 3: 路径获取
    print("3. 测试路径获取")
    print("-" * 70)
    
    fs = get_environment_fs()
    domain_dir = fs.get_domain_dir("structural_biology", "test_case_002")
    print(f"✅ 领域目录: {domain_dir}")
    
    filepath = get_mid_result_path(
        domain="structural_biology",
        filename="test_path",
        case_id="test_case_002"
    )
    print(f"✅ 文件路径: {filepath}")
    
    print()
    
    # 测试 4: 列出文件
    print("4. 测试列出文件")
    print("-" * 70)
    
    files = fs.list_results("structural_biology", "test_case_001")
    print(f"✅ 找到 {len(files)} 个文件:")
    for f in files[:5]:  # 只显示前5个
        print(f"   - {Path(f).name}")
    if len(files) > 5:
        print(f"   ... 还有 {len(files) - 5} 个文件")
    
    print()
    
    # 测试 5: 删除文件
    print("5. 测试删除文件")
    print("-" * 70)
    
    delete_result = fs.delete_result(
        domain="structural_biology",
        filename="test_result",
        case_id="test_case_001"
    )
    if delete_result["success"]:
        print(f"✅ 删除成功: {delete_result.get('deleted')}")
    else:
        print(f"❌ 删除失败: {delete_result.get('error')}")
    
    # 验证文件已删除
    load_after_delete = load_mid_result(
        domain="structural_biology",
        filename="test_result",
        case_id="test_case_001"
    )
    if not load_after_delete["success"]:
        print("✅ 确认文件已删除")
    else:
        print("❌ 文件仍然存在")
    
    print()
    
    # 测试 6: 清理测试文件
    print("6. 清理测试文件")
    print("-" * 70)
    
    test_files = [
        ("physics", "test_json", None),
        ("physics", "test_pickle", None),
        ("physics", "test_txt", None),
    ]
    
    for domain, filename, case_id in test_files:
        fs.delete_result(domain, filename, case_id)
    
    print("✅ 清理完成")
    
    print()
    print("=" * 70)
    print("✅ 所有测试通过！")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        success = test_environment_fs()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
