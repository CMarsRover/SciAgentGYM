import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup, DihedralGroup
from typing import Set, List, Tuple, Union, Callable
import warnings

# ===========================
# 核心数学函数：验证群覆盖定理
# ===========================
def math_func(
    G: Union[Set, List],
    A: Set,
    B: Set,
    multiply: Callable[[any, any], any],
    identity: any = None
) -> dict:
    
    G_set = set(G)
    A_set = set(A)
    B_set = set(B)

    # 输入验证
    if not A_set.issubset(G_set) or not B_set.issubset(G_set):
        raise ValueError("A 和 B 必须是 G 的子集")
    if len(A_set) == 0 or len(B_set) == 0:
        raise ValueError("A 和 B 必须是非空集合")

    # 条件判断
    size_condition = len(A_set) + len(B_set) > len(G_set)

    # 计算 AB = {ab | a ∈ A, b ∈ B}
    AB = set()
    for a in A_set:
        for b in B_set:
            ab = multiply(a, b)
            AB.add(ab)

    # 检查 AB 是否等于 G
    covers_G = AB == G_set

    # 额外：尝试找一个未被覆盖的元素
    missing = G_set - AB

    return {
        "condition_met": size_condition,
        "A_size": len(A_set),
        "B_size": len(B_set),
        "G_size": len(G_set),
        "AB_size": len(AB),
        "covers_G": covers_G,
        "AB_set": AB,
        "missing_elements": missing
    }

# ===========================
# 通用编码函数：计算子集乘积 AB
# ===========================
def coding_func(
    A: Set,
    B: Set,
    multiply: Callable[[any, any], any]
) -> Set:
    
    AB = set()
    for a in A:
        for b in B:
            AB.add(multiply(a, b))
    return AB

# ===========================
# 可视化函数
# ===========================
def visual_func(
    result: dict,
    title: str = "Group Subset Covering Analysis"
):
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：集合大小对比
    labels = ['|A|', '|B|', '|G|', '|A|+|B|']
    sizes = [
        result['A_size'],
        result['B_size'],
        result['G_size'],
        result['A_size'] + result['B_size']
    ]
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
    ax[0].bar(labels, sizes, color=colors)
    ax[0].set_title("Set Sizes Comparison")
    ax[0].set_ylabel("Cardinality")
    for i, v in enumerate(sizes):
        ax[0].text(i, v + 0.1 * result['G_size'], str(v), ha='center', fontsize=10)

    # 添加 |G| 水平线
    ax[0].axhline(y=result['G_size'], color='r', linestyle='--', label=f'|G| = {result["G_size"]}')
    ax[0].legend()

    # 右图：覆盖情况（Venn 风格示意）
    ax[1].text(0.1, 0.8, f"|A| + |B| > |G|: {result['condition_met']}", fontsize=12, color='blue')
    ax[1].text(0.1, 0.6, f"AB = G: {result['covers_G']}", fontsize=12, color='green' if result['covers_G'] else 'red')
    ax[1].text(0.1, 0.4, f"|AB| = {result['AB_size']}", fontsize=12)
    if result['missing_elements']:
        ax[1].text(0.1, 0.2, f"Missing: {len(result['missing_elements'])} elem", fontsize=10, color='orange')
    ax[1].axis('off')
    ax[1].set_title("Coverage Result")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ===========================
# 示例：使用对称群 S3
# ===========================
if __name__ == "__main__":
    # # 构造 S3: 对称群 on 3 elements
    # id = Permutation(3)
    # a = Permutation(0, 1)      # (0 1)
    # b = Permutation(1, 2)      # (1 2)
    # G_perms = list(PermutationGroup(id, a, b).generate())

    # # 映射到标签
    # perm_to_label = {p: f"p{i}" for i, p in enumerate(G_perms)}
    # label_to_perm = {v: k for k, v in perm_to_label.items()}

    # # 定义群乘法（注意：Permutation 左作用，p1*p2 表示先 p2 后 p1）
    # def perm_mult(p1, p2):
    #     return p1 * p2  # sympy 中是左作用

    # # 选择子集 A 和 B，使得 |A| + |B| > |G| = 6
    # A_labels = ['p0', 'p1', 'p2', 'p3']  # |A| = 4
    # B_labels = ['p0', 'p1', 'p2']        # |B| = 3
    # A_perms = {label_to_perm[label] for label in A_labels}
    # B_perms = {label_to_perm[label] for label in B_labels}

    # # 调用 math_func
    # result = math_func(
    #     G=G_perms,
    #     A=A_perms,
    #     B=B_perms,
    #     multiply=perm_mult
    # )

    # # 输出结果
    # print("✅ 数学验证结果：")
    # for k, v in result.items():
    #     if 'set' not in k and 'elements' not in k:
    #         print(f"  {k}: {v}")

    # # 可视化
    # visual_func(result, title="S₃ Group: |A|=4, |B|=3, |G|=6 → |A|+|B|=7 > 6")

    # # 额外验证：计算 AB
    # AB = coding_func(A_perms, B_perms, perm_mult)
    # print(f"🔍 |AB| = {len(AB)}, expected |G| = {len(G_perms)}")
    # print(f"✅ AB == G: {set(G_perms) == AB}")
    # 假设当前环境可访问该路径


  

    # D4 群，生成元 r（旋转）、s（反射）
    G = DihedralGroup(4)
    r, s = G.generators

    # 显式列出 D4 的 8 个元素（与 r,s 同类型）
    G_elems = [r**0, r, r**2, r**3, s, r*s, r**2*s, r**3*s]

    # 文件内约定为左作用：p1*p2 表示先 p2 后 p1
    def mult(p1, p2):
        return p1 * p2

    A = {r**0, r}
    B = {r**2, s, s*r}

    # 因为 |A|+|B|=5 ≤ |G|=8，进入显式计算
    res = math_func(G=G_elems, A=A, B=B, multiply=mult)
    visual_func(res, title="D4 Group: |A|=2, |B|=3, |G|=8 → |A|+|B|=5 ≤ 8")
    print("size_condition(|A|+|B|>|G|):", res["condition_met"])
    print("|A|, |B|, |G| =", res["A_size"], res["B_size"], res["G_size"])
    print("|AB| =", res["AB_size"])
    print("AB==G ?", res["covers_G"])
    print("missing_count:", len(res["missing_elements"]))