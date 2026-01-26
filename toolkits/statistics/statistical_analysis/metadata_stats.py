from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计题目 metadata 中 subject/topic 的分类数量"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default="/Users/shenyujiong/MutliSciTool/dataset/merged_single_questions.json",
        help="输入题目数据的 JSON 文件路径 (默认: dataset/merged_questions_augmented61.json)",
    )
    return parser.parse_args()


def load_items(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    json_path = Path(args.input_path)

    if not json_path.exists():
        raise FileNotFoundError(f"未找到文件: {json_path}")

    items = load_items(json_path)

    totals_by_subject = Counter()
    totals_by_pair = Counter()
    topics_by_subject: dict[str, Counter] = defaultdict(Counter)
    missing_subject = 0
    missing_topic = 0

    for item in items:
        metadata = item.get("metadata") or {}
        subject = metadata.get("subject")
        topic = metadata.get("topic")

        if subject is None or subject == "":
            subject = "UNKNOWN"
            missing_subject += 1
        if topic is None or topic == "":
            topic = "UNKNOWN"
            missing_topic += 1

        totals_by_subject[subject] += 1
        totals_by_pair[(subject, topic)] += 1
        topics_by_subject[subject][topic] += 1

    print(f"总题目数量: {len(items)}")
    if missing_subject:
        print(f"缺失 subject 的题目数量: {missing_subject}")
    if missing_topic:
        print(f"缺失 topic 的题目数量: {missing_topic}")

    print("\n按 subject 分类数量:")
    for subject, count in totals_by_subject.most_common():
        print(f"- {subject}: {count}")

    print("\n按 subject-topic 分类数量:")
    for (subject, topic), count in totals_by_pair.most_common():
        print(f"- {subject} -> {topic}: {count}")

    print("\n每个 subject 下 topic 分布:")
    for subject, topic_counter in topics_by_subject.items():
        print(f"{subject}:")
        for topic, count in topic_counter.most_common():
            print(f"  - {topic}: {count}")


if __name__ == "__main__":
    main()
