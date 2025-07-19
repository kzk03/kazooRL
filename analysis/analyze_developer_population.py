#!/usr/bin/env python3
"""
開発者母集団の詳細分析スクリプト
推薦システムの候補開発者と実際の担当者の関係を分析
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def analyze_developer_population():
    """開発者母集団を詳細分析"""

    print("🔍 開発者母集団分析開始")

    # データ読み込み
    print("📚 データ読み込み中...")

    # 2023年テストデータ
    with open("data/backlog_test_2023.json", "r", encoding="utf-8") as f:
        test_tasks = json.load(f)

    # 開発者プロファイル
    with open("configs/dev_profiles_test_2023.yaml", "r", encoding="utf-8") as f:
        dev_profiles = yaml.safe_load(f)

    print(f"📊 テストタスク数: {len(test_tasks)}")
    print(f"👥 プロファイル開発者数: {len(dev_profiles)}")

    # 実際の担当者を抽出
    actual_assignees = set()
    task_assignee_pairs = []

    for task in test_tasks:
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login"):
                    assignee_name = assignee["login"]
                    actual_assignees.add(assignee_name)
                    task_assignee_pairs.append(
                        {
                            "task_id": task.get("id"),
                            "task_title": task.get("title", "")[:50],
                            "assignee": assignee_name,
                        }
                    )

    print(f"🎯 実際の担当者数（ユニーク）: {len(actual_assignees)}")
    print(f"📋 タスク-担当者ペア数: {len(task_assignee_pairs)}")

    # 担当者の頻度分析
    assignee_counts = Counter([pair["assignee"] for pair in task_assignee_pairs])

    print(f"\n📈 担当者頻度 Top 10:")
    for assignee, count in assignee_counts.most_common(10):
        in_profiles = "✅" if assignee in dev_profiles else "❌"
        print(f"  {assignee:20s}: {count:3d}回 {in_profiles}")

    # プロファイルとの重複分析
    assignees_in_profiles = actual_assignees & set(dev_profiles.keys())
    assignees_not_in_profiles = actual_assignees - set(dev_profiles.keys())

    print(f"\n🔄 プロファイルとの重複分析:")
    print(
        f"  プロファイルに含まれる担当者: {len(assignees_in_profiles)}/{len(actual_assignees)} ({len(assignees_in_profiles)/len(actual_assignees)*100:.1f}%)"
    )
    print(f"  プロファイルに含まれない担当者: {len(assignees_not_in_profiles)}")

    if assignees_not_in_profiles:
        print(f"\n❌ プロファイルに含まれない主要担当者:")
        missing_assignee_counts = {
            name: assignee_counts[name] for name in assignees_not_in_profiles
        }
        for assignee, count in sorted(
            missing_assignee_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"  {assignee:20s}: {count:3d}回")

    # 推薦候補（上位200人）の分析
    candidate_developers = list(dev_profiles.keys())[:200]
    candidates_set = set(candidate_developers)

    print(f"\n🎯 推薦候補分析:")
    print(f"  推薦候補数: {len(candidate_developers)}")
    print(
        f"  候補に含まれる実際の担当者: {len(assignees_in_profiles & candidates_set)}"
    )
    print(
        f"  候補に含まれない実際の担当者: {len(assignees_in_profiles - candidates_set)}"
    )

    # 候補に含まれない担当者の詳細
    missing_from_candidates = assignees_in_profiles - candidates_set
    if missing_from_candidates:
        print(f"\n⚠️ 推薦候補に含まれない担当者（プロファイルはあり）:")
        missing_candidate_counts = {
            name: assignee_counts[name]
            for name in missing_from_candidates
            if name in assignee_counts
        }
        for assignee, count in sorted(
            missing_candidate_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            # プロファイル内での順位を確認
            profile_keys = list(dev_profiles.keys())
            if assignee in profile_keys:
                rank = profile_keys.index(assignee) + 1
                print(f"  {assignee:20s}: {count:3d}回 (プロファイル順位: {rank})")

    # 評価可能タスクの分析
    evaluable_tasks = []
    for task in test_tasks:
        if task.get("assignees"):
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in candidates_set for assignee in assignees):
                evaluable_tasks.append(
                    {
                        "task_id": task.get("id"),
                        "task_title": task.get("title", "")[:50],
                        "assignees": assignees,
                        "assignees_in_candidates": [
                            a for a in assignees if a in candidates_set
                        ],
                    }
                )

    print(f"\n✅ 評価可能タスク分析:")
    print(
        f"  評価可能タスク数: {len(evaluable_tasks)}/{len(test_tasks)} ({len(evaluable_tasks)/len(test_tasks)*100:.1f}%)"
    )

    # 開発者プロファイルの順序分析
    print(f"\n📋 開発者プロファイルの順序分析:")
    profile_keys = list(dev_profiles.keys())

    # 実際の担当者がプロファイル内でどの位置にいるか
    assignee_ranks = {}
    for assignee in assignees_in_profiles:
        if assignee in profile_keys:
            rank = profile_keys.index(assignee) + 1
            assignee_ranks[assignee] = rank

    # 順位別の分布
    rank_ranges = {"1-50": 0, "51-100": 0, "101-200": 0, "201-500": 0, "501+": 0}

    for rank in assignee_ranks.values():
        if rank <= 50:
            rank_ranges["1-50"] += 1
        elif rank <= 100:
            rank_ranges["51-100"] += 1
        elif rank <= 200:
            rank_ranges["101-200"] += 1
        elif rank <= 500:
            rank_ranges["201-500"] += 1
        else:
            rank_ranges["501+"] += 1

    print(f"  実際の担当者のプロファイル内順位分布:")
    for range_name, count in rank_ranges.items():
        percentage = count / len(assignee_ranks) * 100 if assignee_ranks else 0
        print(f"    {range_name:8s}: {count:3d}人 ({percentage:5.1f}%)")

    # 推薦精度への影響分析
    print(f"\n🎯 推薦精度への影響分析:")

    # 理論的最大精度（全ての実際の担当者が候補に含まれている場合）
    max_possible_accuracy = (
        len(assignees_in_profiles & candidates_set) / len(assignees_in_profiles)
        if assignees_in_profiles
        else 0
    )
    print(f"  理論的最大精度: {max_possible_accuracy*100:.1f}%")

    # 現在の候補選択による制約
    coverage_rate = (
        len(evaluable_tasks) / len([t for t in test_tasks if t.get("assignees")])
        if test_tasks
        else 0
    )
    print(f"  評価可能タスクカバー率: {coverage_rate*100:.1f}%")

    return {
        "total_test_tasks": len(test_tasks),
        "total_dev_profiles": len(dev_profiles),
        "unique_actual_assignees": len(actual_assignees),
        "assignees_in_profiles": len(assignees_in_profiles),
        "assignees_not_in_profiles": len(assignees_not_in_profiles),
        "candidate_developers": len(candidate_developers),
        "evaluable_tasks": len(evaluable_tasks),
        "max_possible_accuracy": max_possible_accuracy,
        "coverage_rate": coverage_rate,
        "top_assignees": dict(assignee_counts.most_common(10)),
        "missing_from_profiles": list(assignees_not_in_profiles)[:10],
        "missing_from_candidates": (
            list(missing_from_candidates)[:10] if missing_from_candidates else []
        ),
    }


if __name__ == "__main__":
    results = analyze_developer_population()

    # 結果をJSONで保存
    with open("developer_population_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 分析結果を保存: developer_population_analysis.json")
