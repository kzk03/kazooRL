#!/usr/bin/env python3
"""
ボット除外版データでの簡単な評価テスト

既存の評価システムを使用して、ボット除外の効果を確認する
"""

import json
import pickle
from typing import Dict, List, Set


def is_bot_developer(developer_name: str) -> bool:
    """開発者名がボットかどうかを判定"""
    if not developer_name:
        return True
    return "[bot]" in developer_name.lower()


def filter_bot_recommendations(recommendations: List[str]) -> List[str]:
    """推薦リストからボット開発者を除外"""
    return [dev for dev in recommendations if not is_bot_developer(dev)]


def evaluate_bot_filtering_impact():
    """ボット除外の影響を評価"""

    # 既存の評価結果を読み込み
    with open("evaluation_results_2023.json", "r") as f:
        results = json.load(f)

    print("🤖 Bot Filtering Impact Analysis")
    print("=" * 50)

    original_results = results["detailed_results"]

    # ボット除外前後の統計
    total_tasks = len(original_results)
    bot_in_top1_count = 0
    bot_in_top3_count = 0
    bot_in_top5_count = 0

    # ボット除外後の精度計算用
    filtered_correct_top1 = 0
    filtered_correct_top3 = 0
    filtered_correct_top5 = 0

    for task_result in original_results:
        recommendations = task_result["recommendations"]
        actual_assignees = task_result["actual_assignees"]

        # 元の推薦にボットが含まれているかチェック
        if recommendations and is_bot_developer(recommendations[0]):
            bot_in_top1_count += 1

        bot_in_top3 = any(is_bot_developer(dev) for dev in recommendations[:3])
        bot_in_top5 = any(is_bot_developer(dev) for dev in recommendations[:5])

        if bot_in_top3:
            bot_in_top3_count += 1
        if bot_in_top5:
            bot_in_top5_count += 1

        # ボット除外後の推薦リスト
        filtered_recs = filter_bot_recommendations(recommendations)

        # ボット除外後の精度計算
        if filtered_recs:
            if filtered_recs[0] in actual_assignees:
                filtered_correct_top1 += 1

            if any(dev in actual_assignees for dev in filtered_recs[:3]):
                filtered_correct_top3 += 1

            if any(dev in actual_assignees for dev in filtered_recs[:5]):
                filtered_correct_top5 += 1

    # 結果表示
    print(f"📊 Original Results:")
    print(f"  Total tasks: {total_tasks}")
    print(f"  Top-1 accuracy: {results['top_1_accuracy']:.1%}")
    print(f"  Top-3 accuracy: {results['top_3_accuracy']:.1%}")
    print(f"  Top-5 accuracy: {results['top_5_accuracy']:.1%}")

    print(f"\n🤖 Bot Presence in Recommendations:")
    print(
        f"  Bot in Top-1: {bot_in_top1_count}/{total_tasks} ({bot_in_top1_count/total_tasks:.1%})"
    )
    print(
        f"  Bot in Top-3: {bot_in_top3_count}/{total_tasks} ({bot_in_top3_count/total_tasks:.1%})"
    )
    print(
        f"  Bot in Top-5: {bot_in_top5_count}/{total_tasks} ({bot_in_top5_count/total_tasks:.1%})"
    )

    print(f"\n✨ Bot-Filtered Results:")
    print(f"  Top-1 accuracy: {filtered_correct_top1/total_tasks:.1%}")
    print(f"  Top-3 accuracy: {filtered_correct_top3/total_tasks:.1%}")
    print(f"  Top-5 accuracy: {filtered_correct_top5/total_tasks:.1%}")

    # 改善度計算
    improvement_top1 = (filtered_correct_top1 / total_tasks) - results["top_1_accuracy"]
    improvement_top3 = (filtered_correct_top3 / total_tasks) - results["top_3_accuracy"]
    improvement_top5 = (filtered_correct_top5 / total_tasks) - results["top_5_accuracy"]

    print(f"\n📈 Improvement:")
    print(f"  Top-1: +{improvement_top1:.1%}")
    print(f"  Top-3: +{improvement_top3:.1%}")
    print(f"  Top-5: +{improvement_top5:.1%}")

    # サンプル表示
    print(f"\n📋 Sample Bot-Filtered Recommendations:")
    for i, task_result in enumerate(original_results[:5]):
        original_recs = task_result["recommendations"][:3]
        filtered_recs = filter_bot_recommendations(task_result["recommendations"])[:3]
        actual = task_result["actual_assignees"]

        print(f"  Task {i+1}:")
        print(f"    Original: {original_recs}")
        print(f"    Filtered: {filtered_recs}")
        print(f"    Actual: {actual}")
        print()


def analyze_expert_trajectories():
    """エキスパート軌跡の比較分析"""

    print("🎯 Expert Trajectories Comparison")
    print("=" * 50)

    # 元のエキスパート軌跡
    try:
        with open("data/expert_trajectories.pkl", "rb") as f:
            original_trajectories = pickle.load(f)
        print(
            f"Original trajectories loaded: {len(original_trajectories)} trajectories"
        )
        if original_trajectories:
            print(f"  Total steps: {len(original_trajectories[0])}")
    except FileNotFoundError:
        print("Original trajectories not found")
        original_trajectories = []

    # ボット除外版エキスパート軌跡
    try:
        with open("data/expert_trajectories_bot_excluded.pkl", "rb") as f:
            bot_excluded_trajectories = pickle.load(f)
        print(
            f"Bot-excluded trajectories loaded: {len(bot_excluded_trajectories)} trajectories"
        )
        if bot_excluded_trajectories:
            print(f"  Total steps: {len(bot_excluded_trajectories[0])}")

            # 開発者の分析
            developers = set()
            for step in bot_excluded_trajectories[0]:
                dev = step["action_details"].get("developer")
                if dev:
                    developers.add(dev)

            print(f"  Unique developers: {len(developers)}")
            print(f"  Sample developers: {list(developers)[:10]}")

            # ボット開発者がいないことを確認
            bot_developers = [dev for dev in developers if is_bot_developer(dev)]
            print(f"  Bot developers found: {len(bot_developers)} {bot_developers}")

    except FileNotFoundError:
        print("Bot-excluded trajectories not found")


if __name__ == "__main__":
    print("🚀 Testing Bot Exclusion Impact")
    print("=" * 60)

    evaluate_bot_filtering_impact()
    print("\n" + "=" * 60)
    analyze_expert_trajectories()
