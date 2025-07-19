#!/usr/bin/env python3
"""
シンプルなアクティブ開発者推薦システム
特徴量抽出エラーを回避し、基本的な推薦を実行
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# パッケージのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))


def parse_datetime(date_str):
    """日付文字列をdatetimeオブジェクトに変換"""
    try:
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        return datetime.fromisoformat(date_str)
    except:
        return None


def get_active_developers_for_task(
    task, backlog_data, dev_profiles_data, activity_window_months=3
):
    """タスクの時期に基づいてアクティブな開発者を取得"""
    task_date = parse_datetime(task.get("updated_at", task.get("created_at", "")))
    if not task_date:
        return []

    # タスク日付から活動期間を設定（過去Nヶ月）
    activity_start = task_date - timedelta(days=activity_window_months * 30)

    # 活動期間内でアクティブな開発者を抽出
    active_developers = set()

    for other_task in backlog_data:
        other_task_date = parse_datetime(
            other_task.get("updated_at", other_task.get("created_at", ""))
        )
        if not other_task_date:
            continue

        # 活動期間内のタスクかチェック（過去のみ、未来は含めない）
        if activity_start <= other_task_date < task_date:
            # タスクの担当者を追加
            if other_task.get("assignees"):
                for assignee in other_task["assignees"]:
                    if assignee.get("login") and assignee["login"] in dev_profiles_data:
                        active_developers.add(assignee["login"])

            # PR作成者を追加（実装能力が証明済み）
            if (
                other_task.get("pull_request")
                or other_task.get("type") == "pull_request"
            ):
                author = other_task.get("user", other_task.get("author", {}))
                if (
                    author
                    and author.get("login")
                    and author["login"] in dev_profiles_data
                ):
                    active_developers.add(author["login"])

    return list(active_developers)


class SimpleActiveRecommendationSystem:
    """シンプルなアクティブ開発者推薦システム"""

    def __init__(self, backlog_data, dev_profiles_data):
        self.backlog_data = backlog_data
        self.dev_profiles = dev_profiles_data

        # 開発者の活動統計を事前計算
        self.developer_stats = self._calculate_developer_stats()

    def _calculate_developer_stats(self):
        """開発者の活動統計を計算"""
        stats = defaultdict(
            lambda: {
                "total_tasks": 0,
                "recent_tasks": 0,
                "pr_count": 0,
                "issue_count": 0,
                "last_activity": None,
            }
        )

        current_time = datetime.now().replace(tzinfo=None)
        recent_threshold = current_time - timedelta(days=90)

        for task in self.backlog_data:
            task_date = parse_datetime(
                task.get("updated_at", task.get("created_at", ""))
            )
            if not task_date:
                continue

            # 担当者の統計
            if task.get("assignees"):
                for assignee in task["assignees"]:
                    if assignee.get("login") and assignee["login"] in self.dev_profiles:
                        dev_name = assignee["login"]
                        stats[dev_name]["total_tasks"] += 1
                        if task_date.replace(tzinfo=None) > recent_threshold:
                            stats[dev_name]["recent_tasks"] += 1
                        if (
                            not stats[dev_name]["last_activity"]
                            or task_date > stats[dev_name]["last_activity"]
                        ):
                            stats[dev_name]["last_activity"] = task_date

            # PR作成者の統計
            if task.get("pull_request") or task.get("type") == "pull_request":
                author = task.get("user", task.get("author", {}))
                if (
                    author
                    and author.get("login")
                    and author["login"] in self.dev_profiles
                ):
                    dev_name = author["login"]
                    stats[dev_name]["pr_count"] += 1
                    if (
                        not stats[dev_name]["last_activity"]
                        or task_date > stats[dev_name]["last_activity"]
                    ):
                        stats[dev_name]["last_activity"] = task_date
            else:
                # Issue作成者の統計
                author = task.get("user", task.get("author", {}))
                if (
                    author
                    and author.get("login")
                    and author["login"] in self.dev_profiles
                ):
                    dev_name = author["login"]
                    stats[dev_name]["issue_count"] += 1

        return dict(stats)

    def calculate_simple_score(self, dev_name, task):
        """シンプルなスコア計算"""
        if dev_name not in self.developer_stats:
            return 0.0

        stats = self.developer_stats[dev_name]
        score = 0.0

        # 最近の活動（重要度: 40%）
        score += stats["recent_tasks"] * 0.4

        # 総活動量（重要度: 30%）
        score += min(stats["total_tasks"] / 10.0, 5.0) * 0.3

        # PR作成経験（重要度: 20%）
        score += min(stats["pr_count"] / 5.0, 2.0) * 0.2

        # 最新活動からの経過時間（重要度: 10%）
        if stats["last_activity"]:
            days_since_last = (
                datetime.now().replace(tzinfo=None)
                - stats["last_activity"].replace(tzinfo=None)
            ).days
            freshness_score = max(0, 1.0 - days_since_last / 365.0)  # 1年で0になる
            score += freshness_score * 0.1

        return score

    def recommend_developers(
        self, task, activity_window_months=3, num_recommendations=5
    ):
        """開発者を推薦"""
        # 1. アクティブ開発者を取得
        active_developers = get_active_developers_for_task(
            task, self.backlog_data, self.dev_profiles, activity_window_months
        )

        # 2. 現在のタスクの担当者を追加
        actual_assignees = set()
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in self.dev_profiles:
                    actual_assignees.add(assignee["login"])

        # 3. 現在のタスクの作成者を追加
        task_author = set()
        author = task.get("user", task.get("author", {}))
        if author and author.get("login") and author["login"] in self.dev_profiles:
            task_author.add(author["login"])

        # 4. 候補開発者リストを統合
        candidate_developers = list(
            task_author | actual_assignees | set(active_developers)
        )

        if not candidate_developers:
            return [], {
                "total_active_developers": 0,
                "actual_assignees_count": 0,
                "task_author_count": 0,
                "final_candidates_count": 0,
                "activity_window_months": activity_window_months,
            }

        # 5. スコアを計算
        developer_scores = []
        for dev_name in candidate_developers:
            score = self.calculate_simple_score(dev_name, task)
            developer_scores.append((dev_name, score))

        # 6. スコア順にソート
        developer_scores.sort(key=lambda x: x[1], reverse=True)

        candidate_info = {
            "total_active_developers": len(active_developers),
            "actual_assignees_count": len(actual_assignees),
            "task_author_count": len(task_author),
            "final_candidates_count": len(candidate_developers),
            "activity_window_months": activity_window_months,
        }

        return developer_scores[:num_recommendations], candidate_info


def evaluate_simple_recommendations(
    backlog_data, dev_profiles_data, activity_window_months=3
):
    """シンプル推薦システムを評価"""

    # 推薦システムを初期化
    recommender = SimpleActiveRecommendationSystem(backlog_data, dev_profiles_data)

    # 評価対象タスクを抽出
    eval_tasks = []
    for task in backlog_data:
        has_assignees = False
        has_author = False

        # 担当者がいるかチェック
        if task.get("assignees") and len(task["assignees"]) > 0:
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in dev_profiles_data for assignee in assignees):
                has_assignees = True

        # 作成者がいるかチェック
        author = task.get("user", task.get("author", {}))
        if author and author.get("login") and author["login"] in dev_profiles_data:
            has_author = True

        if has_assignees or has_author:
            eval_tasks.append(task)

    print(f"📊 評価対象タスク: {len(eval_tasks)}/{len(backlog_data)}")

    results = {
        "total_tasks": 0,
        "correct_recommendations": 0,
        "top_k_hits": defaultdict(int),
        "recommendation_details": [],
    }

    # 評価実行
    for task_idx, task in enumerate(tqdm(eval_tasks, desc="🎯 シンプル推薦評価")):
        # 実際の正解を取得
        ground_truth = set()

        # 担当者を正解に追加
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in dev_profiles_data:
                    ground_truth.add(assignee["login"])

        # PR作成者を正解に追加
        if task.get("pull_request") or task.get("type") == "pull_request":
            author = task.get("user", task.get("author", {}))
            if author and author.get("login") and author["login"] in dev_profiles_data:
                ground_truth.add(author["login"])

        if not ground_truth:
            continue

        # 推薦実行
        try:
            recommendations_with_scores, candidate_info = (
                recommender.recommend_developers(task, activity_window_months)
            )

            if not recommendations_with_scores:
                continue

            recommendations = [
                dev_name for dev_name, score in recommendations_with_scores
            ]

            # 正解率計算
            correct_in_top_k = []
            for k in [1, 3, 5]:
                top_k_recs = recommendations[:k]
                hit = any(gt in top_k_recs for gt in ground_truth)
                if hit:
                    results["top_k_hits"][f"top_{k}"] += 1
                correct_in_top_k.append(hit)

            # 詳細結果記録
            results["recommendation_details"].append(
                {
                    "task_id": task.get("id"),
                    "task_title": task.get("title", "Unknown")[:50],
                    "ground_truth": list(ground_truth),
                    "recommendations": recommendations,
                    "scores": [
                        score for dev_name, score in recommendations_with_scores
                    ],
                    "candidate_info": candidate_info,
                    "correct_in_top_1": correct_in_top_k[0],
                    "correct_in_top_3": correct_in_top_k[1],
                    "correct_in_top_5": correct_in_top_k[2],
                }
            )

            results["total_tasks"] += 1

        except Exception as e:
            print(f"⚠️ タスク {task_idx} でエラー: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="シンプルなアクティブ開発者推薦システム評価"
    )
    parser.add_argument("--config", required=True, help="設定ファイルのパス")
    parser.add_argument("--activity-months", type=int, default=3, help="活動期間の月数")
    parser.add_argument(
        "--output",
        default="simple_active_recommendation_results.json",
        help="結果出力ファイル",
    )

    args = parser.parse_args()

    print("🎯 シンプルアクティブ開発者推薦システム評価開始")
    print(f"📝 設定: {args.config}")
    print(f"📅 活動期間: {args.activity_months}ヶ月")

    # 設定読み込み
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # データ読み込み
    with open(config["env"]["backlog_path"], "r", encoding="utf-8") as f:
        backlog_data = json.load(f)
    with open(config["env"]["dev_profiles_path"], "r", encoding="utf-8") as f:
        dev_profiles_data = yaml.safe_load(f)

    print(f"📊 データ: {len(backlog_data)} タスク, {len(dev_profiles_data)} 開発者")

    # 評価実行
    results = evaluate_simple_recommendations(
        backlog_data, dev_profiles_data, args.activity_months
    )

    # 結果計算・表示
    total_tasks = results["total_tasks"]
    if total_tasks > 0:
        accuracy_top_1 = results["top_k_hits"]["top_1"] / total_tasks
        accuracy_top_3 = results["top_k_hits"]["top_3"] / total_tasks
        accuracy_top_5 = results["top_k_hits"]["top_5"] / total_tasks

        print("\n" + "=" * 60)
        print("🎯 シンプル推薦システム評価結果")
        print("=" * 60)
        print(f"評価タスク数: {total_tasks}")
        print(f"活動期間: {args.activity_months}ヶ月")
        print(f"")
        print(
            f"Top-1 Accuracy: {accuracy_top_1:.3f} ({results['top_k_hits']['top_1']}/{total_tasks})"
        )
        print(
            f"Top-3 Accuracy: {accuracy_top_3:.3f} ({results['top_k_hits']['top_3']}/{total_tasks})"
        )
        print(
            f"Top-5 Accuracy: {accuracy_top_5:.3f} ({results['top_k_hits']['top_5']}/{total_tasks})"
        )
        print("=" * 60)

        # 結果保存
        final_results = {
            "evaluation_config": args.config,
            "activity_window_months": args.activity_months,
            "total_tasks_evaluated": total_tasks,
            "results": {
                "top_1_accuracy": float(accuracy_top_1),
                "top_3_accuracy": float(accuracy_top_3),
                "top_5_accuracy": float(accuracy_top_5),
            },
            "method": "Simple_Active_Developer_Recommendation",
            "sample_results": results["recommendation_details"][:10],
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"💾 結果を保存: {args.output}")

        # サンプル結果表示
        print("\n📋 サンプル推薦結果:")
        for i, detail in enumerate(results["recommendation_details"][:5]):
            print(f"\nタスク {i+1}: {detail['task_title']}")
            print(f"  正解: {detail['ground_truth']}")
            print(f"  推薦: {detail['recommendations'][:3]}")
            print(f"  スコア: {[f'{s:.2f}' for s in detail['scores'][:3]]}")
            print(f"  Top-1正解: {'✅' if detail['correct_in_top_1'] else '❌'}")

    else:
        print("⚠️ 評価できるタスクが見つかりませんでした")


if __name__ == "__main__":
    main()
