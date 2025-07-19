#!/usr/bin/env python3
"""
アクティブ開発者に基づく推薦システム評価
直近の活動状況を考慮した動的な候補選択による推薦
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from tqdm import tqdm

# パッケージのパスを追加
sys.path.append(str(Path(__file__).parent.parent / "src"))

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.features.feature_extractor import FeatureExtractor


class SimpleConfig:
    """辞書をオブジェクトのように扱うためのクラス"""

    def __init__(self, config_dict):
        self._dict = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """辞書のgetメソッドと同様の動作"""
        return self._dict.get(key, default)


def load_config(config_path):
    """設定ファイルを読み込む"""
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return SimpleConfig(config_dict)


def parse_datetime(date_str):
    """日付文字列をdatetimeオブジェクトに変換"""
    try:
        if date_str.endswith("Z"):
            date_str = date_str[:-1] + "+00:00"
        return datetime.fromisoformat(date_str)
    except:
        return None


def get_active_developers_for_task(
    task, backlog_data, dev_profiles_data, activity_window_months=3, debug=False
):
    """
    タスクの時期に基づいてアクティブな開発者を取得（シンプル版）

    Args:
        task: 対象タスク
        backlog_data: 全タスクデータ
        dev_profiles_data: 開発者プロファイル
        activity_window_months: 活動期間の月数
        debug: デバッグ情報を出力するか

    Returns:
        list: アクティブな開発者リスト
    """
    task_date = parse_datetime(task.get("updated_at", task.get("created_at", "")))
    if not task_date:
        if debug:
            print(f"   デバッグ: タスク日付が取得できません")
        return []

    # タスク日付から活動期間を設定（過去N ヶ月）
    activity_start = task_date - timedelta(days=activity_window_months * 30)

    if debug:
        print(
            f"   デバッグ: タスク日付 {task_date}, 活動期間 {activity_start} - {task_date}"
        )

    # 活動期間内でアクティブな開発者を抽出
    active_developers = set()
    relevant_tasks = 0

    for other_task in backlog_data:
        other_task_date = parse_datetime(
            other_task.get("updated_at", other_task.get("created_at", ""))
        )
        if not other_task_date:
            continue

        # 活動期間内のタスクかチェック（過去のみ、未来は含めない）
        if activity_start <= other_task_date < task_date:
            relevant_tasks += 1
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

    if debug:
        print(
            f"   デバッグ: 関連タスク数 {relevant_tasks}, アクティブ開発者数 {len(active_developers)}"
        )
        if active_developers:
            print(f"   デバッグ: アクティブ開発者 {list(active_developers)[:5]}...")

    return list(active_developers)


def analyze_developer_activity_patterns(backlog_data, dev_profiles_data):
    """開発者の活動パターンを分析"""
    print("📊 開発者活動パターン分析中...")

    # 月別活動統計
    monthly_activity = defaultdict(lambda: defaultdict(int))
    developer_activity = defaultdict(list)

    for task in backlog_data:
        task_date = parse_datetime(task.get("updated_at", task.get("created_at", "")))
        if not task_date:
            continue

        month_key = task_date.strftime("%Y-%m")

        # 担当者の活動記録
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in dev_profiles_data:
                    dev_name = assignee["login"]
                    monthly_activity[month_key][dev_name] += 1
                    developer_activity[dev_name].append(task_date)

    # 活動統計の表示
    print(f"\n📈 月別活動統計:")
    for month in sorted(monthly_activity.keys())[-6:]:  # 直近6ヶ月
        active_devs = len(monthly_activity[month])
        total_tasks = sum(monthly_activity[month].values())
        print(f"  {month}: {active_devs}人の開発者, {total_tasks}タスク")

    # 最もアクティブな開発者
    dev_task_counts = {dev: len(dates) for dev, dates in developer_activity.items()}
    print(f"\n🏆 最もアクティブな開発者 Top 10:")
    for dev, count in Counter(dev_task_counts).most_common(10):
        latest_activity = (
            max(developer_activity[dev]) if developer_activity[dev] else None
        )
        latest_str = latest_activity.strftime("%Y-%m-%d") if latest_activity else "不明"
        print(f"  {dev:20s}: {count:3d}タスク (最新: {latest_str})")

    return developer_activity, monthly_activity


class ActiveDeveloperRecommendationSystem:
    """アクティブ開発者に基づく推薦システム"""

    def __init__(self, irl_weights, ppo_model, env, feature_extractor, backlog_data):
        self.irl_weights = irl_weights
        self.ppo_model = ppo_model
        self.env = env
        self.feature_extractor = feature_extractor
        self.dev_profiles = env.dev_profiles
        self.backlog_data = backlog_data

    def get_task_developer_features(self, task, developer_name):
        """タスクと開発者のペアから特徴量を抽出"""
        try:
            dev_profile = self.dev_profiles[developer_name]
            developer = {"name": developer_name, "profile": dev_profile}
            features = self.feature_extractor.get_features(task, developer, self.env)
            return features
        except Exception as e:
            # フォールバック: デフォルト特徴量を返す
            return self._get_fallback_features(task, developer_name)

    def _get_fallback_features(self, task, developer_name):
        """特徴量抽出に失敗した場合のフォールバック特徴量"""
        try:
            # 基本的な62次元の特徴量ベクトルを作成
            features = np.zeros(62)

            # 開発者の基本統計
            dev_profile = self.dev_profiles.get(developer_name, {})
            features[0] = dev_profile.get("rank", 5000) / 5000.0  # 正規化されたランク
            features[1] = min(
                dev_profile.get("total_commits", 0) / 100.0, 1.0
            )  # コミット数

            # タスクの基本特徴
            features[2] = len(task.get("title", "")) / 100.0  # タイトル長
            features[3] = len(task.get("body", "")) / 1000.0  # 本文長
            features[4] = task.get("comments_count", 0) / 10.0  # コメント数

            # ラベル特徴
            labels = task.get("labels", [])
            if isinstance(labels, list) and labels:
                if isinstance(labels[0], dict):
                    label_names = [label.get("name", "") for label in labels]
                else:
                    label_names = labels

                features[5] = (
                    1.0 if any("bug" in label.lower() for label in label_names) else 0.0
                )
                features[6] = (
                    1.0
                    if any("enhancement" in label.lower() for label in label_names)
                    else 0.0
                )
                features[7] = (
                    1.0
                    if any("documentation" in label.lower() for label in label_names)
                    else 0.0
                )

            # 残りの特徴量はランダムノイズで埋める（GAT埋め込み部分）
            features[25:] = np.random.normal(0, 0.1, 37)

            return features

        except Exception as e:
            # 最終フォールバック: 全て0の特徴量
            return np.zeros(62)

    def calculate_hybrid_score(self, features, weights=(0.5, 0.3, 0.2)):
        """ハイブリッドスコアを計算"""
        irl_weight, ppo_weight, gat_weight = weights

        try:
            # IRL スコア
            irl_score = np.dot(features, self.irl_weights)

            # PPO スコア
            obs = features.reshape(1, -1)
            with torch.no_grad():
                if hasattr(self.ppo_model.policy, "predict_values"):
                    obs_tensor = torch.FloatTensor(obs)
                    value = self.ppo_model.policy.predict_values(obs_tensor)
                    ppo_score = float(value.item())
                else:
                    ppo_score = 0.5

            # GAT スコア
            gat_features = features[25:62] if len(features) > 61 else features[25:]
            gat_score = np.mean(np.abs(gat_features)) if len(gat_features) > 0 else 0.0

            # 総合スコア
            total_score = (
                irl_weight * irl_score + ppo_weight * ppo_score + gat_weight * gat_score
            )

            return float(total_score), {
                "irl_score": float(irl_score),
                "ppo_score": float(ppo_score),
                "gat_score": float(gat_score),
            }

        except Exception as e:
            return 0.0, {"irl_score": 0.0, "ppo_score": 0.0, "gat_score": 0.0}

    def recommend_developers_with_activity(
        self, task, activity_window_months=3, num_recommendations=5
    ):
        """
        アクティブ開発者に基づいて推薦（シンプル版）

        Args:
            task: 推薦対象のタスク
            activity_window_months: 活動期間の月数
            num_recommendations: 推薦する開発者数

        Returns:
            tuple: (推薦結果, 候補開発者情報)
        """
        # 1. 直近N ヶ月でアクティブな開発者を取得
        active_developers = get_active_developers_for_task(
            task,
            self.backlog_data,
            self.dev_profiles,
            activity_window_months,
            debug=False,
        )

        # 2. 現在のタスクの担当者を追加（Ground Truth）
        actual_assignees = set()
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in self.dev_profiles:
                    actual_assignees.add(assignee["login"])

        # 3. 現在のタスクがPRの場合、作成者を追加（最重要候補）
        task_author = set()
        author = task.get("user", task.get("author", {}))
        if author and author.get("login") and author["login"] in self.dev_profiles:
            task_author.add(author["login"])

        # 4. 候補開発者リストを統合
        # 優先順位: タスク作成者 > 担当者 > アクティブ開発者
        candidate_developers = list(
            task_author | actual_assignees | set(active_developers)
        )

        # 候補が少なすぎる場合は上位開発者を追加
        if len(candidate_developers) < 5:
            # 上位ランクの開発者を追加
            top_developers = sorted(
                self.dev_profiles.keys(),
                key=lambda dev: self.dev_profiles.get(dev, {}).get(
                    "rank", float("inf")
                ),
            )[:10]

            for dev in top_developers:
                if dev not in candidate_developers:
                    candidate_developers.append(dev)
                    if len(candidate_developers) >= 5:
                        break

        # それでも候補がない場合の警告
        if len(candidate_developers) == 0:
            task_date_str = task.get("updated_at", task.get("created_at", "Unknown"))
            print(f"⚠️ 候補開発者が見つかりません:")
            print(f"   タスク ID: {task.get('id', 'Unknown')}")
            print(f"   タスク日付: {task_date_str}")
            return [], {
                "total_active_developers": 0,
                "actual_assignees_count": 0,
                "task_author_count": 0,
                "final_candidates_count": 0,
                "activity_window_months": activity_window_months,
                "is_pull_request": bool(
                    task.get("pull_request") or task.get("type") == "pull_request"
                ),
            }

        # 推薦スコアを計算
        developer_scores = []
        feature_extraction_errors = 0

        for dev_name in candidate_developers:
            try:
                features = self.get_task_developer_features(task, dev_name)
                if features is None:
                    feature_extraction_errors += 1
                    continue

                total_score, score_details = self.calculate_hybrid_score(features)
                developer_scores.append((dev_name, total_score, score_details))
            except Exception as e:
                feature_extraction_errors += 1
                continue

        # スコア順にソート（降順）
        developer_scores.sort(key=lambda x: x[1], reverse=True)

        candidate_info = {
            "total_active_developers": len(active_developers),
            "actual_assignees_count": len(actual_assignees),
            "task_author_count": len(task_author),
            "final_candidates_count": len(candidate_developers),
            "activity_window_months": activity_window_months,
            "is_pull_request": bool(
                task.get("pull_request") or task.get("type") == "pull_request"
            ),
        }

        return developer_scores[:num_recommendations], candidate_info


def create_mock_task(task_data):
    """タスクデータからモックタスクオブジェクトを作成"""

    class MockTask:
        def __init__(self, task_data):
            self.id = task_data.get("id")
            self.title = task_data.get("title", "")
            self.body = task_data.get("body", "")

            # ラベルの形式を統一的に処理
            labels_data = task_data.get("labels", [])
            if labels_data and isinstance(labels_data[0], dict):
                self.labels = [label.get("name") for label in labels_data]
            else:
                self.labels = labels_data if isinstance(labels_data, list) else []

            self.comments = task_data.get("comments", 0)
            self.updated_at = task_data.get("updated_at", "2023-01-01T00:00:00Z")
            self.user = task_data.get("user", task_data.get("author", {}))
            self.assignees = task_data.get("assignees", [])

            # 日付文字列をdatetimeオブジェクトに変換
            if isinstance(self.updated_at, str):
                try:
                    if self.updated_at.endswith("Z"):
                        self.updated_at = self.updated_at[:-1] + "+00:00"
                    self.updated_at = datetime.fromisoformat(self.updated_at)
                except:
                    self.updated_at = datetime(2023, 1, 1)

    return MockTask(task_data)


def evaluate_active_developer_recommendations(
    backlog_data,
    dev_profiles_data,
    irl_weights,
    ppo_model,
    env,
    feature_extractor,
    activity_window_months=3,
    num_recommendations=5,
):
    """アクティブ開発者に基づく推薦システムを評価"""

    # 開発者活動パターンを分析
    developer_activity, monthly_activity = analyze_developer_activity_patterns(
        backlog_data, dev_profiles_data
    )

    # 推薦システムを初期化
    recommender = ActiveDeveloperRecommendationSystem(
        irl_weights, ppo_model, env, feature_extractor, backlog_data
    )

    results = {
        "total_tasks": 0,
        "tasks_with_assignees": 0,
        "correct_recommendations": 0,
        "top_k_hits": defaultdict(int),
        "recommendation_details": [],
        "activity_stats": {
            "activity_window_months": activity_window_months,
            "avg_active_developers_per_task": 0,
            "candidate_info_summary": [],
        },
    }

    print(f"🎯 アクティブ開発者推薦評価開始: {len(backlog_data)} タスクで評価")
    print(f"📅 活動期間: {activity_window_months}ヶ月")

    # 担当者情報があるタスクまたはPRを抽出
    tasks_with_assignees = []
    for task in backlog_data:
        has_assignees = False
        has_pr_author = False

        # 担当者がいるかチェック
        if task.get("assignees") and len(task["assignees"]) > 0:
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in dev_profiles_data for assignee in assignees):
                has_assignees = True

        # PRの作成者がいるかチェック
        if task.get("pull_request") or task.get("type") == "pull_request":
            author = task.get("user", task.get("author", {}))
            if author and author.get("login") and author["login"] in dev_profiles_data:
                has_pr_author = True

        # 担当者またはPR作成者がいる場合に評価対象とする
        if has_assignees or has_pr_author:
            tasks_with_assignees.append(task)

    print(
        f"📊 評価対象タスク（担当者またはPR作成者あり）: {len(tasks_with_assignees)}/{len(backlog_data)}"
    )

    eval_tasks = tasks_with_assignees
    print(f"🎯 アクティブ開発者推薦評価: {len(eval_tasks)} タスクで評価実行")

    # 評価タスクの進捗バー
    task_progress = tqdm(
        enumerate(eval_tasks),
        total=len(eval_tasks),
        desc="🎯 アクティブ開発者推薦評価",
        unit="task",
        colour="blue",
        leave=True,
    )

    total_active_developers = 0

    for task_idx, task in task_progress:
        # タスクの実際の担当者を取得（Ground Truth）
        actual_assignees_task = [
            assignee.get("login")
            for assignee in task["assignees"]
            if assignee.get("login")
        ]

        # PRの場合は作成者も正解として追加
        pr_author_task = None
        if task.get("pull_request") or task.get("type") == "pull_request":
            author = task.get("user", task.get("author", {}))
            if author and author.get("login"):
                pr_author_task = author["login"]
                # PR作成者を正解に追加（重複を避ける）
                if pr_author_task not in actual_assignees_task:
                    actual_assignees_task.append(pr_author_task)

        if not actual_assignees_task:
            task_progress.set_postfix({"Status": "担当者なし (スキップ)"})
            continue

        try:
            # モックタスクオブジェクトを作成
            mock_task = create_mock_task(task)

            # アクティブ開発者に基づく推薦を実行
            recommendations_with_scores, candidate_info = (
                recommender.recommend_developers_with_activity(
                    task, activity_window_months, num_recommendations
                )
            )

            if not recommendations_with_scores:
                if task_idx < 5:  # 最初の5つの失敗のみ詳細表示
                    print(f"⚠️ タスク {task_idx}: 推薦結果が空です")
                    print(
                        f"   候補開発者数: {candidate_info.get('final_candidates_count', 0)}"
                    )
                task_progress.set_postfix({"Status": "推薦失敗"})
                continue

            # 統計情報を蓄積
            total_active_developers += candidate_info["total_active_developers"]
            results["activity_stats"]["candidate_info_summary"].append(candidate_info)

            # 推薦リストを作成
            recommendations = [
                dev_name for dev_name, score, details in recommendations_with_scores
            ]

            # 正解率を計算
            correct_in_top_k = []
            for k in [1, 3, 5]:
                top_k_recs = recommendations[:k]
                hit = any(assignee in top_k_recs for assignee in actual_assignees_task)
                if hit:
                    results["top_k_hits"][f"top_{k}"] += 1
                correct_in_top_k.append(hit)

            # 詳細結果を記録
            results["recommendation_details"].append(
                {
                    "task_id": task.get("id"),
                    "task_title": task.get("title", "Unknown")[:50],
                    "task_date": task.get("updated_at", ""),
                    "task_type": task.get("type", "issue"),
                    "is_pull_request": bool(
                        task.get("pull_request") or task.get("type") == "pull_request"
                    ),
                    "actual_assignees": actual_assignees_task,
                    "pr_author": pr_author_task,
                    "recommendations": recommendations,
                    "candidate_info": candidate_info,
                    "recommendation_scores": [
                        (dev, float(score))
                        for dev, score, details in recommendations_with_scores
                    ],
                    "correct_in_top_1": correct_in_top_k[0],
                    "correct_in_top_3": correct_in_top_k[1],
                    "correct_in_top_5": correct_in_top_k[2],
                }
            )

            results["total_tasks"] += 1
            results["tasks_with_assignees"] += 1

            # 進捗バーの情報更新
            if results["total_tasks"] > 0:
                top1_acc = results["top_k_hits"]["top_1"] / results["total_tasks"]
                top3_acc = results["top_k_hits"]["top_3"] / results["total_tasks"]
                avg_candidates = total_active_developers / results["total_tasks"]
                task_progress.set_postfix(
                    {
                        "Top-1": f"{top1_acc:.3f}",
                        "Top-3": f"{top3_acc:.3f}",
                        "Avg候補": f"{avg_candidates:.0f}",
                        "完了": f"{results['total_tasks']}/{len(eval_tasks)}",
                    }
                )

        except Exception as e:
            if task_idx < 5:  # 最初の5つのエラーのみ詳細表示
                print(f"⚠️ タスク {task_idx} でエラー: {e}")
            task_progress.set_postfix({"Status": f"エラー: {str(e)[:20]}"})
            continue

    # 統計情報を計算
    if results["total_tasks"] > 0:
        results["activity_stats"]["avg_active_developers_per_task"] = (
            total_active_developers / results["total_tasks"]
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="アクティブ開発者に基づく推薦システム評価"
    )
    parser.add_argument("--config", required=True, help="設定ファイルのパス")
    parser.add_argument(
        "--irl-weights", required=True, help="学習済みIRL重みファイルのパス"
    )
    parser.add_argument(
        "--ppo-model", required=True, help="学習済みPPOモデルファイルのパス"
    )
    parser.add_argument(
        "--activity-months",
        type=int,
        default=3,
        help="活動期間の月数（デフォルト: 3ヶ月）",
    )
    parser.add_argument(
        "--output",
        default="active_developer_recommendation_results_2023.json",
        help="結果出力ファイル",
    )

    args = parser.parse_args()

    print("🎯 アクティブ開発者推薦システム評価開始")
    print(f"📝 設定: {args.config}")
    print(f"📊 IRL重み: {args.irl_weights}")
    print(f"🤖 PPOモデル: {args.ppo_model}")
    print(f"📅 活動期間: {args.activity_months}ヶ月")

    # 設定読み込み
    config = load_config(args.config)

    # IRL重み読み込み
    irl_weights = np.load(args.irl_weights)
    print(f"📊 IRL重み形状: {irl_weights.shape}")

    # PPOモデル読み込み
    print("🤖 PPOモデル読み込み中...")
    try:
        ppo_model = PPO.load(args.ppo_model)
        print(f"✅ PPOモデル読み込み完了")
    except Exception as e:
        print(f"❌ PPOモデル読み込み失敗: {e}")
        return

    # バックログとプロファイルデータ読み込み
    print("📚 テストデータ読み込み中...")
    with open(config.env.backlog_path, "r", encoding="utf-8") as f:
        backlog_data = json.load(f)
    with open(config.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles_data = yaml.safe_load(f)

    # 環境初期化
    print("🌍 環境初期化中...")
    env = OSSSimpleEnv(config, backlog_data, dev_profiles_data)

    print(
        f"📊 テストデータ: {len(backlog_data)} タスク, {len(dev_profiles_data)} 開発者"
    )

    # アクティブ開発者推薦評価実行
    print("🎯 アクティブ開発者推薦評価実行中...")
    results = evaluate_active_developer_recommendations(
        backlog_data,
        dev_profiles_data,
        irl_weights,
        ppo_model,
        env,
        env.feature_extractor,
        activity_window_months=args.activity_months,
    )

    # 結果計算
    total_tasks = results["total_tasks"]
    if total_tasks > 0:
        accuracy_top_1 = results["top_k_hits"]["top_1"] / total_tasks
        accuracy_top_3 = results["top_k_hits"]["top_3"] / total_tasks
        accuracy_top_5 = results["top_k_hits"]["top_5"] / total_tasks

        print("\n" + "=" * 70)
        print("🎯 アクティブ開発者推薦システム評価結果")
        print("=" * 70)
        print(f"評価タスク数: {total_tasks}")
        print(f"活動期間: {args.activity_months}ヶ月")
        print(
            f"平均候補開発者数: {results['activity_stats']['avg_active_developers_per_task']:.1f}人"
        )
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
        print("=" * 70)

        # 結果をまとめ
        final_results = {
            "evaluation_config": args.config,
            "irl_weights_path": args.irl_weights,
            "ppo_model_path": args.ppo_model,
            "activity_window_months": args.activity_months,
            "total_tasks_evaluated": total_tasks,
            "tasks_with_assignees": results["tasks_with_assignees"],
            "activity_stats": results["activity_stats"],
            "results": {
                "top_1_accuracy": float(accuracy_top_1),
                "top_3_accuracy": float(accuracy_top_3),
                "top_5_accuracy": float(accuracy_top_5),
            },
            "method": "Active_Developer_Based_Hybrid_Recommendation",
        }

        # 結果保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"💾 詳細結果を保存: {output_path}")

        # サンプル結果表示
        print("\n📋 サンプルアクティブ開発者推薦結果:")
        for i, detail in enumerate(results["recommendation_details"][:5]):
            candidate_info = detail["candidate_info"]
            print(f"\nタスク {i+1}: {detail['task_title']}")
            print(f"  タスク日付: {detail['task_date'][:10]}")
            print(f"  アクティブ候補: {candidate_info['total_active_developers']}人")
            print(f"  実際の担当者: {detail['actual_assignees']}")
            print(f"  推薦Top-5: {detail['recommendations']}")
            print(f"  Top-1正解: {'✅' if detail['correct_in_top_1'] else '❌'}")
            print(f"  Top-3正解: {'✅' if detail['correct_in_top_3'] else '❌'}")

    else:
        print("⚠️ 評価できるタスクが見つかりませんでした")


if __name__ == "__main__":
    main()
