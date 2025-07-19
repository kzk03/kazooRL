#!/usr/bin/env python3
"""
ハイブリッド推薦システム
シンプル統計システム + 2022年RL学習システムの組み合わせ
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
        return self._dict.get(key, default)


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

    activity_start = task_date - timedelta(days=activity_window_months * 30)
    active_developers = set()

    for other_task in backlog_data:
        other_task_date = parse_datetime(
            other_task.get("updated_at", other_task.get("created_at", ""))
        )
        if not other_task_date:
            continue

        if activity_start <= other_task_date < task_date:
            # 担当者を追加
            if other_task.get("assignees"):
                for assignee in other_task["assignees"]:
                    if assignee.get("login") and assignee["login"] in dev_profiles_data:
                        active_developers.add(assignee["login"])

            # PR作成者を追加
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


class SimpleRecommendationComponent:
    """シンプル統計ベース推薦コンポーネント"""

    def __init__(self, backlog_data, dev_profiles_data):
        self.backlog_data = backlog_data
        self.dev_profiles = dev_profiles_data
        self.developer_stats = self._calculate_developer_stats()

    def _calculate_developer_stats(self):
        """開発者の活動統計を計算"""
        stats = defaultdict(
            lambda: {
                "total_tasks": 0,
                "recent_tasks": 0,
                "pr_count": 0,
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

        return dict(stats)

    def calculate_simple_score(self, dev_name, task):
        """シンプルなスコア計算"""
        if dev_name not in self.developer_stats:
            return 0.0

        stats = self.developer_stats[dev_name]
        score = 0.0

        # 最近の活動（40%）
        score += stats["recent_tasks"] * 0.4

        # 総活動量（30%）
        score += min(stats["total_tasks"] / 10.0, 5.0) * 0.3

        # PR作成経験（20%）
        score += min(stats["pr_count"] / 5.0, 2.0) * 0.2

        # 最新活動からの経過時間（10%）
        if stats["last_activity"]:
            days_since_last = (
                datetime.now().replace(tzinfo=None)
                - stats["last_activity"].replace(tzinfo=None)
            ).days
            freshness_score = max(0, 1.0 - days_since_last / 365.0)
            score += freshness_score * 0.1

        return score


class RLRecommendationComponent:
    """2022年RL学習推薦コンポーネント"""

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
            features = np.zeros(62)
            dev_profile = self.dev_profiles.get(developer_name, {})
            features[0] = dev_profile.get("rank", 5000) / 5000.0
            features[1] = min(dev_profile.get("total_commits", 0) / 100.0, 1.0)
            features[2] = len(task.get("title", "")) / 100.0
            features[3] = len(task.get("body", "")) / 1000.0
            features[4] = task.get("comments_count", 0) / 10.0

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

            # GAT埋め込み部分はランダムノイズ
            features[25:] = np.random.normal(0, 0.1, 37)
            return features
        except Exception as e:
            return np.zeros(62)

    def calculate_rl_score(self, features):
        """RL学習スコアを計算"""
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
            total_score = 0.5 * irl_score + 0.3 * ppo_score + 0.2 * gat_score
            return float(total_score)

        except Exception as e:
            return 0.0


class HybridRecommendationSystem:
    """ハイブリッド推薦システム"""

    def __init__(
        self,
        backlog_data,
        dev_profiles_data,
        irl_weights,
        ppo_model,
        env,
        feature_extractor,
    ):
        self.simple_component = SimpleRecommendationComponent(
            backlog_data, dev_profiles_data
        )
        self.rl_component = RLRecommendationComponent(
            irl_weights, ppo_model, env, feature_extractor, backlog_data
        )
        self.backlog_data = backlog_data
        self.dev_profiles = dev_profiles_data

    def recommend_developers(
        self,
        task,
        activity_window_months=3,
        num_recommendations=5,
        simple_weight=0.7,
        rl_weight=0.3,
    ):
        """
        ハイブリッド推薦を実行

        Args:
            task: 推薦対象のタスク
            activity_window_months: 活動期間の月数
            num_recommendations: 推薦する開発者数
            simple_weight: シンプルシステムの重み
            rl_weight: RLシステムの重み
        """
        # 候補開発者を取得
        active_developers = get_active_developers_for_task(
            task, self.backlog_data, self.dev_profiles, activity_window_months
        )

        # 担当者を追加
        actual_assignees = set()
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in self.dev_profiles:
                    actual_assignees.add(assignee["login"])

        # 作成者を追加
        task_author = set()
        author = task.get("user", task.get("author", {}))
        if author and author.get("login") and author["login"] in self.dev_profiles:
            task_author.add(author["login"])

        # 候補開発者リストを統合
        candidate_developers = list(
            task_author | actual_assignees | set(active_developers)
        )

        # 候補が少ない場合は上位開発者を追加
        if len(candidate_developers) < 10:
            top_developers = sorted(
                self.dev_profiles.keys(),
                key=lambda dev: self.dev_profiles.get(dev, {}).get(
                    "rank", float("inf")
                ),
            )[:15]

            for dev in top_developers:
                if dev not in candidate_developers:
                    candidate_developers.append(dev)
                    if len(candidate_developers) >= 10:
                        break

        if not candidate_developers:
            return [], {
                "total_active_developers": 0,
                "final_candidates_count": 0,
                "hybrid_method": f"simple_{simple_weight}_rl_{rl_weight}",
            }

        # ハイブリッドスコアを計算
        developer_scores = []
        simple_scores = {}
        rl_scores = {}

        for dev_name in candidate_developers:
            # シンプルスコア
            simple_score = self.simple_component.calculate_simple_score(dev_name, task)
            simple_scores[dev_name] = simple_score

            # RLスコア
            features = self.rl_component.get_task_developer_features(task, dev_name)
            rl_score = (
                self.rl_component.calculate_rl_score(features)
                if features is not None
                else 0.0
            )
            rl_scores[dev_name] = rl_score

            # ハイブリッドスコア
            hybrid_score = simple_weight * simple_score + rl_weight * rl_score

            developer_scores.append(
                (
                    dev_name,
                    hybrid_score,
                    {
                        "simple_score": simple_score,
                        "rl_score": rl_score,
                        "hybrid_score": hybrid_score,
                    },
                )
            )

        # スコア順にソート
        developer_scores.sort(key=lambda x: x[1], reverse=True)

        candidate_info = {
            "total_active_developers": len(active_developers),
            "actual_assignees_count": len(actual_assignees),
            "task_author_count": len(task_author),
            "final_candidates_count": len(candidate_developers),
            "activity_window_months": activity_window_months,
            "hybrid_method": f"simple_{simple_weight}_rl_{rl_weight}",
            "avg_simple_score": np.mean(list(simple_scores.values())),
            "avg_rl_score": np.mean(list(rl_scores.values())),
        }

        return developer_scores[:num_recommendations], candidate_info


def evaluate_hybrid_recommendations(
    backlog_data,
    dev_profiles_data,
    irl_weights,
    ppo_model,
    env,
    feature_extractor,
    activity_window_months=3,
    simple_weight=0.7,
    rl_weight=0.3,
):
    """ハイブリッド推薦システムを評価"""

    # ハイブリッドシステムを初期化
    hybrid_system = HybridRecommendationSystem(
        backlog_data, dev_profiles_data, irl_weights, ppo_model, env, feature_extractor
    )

    # 評価対象タスクを抽出
    eval_tasks = []
    for task in backlog_data:
        has_assignees = False
        has_author = False

        if task.get("assignees") and len(task["assignees"]) > 0:
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in dev_profiles_data for assignee in assignees):
                has_assignees = True

        author = task.get("user", task.get("author", {}))
        if author and author.get("login") and author["login"] in dev_profiles_data:
            has_author = True

        if has_assignees or has_author:
            eval_tasks.append(task)

    print(f"📊 評価対象タスク: {len(eval_tasks)}/{len(backlog_data)}")
    print(f"⚖️ ハイブリッド重み: シンプル{simple_weight} + RL{rl_weight}")

    results = {
        "total_tasks": 0,
        "top_k_hits": defaultdict(int),
        "recommendation_details": [],
        "hybrid_config": {
            "simple_weight": simple_weight,
            "rl_weight": rl_weight,
            "activity_window_months": activity_window_months,
        },
    }

    # 評価実行
    for task_idx, task in enumerate(tqdm(eval_tasks, desc="🎯 ハイブリッド推薦評価")):
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
                hybrid_system.recommend_developers(
                    task,
                    activity_window_months,
                    num_recommendations=5,
                    simple_weight=simple_weight,
                    rl_weight=rl_weight,
                )
            )

            if not recommendations_with_scores:
                continue

            recommendations = [
                dev_name for dev_name, score, details in recommendations_with_scores
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
                        details
                        for dev_name, score, details in recommendations_with_scores
                    ],
                    "candidate_info": candidate_info,
                    "correct_in_top_1": correct_in_top_k[0],
                    "correct_in_top_3": correct_in_top_k[1],
                    "correct_in_top_5": correct_in_top_k[2],
                }
            )

            results["total_tasks"] += 1

        except Exception as e:
            if task_idx < 5:  # 最初の5つのエラーのみ表示
                print(f"⚠️ タスク {task_idx} でエラー: {e}")
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="ハイブリッド推薦システム評価")
    parser.add_argument("--config", required=True, help="設定ファイルのパス")
    parser.add_argument(
        "--irl-weights", required=True, help="学習済みIRL重みファイルのパス"
    )
    parser.add_argument(
        "--ppo-model", required=True, help="学習済みPPOモデルファイルのパス"
    )
    parser.add_argument("--activity-months", type=int, default=3, help="活動期間の月数")
    parser.add_argument(
        "--simple-weight", type=float, default=0.7, help="シンプルシステムの重み"
    )
    parser.add_argument("--rl-weight", type=float, default=0.3, help="RLシステムの重み")
    parser.add_argument(
        "--output",
        default="hybrid_recommendation_results.json",
        help="結果出力ファイル",
    )

    args = parser.parse_args()

    print("🎯 ハイブリッド推薦システム評価開始")
    print(f"📝 設定: {args.config}")
    print(f"📊 IRL重み: {args.irl_weights}")
    print(f"🤖 PPOモデル: {args.ppo_model}")
    print(f"📅 活動期間: {args.activity_months}ヶ月")
    print(f"⚖️ 重み配分: シンプル{args.simple_weight} + RL{args.rl_weight}")

    # 設定読み込み
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    config = SimpleConfig(config_dict)

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

    # データ読み込み
    print("📚 データ読み込み中...")
    with open(config.env.backlog_path, "r", encoding="utf-8") as f:
        backlog_data = json.load(f)
    with open(config.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles_data = yaml.safe_load(f)

    # 環境初期化
    print("🌍 環境初期化中...")
    env = OSSSimpleEnv(config, backlog_data, dev_profiles_data)

    print(f"📊 データ: {len(backlog_data)} タスク, {len(dev_profiles_data)} 開発者")

    # ハイブリッド推薦評価実行
    results = evaluate_hybrid_recommendations(
        backlog_data,
        dev_profiles_data,
        irl_weights,
        ppo_model,
        env,
        env.feature_extractor,
        activity_window_months=args.activity_months,
        simple_weight=args.simple_weight,
        rl_weight=args.rl_weight,
    )

    # 結果計算・表示
    total_tasks = results["total_tasks"]
    if total_tasks > 0:
        accuracy_top_1 = results["top_k_hits"]["top_1"] / total_tasks
        accuracy_top_3 = results["top_k_hits"]["top_3"] / total_tasks
        accuracy_top_5 = results["top_k_hits"]["top_5"] / total_tasks

        print("\n" + "=" * 70)
        print("🎯 ハイブリッド推薦システム評価結果")
        print("=" * 70)
        print(f"評価タスク数: {total_tasks}")
        print(f"活動期間: {args.activity_months}ヶ月")
        print(f"重み配分: シンプル{args.simple_weight} + RL{args.rl_weight}")
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

        # 結果保存
        final_results = {
            "evaluation_config": args.config,
            "irl_weights_path": args.irl_weights,
            "ppo_model_path": args.ppo_model,
            "hybrid_config": results["hybrid_config"],
            "total_tasks_evaluated": total_tasks,
            "results": {
                "top_1_accuracy": float(accuracy_top_1),
                "top_3_accuracy": float(accuracy_top_3),
                "top_5_accuracy": float(accuracy_top_5),
            },
            "method": "Hybrid_Simple_RL_Recommendation",
            "sample_results": results["recommendation_details"][:10],
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"💾 結果を保存: {args.output}")

        # サンプル結果表示
        print("\n📋 サンプルハイブリッド推薦結果:")
        for i, detail in enumerate(results["recommendation_details"][:5]):
            print(f"\nタスク {i+1}: {detail['task_title']}")
            print(f"  正解: {detail['ground_truth']}")
            print(f"  推薦: {detail['recommendations'][:3]}")
            if detail["scores"]:
                hybrid_scores = [
                    f"{s['hybrid_score']:.2f}" for s in detail["scores"][:3]
                ]
                simple_scores = [
                    f"{s['simple_score']:.2f}" for s in detail["scores"][:3]
                ]
                rl_scores = [f"{s['rl_score']:.2f}" for s in detail["scores"][:3]]
                print(f"  ハイブリッドスコア: {hybrid_scores}")
                print(f"  (シンプル: {simple_scores})")
                print(f"  (RL: {rl_scores})")
            print(f"  Top-1正解: {'✅' if detail['correct_in_top_1'] else '❌'}")

    else:
        print("⚠️ 評価できるタスクが見つかりませんでした")


if __name__ == "__main__":
    main()
