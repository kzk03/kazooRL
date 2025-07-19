#!/usr/bin/env python3
"""
実際の担当者を含む適切な候補群での推薦システム評価
"""

import argparse
import json
import os
import sys
from collections import defaultdict
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


def get_smart_candidate_developers(backlog_data, dev_profiles_data, num_candidates=200):
    """
    実際の担当者を含む賢い候補開発者リストを作成
    """
    # 実際の担当者を抽出
    actual_assignees = set()
    for task in backlog_data:
        if task.get("assignees"):
            for assignee in task["assignees"]:
                if assignee.get("login") and assignee["login"] in dev_profiles_data:
                    actual_assignees.add(assignee["login"])

    print(f"🎯 実際の担当者数: {len(actual_assignees)}")
    print(f"📋 実際の担当者: {sorted(actual_assignees)}")

    # プロファイル上位開発者
    all_developers = list(dev_profiles_data.keys())
    top_developers = all_developers[: num_candidates - len(actual_assignees)]

    # 実際の担当者 + 上位開発者の組み合わせ
    candidate_developers = list(actual_assignees) + [
        dev for dev in top_developers if dev not in actual_assignees
    ]

    print(f"👥 候補開発者数: {len(candidate_developers)}")
    print(f"   - 実際の担当者: {len(actual_assignees)}")
    print(f"   - 上位開発者: {len(candidate_developers) - len(actual_assignees)}")

    return candidate_developers, actual_assignees


class ImprovedRecommendationSystem:
    """改良された推薦システム"""

    def __init__(self, irl_weights, ppo_model, env, feature_extractor):
        self.irl_weights = irl_weights
        self.ppo_model = ppo_model
        self.env = env
        self.feature_extractor = feature_extractor
        self.dev_profiles = env.dev_profiles

    def get_task_developer_features(self, task, developer_name):
        """タスクと開発者のペアから特徴量を抽出"""
        try:
            dev_profile = self.dev_profiles[developer_name]
            developer = {"name": developer_name, "profile": dev_profile}
            features = self.feature_extractor.get_features(task, developer, self.env)
            return features
        except Exception as e:
            return None

    def calculate_hybrid_score(self, features, weights=(0.6, 0.3, 0.1)):
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

    def recommend_developers(self, task, candidate_developers, num_recommendations=5):
        """開発者を推薦"""
        developer_scores = []

        for dev_name in candidate_developers:
            features = self.get_task_developer_features(task, dev_name)
            if features is None:
                continue

            total_score, score_details = self.calculate_hybrid_score(features)
            developer_scores.append((dev_name, total_score, score_details))

        # スコア順にソート（降順）
        developer_scores.sort(key=lambda x: x[1], reverse=True)
        return developer_scores[:num_recommendations]


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
            from datetime import datetime

            if isinstance(self.updated_at, str):
                try:
                    if self.updated_at.endswith("Z"):
                        self.updated_at = self.updated_at[:-1] + "+00:00"
                    self.updated_at = datetime.fromisoformat(self.updated_at)
                except:
                    self.updated_at = datetime(2023, 1, 1)

    return MockTask(task_data)


def evaluate_improved_recommendations(
    backlog_data,
    dev_profiles_data,
    irl_weights,
    ppo_model,
    env,
    feature_extractor,
    num_recommendations=5,
):
    """改良された推薦システムを評価"""

    # 賢い候補開発者リストを作成
    candidate_developers, actual_assignees = get_smart_candidate_developers(
        backlog_data, dev_profiles_data, num_candidates=200
    )

    # 推薦システムを初期化
    recommender = ImprovedRecommendationSystem(
        irl_weights, ppo_model, env, feature_extractor
    )

    results = {
        "total_tasks": 0,
        "tasks_with_assignees": 0,
        "correct_recommendations": 0,
        "top_k_hits": defaultdict(int),
        "recommendation_details": [],
        "candidate_info": {
            "total_candidates": len(candidate_developers),
            "actual_assignees_in_candidates": len(actual_assignees),
            "actual_assignees": list(actual_assignees),
        },
    }

    print(f"🤖 改良推薦評価開始: {len(backlog_data)} タスクで評価")

    # 担当者情報があるタスクのみを抽出
    tasks_with_assignees = []
    for task in backlog_data:
        if task.get("assignees") and len(task["assignees"]) > 0:
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in dev_profiles_data for assignee in assignees):
                tasks_with_assignees.append(task)

    print(f"📊 担当者情報があるタスク: {len(tasks_with_assignees)}/{len(backlog_data)}")

    eval_tasks = tasks_with_assignees
    print(f"🎯 改良推薦評価: {len(eval_tasks)} タスクで評価実行")

    # 評価タスクの進捗バー
    task_progress = tqdm(
        enumerate(eval_tasks),
        total=len(eval_tasks),
        desc="🚀 改良推薦評価",
        unit="task",
        colour="green",
        leave=True,
    )

    for task_idx, task in task_progress:
        # タスクの実際の担当者を取得（Ground Truth）
        actual_assignees_task = [
            assignee.get("login")
            for assignee in task["assignees"]
            if assignee.get("login")
        ]

        if not actual_assignees_task:
            task_progress.set_postfix({"Status": "担当者なし (スキップ)"})
            continue

        try:
            # モックタスクオブジェクトを作成
            mock_task = create_mock_task(task)

            # 推薦を実行
            recommendations_with_scores = recommender.recommend_developers(
                mock_task, candidate_developers, num_recommendations
            )

            if not recommendations_with_scores:
                task_progress.set_postfix({"Status": "推薦失敗"})
                continue

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
                    "actual_assignees": actual_assignees_task,
                    "recommendations": recommendations,
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
                task_progress.set_postfix(
                    {
                        "Top-1": f"{top1_acc:.3f}",
                        "Top-3": f"{top3_acc:.3f}",
                        "完了": f"{results['total_tasks']}/{len(eval_tasks)}",
                    }
                )

        except Exception as e:
            print(f"⚠️ タスク {task_idx} でエラー: {e}")
            task_progress.set_postfix({"Status": f"エラー: {str(e)[:20]}"})
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="実際の担当者を含む適切な候補群での推薦システム評価"
    )
    parser.add_argument("--config", required=True, help="設定ファイルのパス")
    parser.add_argument(
        "--irl-weights", required=True, help="学習済みIRL重みファイルのパス"
    )
    parser.add_argument(
        "--ppo-model", required=True, help="学習済みPPOモデルファイルのパス"
    )
    parser.add_argument(
        "--output",
        default="improved_recommendation_results_2023.json",
        help="結果出力ファイル",
    )

    args = parser.parse_args()

    print("🚀 改良推薦システム評価開始")
    print(f"📝 設定: {args.config}")
    print(f"📊 IRL重み: {args.irl_weights}")
    print(f"🤖 PPOモデル: {args.ppo_model}")

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

    # 改良推薦評価実行
    print("🚀 改良推薦評価実行中...")
    results = evaluate_improved_recommendations(
        backlog_data,
        dev_profiles_data,
        irl_weights,
        ppo_model,
        env,
        env.feature_extractor,
    )

    # 結果計算
    total_tasks = results["total_tasks"]
    if total_tasks > 0:
        accuracy_top_1 = results["top_k_hits"]["top_1"] / total_tasks
        accuracy_top_3 = results["top_k_hits"]["top_3"] / total_tasks
        accuracy_top_5 = results["top_k_hits"]["top_5"] / total_tasks

        print("\n" + "=" * 70)
        print("🚀 改良推薦システム評価結果")
        print("=" * 70)
        print(f"評価タスク数: {total_tasks}")
        print(f"候補開発者数: {results['candidate_info']['total_candidates']}")
        print(
            f"実際の担当者が候補に含まれる数: {results['candidate_info']['actual_assignees_in_candidates']}"
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
            "total_tasks_evaluated": total_tasks,
            "tasks_with_assignees": results["tasks_with_assignees"],
            "candidate_info": results["candidate_info"],
            "results": {
                "top_1_accuracy": float(accuracy_top_1),
                "top_3_accuracy": float(accuracy_top_3),
                "top_5_accuracy": float(accuracy_top_5),
            },
            "method": "Improved_Hybrid_Recommendation_With_Actual_Candidates",
        }

        # 結果保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"💾 詳細結果を保存: {output_path}")

        # サンプル結果表示
        print("\n📋 サンプル改良推薦結果:")
        for i, detail in enumerate(results["recommendation_details"][:5]):
            print(f"\nタスク {i+1}: {detail['task_title']}")
            print(f"  実際の担当者: {detail['actual_assignees']}")
            print(f"  改良推薦Top-5: {detail['recommendations']}")
            print(f"  Top-1正解: {'✅' if detail['correct_in_top_1'] else '❌'}")
            print(f"  Top-3正解: {'✅' if detail['correct_in_top_3'] else '❌'}")

    else:
        print("⚠️ 評価できるタスクが見つかりませんでした")


if __name__ == "__main__":
    main()
