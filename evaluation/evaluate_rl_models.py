#!/usr/bin/env python3
"""
学習済みRLモデル（PPO）を使って2023年テストデータで評価するスクリプト
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


def evaluate_rl_recommendations(
    backlog_data,
    dev_profiles_data,
    rl_model,
    env,
    num_recommendations=5,
):
    """
    学習済みRLモデル（PPO）を使って推薦システムを評価する

    Args:
        backlog_data: テストタスクデータ
        dev_profiles_data: 開発者プロファイルデータ
        rl_model: 学習済みPPOモデル
        env: 環境オブジェクト
        num_recommendations: 推薦する開発者数

    Returns:
        dict: 評価結果（accuracy, precision, recall, etc.）
    """

    results = {
        "total_tasks": 0,
        "tasks_with_assignees": 0,
        "correct_recommendations": 0,
        "top_k_hits": defaultdict(int),
        "recommendation_details": [],
    }

    print(f"🔍 RL評価開始: {len(backlog_data)} タスクで評価")

    # 担当者情報があるタスクのみを抽出
    tasks_with_assignees = []
    for task in backlog_data:
        if task.get("assignees") and len(task["assignees"]) > 0:
            # 担当者がdev_profiles_dataに含まれているかチェック
            assignees = [a.get("login") for a in task["assignees"] if a.get("login")]
            if any(assignee in dev_profiles_data for assignee in assignees):
                tasks_with_assignees.append(task)

    print(f"📊 担当者情報があるタスク: {len(tasks_with_assignees)}/{len(backlog_data)}")

    eval_tasks = tasks_with_assignees
    print(f"🎯 RL評価: {len(eval_tasks)} タスクで評価実行")

    # 評価タスクの進捗バー
    task_progress = tqdm(
        enumerate(eval_tasks),
        total=len(eval_tasks),
        desc="🤖 RL推薦評価",
        unit="task",
        colour="green",
        leave=True,
    )

    for task_idx, task in task_progress:
        # タスクの実際の担当者を取得（Ground Truth）
        actual_assignees = [
            assignee.get("login")
            for assignee in task["assignees"]
            if assignee.get("login")
        ]

        if not actual_assignees:
            task_progress.set_postfix({"Status": "担当者なし (スキップ)"})
            continue

        try:
            # 環境にタスクを設定
            env.reset()

            # タスクを環境に追加（簡易実装）
            task_obj = type(
                "Task",
                (),
                {
                    "id": task.get("id"),
                    "title": task.get("title", ""),
                    "body": task.get("body", ""),
                    "labels": (
                        task.get("labels", [])
                        if isinstance(task.get("labels", []), list)
                        else []
                    ),
                    "updated_at": task.get("updated_at", "2023-01-01T00:00:00Z"),
                    "user": task.get("user", task.get("author", {})),
                    "assignees": task.get("assignees", []),
                },
            )()

            # 利用可能な開発者リストを取得
            available_developers = list(dev_profiles_data.keys())[
                :200
            ]  # 上位200人で評価

            # 各開発者に対してRLモデルで行動確率を計算
            developer_scores = []

            for dev_name in available_developers:
                try:
                    # 開発者プロファイルを取得
                    dev_profile = dev_profiles_data[dev_name]
                    developer = {"name": dev_name, "profile": dev_profile}

                    # 特徴量を抽出（状態として使用）
                    features = env.feature_extractor.get_features(
                        task_obj, developer, env
                    )

                    # RLモデルで行動確率を予測
                    obs = features.reshape(1, -1)  # バッチ次元を追加
                    action_probs = rl_model.predict(obs, deterministic=True)

                    # 行動確率をスコアとして使用
                    if isinstance(action_probs, tuple):
                        score = float(action_probs[0][0])  # 最初の行動の確率
                    else:
                        score = float(action_probs[0])

                    developer_scores.append((dev_name, score))

                except Exception as e:
                    # エラーが発生した場合はスキップ
                    continue

            if not developer_scores:
                task_progress.set_postfix({"Status": "スコア計算失敗"})
                continue

            # スコア順にソート（降順）
            developer_scores.sort(key=lambda x: x[1], reverse=True)

            # 上位N人の推薦リストを作成
            recommendations = [
                dev_name for dev_name, score in developer_scores[:num_recommendations]
            ]

            # 正解率を計算
            correct_in_top_k = []
            for k in [1, 3, 5]:
                top_k_recs = recommendations[:k]
                hit = any(assignee in top_k_recs for assignee in actual_assignees)
                if hit:
                    results["top_k_hits"][f"top_{k}"] += 1
                correct_in_top_k.append(hit)

            # 詳細結果を記録
            results["recommendation_details"].append(
                {
                    "task_id": task.get("id"),
                    "task_title": task.get("title", "Unknown")[:50],
                    "actual_assignees": actual_assignees,
                    "recommendations": recommendations,
                    "top_scores": [
                        (dev, float(score)) for dev, score in developer_scores[:5]
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
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="学習済みRLモデル（PPO）を2023年テストデータで評価"
    )
    parser.add_argument("--config", required=True, help="設定ファイルのパス")
    parser.add_argument(
        "--rl-model", required=True, help="学習済みRLモデルファイルのパス"
    )
    parser.add_argument(
        "--output", default="rl_evaluation_results_2023.json", help="結果出力ファイル"
    )

    args = parser.parse_args()

    print("🚀 2023年テストデータでのRL評価開始")
    print(f"📝 設定: {args.config}")
    print(f"🤖 学習済みRLモデル: {args.rl_model}")

    # 設定読み込み
    config = load_config(args.config)

    # 学習済みRLモデル読み込み
    print("🤖 RLモデル読み込み中...")
    try:
        rl_model = PPO.load(args.rl_model)
        print(f"✅ RLモデル読み込み完了")
    except Exception as e:
        print(f"❌ RLモデル読み込み失敗: {e}")
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

    # RL評価実行
    print("🤖 RL推薦評価実行中...")
    results = evaluate_rl_recommendations(
        backlog_data, dev_profiles_data, rl_model, env
    )

    # 結果計算
    total_tasks = results["total_tasks"]
    if total_tasks > 0:
        accuracy_top_1 = results["top_k_hits"]["top_1"] / total_tasks
        accuracy_top_3 = results["top_k_hits"]["top_3"] / total_tasks
        accuracy_top_5 = results["top_k_hits"]["top_5"] / total_tasks

        print("\n" + "=" * 60)
        print("🤖 RL評価結果")
        print("=" * 60)
        print(f"評価タスク数: {total_tasks}")
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

        # 結果をまとめ
        final_results = {
            "evaluation_config": args.config,
            "rl_model_path": args.rl_model,
            "total_tasks_evaluated": total_tasks,
            "tasks_with_assignees": results["tasks_with_assignees"],
            "top_1_accuracy": float(accuracy_top_1),
            "top_3_accuracy": float(accuracy_top_3),
            "top_5_accuracy": float(accuracy_top_5),
            "detailed_results": results["recommendation_details"],
        }

        # 結果保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        print(f"💾 詳細結果を保存: {output_path}")

        # サンプル結果表示
        print("\n📋 サンプルRL推薦結果:")
        for i, detail in enumerate(results["recommendation_details"][:3]):
            print(f"\nタスク {i+1}: {detail['task_title']}")
            print(f"  実際の担当者: {detail['actual_assignees']}")
            print(f"  RL推薦Top-5: {detail['recommendations']}")
            print(f"  Top-1正解: {'✅' if detail['correct_in_top_1'] else '❌'}")
            print(f"  Top-3正解: {'✅' if detail['correct_in_top_3'] else '❌'}")

    else:
        print("⚠️ 評価できるタスクが見つかりませんでした")


if __name__ == "__main__":
    main()
