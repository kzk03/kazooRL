#!/usr/bin/env python3
"""
Bot除外で学習されたモデルの推薦システム精度評価

このスクリプトは：
1. 学習時と同じ条件でbot除外とデータ分割を実行
2. 学習済みPPOモデルを読み込み
3. テストデータに対してのみ推薦を実行
4. 正解データ（エキスパートトラジェクトリ）と比較
5. 推薦精度（Top-K accuracy）を計算
"""

import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController


def is_bot_user(username):
    """Check if a username indicates a bot user"""
    if not username:
        return False
    username_lower = username.lower()
    bot_indicators = ["bot", "stale", "dependabot", "renovate", "greenkeeper"]
    return any(indicator in username_lower for indicator in bot_indicators)


def filter_bot_tasks_and_developers(backlog, dev_profiles, expert_trajectories=None):
    """
    Filter out bot tasks and bot developers from data.
    Returns filtered data and statistics.
    """
    print("\n=== Filtering Bot Tasks and Developers ===")

    # Filter developer profiles first
    human_dev_profiles = {}
    bot_developers = []

    for dev_id, profile in dev_profiles.items():
        if is_bot_user(dev_id):
            bot_developers.append(dev_id)
        else:
            human_dev_profiles[dev_id] = profile

    print(f"Original developers: {len(dev_profiles)}")
    print(f"Bot developers filtered out: {len(bot_developers)}")
    print(f"Human developers remaining: {len(human_dev_profiles)}")

    if bot_developers:
        print(f"Bot developers: {bot_developers[:5]}...")  # Show first 5

    # Get expert task assignments (excluding bots)
    expert_assignments = {}
    bot_task_count = 0

    if expert_trajectories:
        for trajectory_episode in expert_trajectories:
            if isinstance(trajectory_episode, list):
                for step in trajectory_episode:
                    if isinstance(step, dict) and "action_details" in step:
                        action_details = step["action_details"]
                        task_id = action_details.get("task_id")
                        developer = action_details.get("developer")

                        if task_id and developer:
                            if is_bot_user(developer):
                                bot_task_count += 1
                                continue
                            expert_assignments[task_id] = developer

    print(f"Expert assignments found: {len(expert_assignments)}")
    print(f"Bot expert assignments excluded: {bot_task_count}")

    # Filter backlog tasks
    human_tasks = []
    bot_tasks = []

    for task in backlog:
        task_id = task.get("id")
        assigned_to = task.get("assigned_to")

        # Check if task is assigned to a bot in the task data
        if assigned_to and is_bot_user(assigned_to):
            bot_tasks.append(task)
            continue

        # Check if task has expert assignment to a bot
        expert_dev = expert_assignments.get(task_id)
        if expert_dev and is_bot_user(expert_dev):
            bot_tasks.append(task)
            continue

        human_tasks.append(task)

    print(f"Original tasks: {len(backlog)}")
    print(f"Bot tasks filtered out: {len(bot_tasks)}")
    print(f"Human tasks remaining: {len(human_tasks)}")

    # Filter expert trajectories if provided
    filtered_trajectories = None
    if expert_trajectories:
        filtered_trajectories = []
        for trajectory_episode in expert_trajectories:
            if isinstance(trajectory_episode, list):
                filtered_episode = []
                for step in trajectory_episode:
                    if isinstance(step, dict) and "action_details" in step:
                        action_details = step["action_details"]
                        task_id = action_details.get("task_id")
                        developer = action_details.get("developer")

                        # Skip if developer is a bot
                        if developer and is_bot_user(developer):
                            continue

                        # Skip if task is not in human tasks
                        if not any(task.get("id") == task_id for task in human_tasks):
                            continue

                        filtered_episode.append(step)

                if filtered_episode:
                    filtered_trajectories.append(filtered_episode)

        print(f"Original expert trajectory episodes: {len(expert_trajectories)}")
        print(
            f"Human expert trajectory episodes remaining: {len(filtered_trajectories)}"
        )

    return human_tasks, human_dev_profiles, filtered_trajectories


def split_tasks_by_time(tasks, train_ratio=0.7):
    """
    Split tasks into train/test based on creation time.
    """
    # Filter tasks with valid dates
    tasks_with_dates = []
    for task in tasks:
        created_at = task.get("created_at")
        if created_at:
            try:
                if isinstance(created_at, str):
                    # Parse ISO format datetime
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    dt = created_at
                tasks_with_dates.append((dt, task))
            except:
                continue

    if not tasks_with_dates:
        print("Warning: No tasks with valid dates found")
        return tasks, []

    # Sort by date
    tasks_with_dates.sort(key=lambda x: x[0])

    # Split by time
    split_idx = int(len(tasks_with_dates) * train_ratio)
    train_tasks = [task for _, task in tasks_with_dates[:split_idx]]
    test_tasks = [task for _, task in tasks_with_dates[split_idx:]]

    print(f"Time-based split: {len(train_tasks)} train, {len(test_tasks)} test")

    return train_tasks, test_tasks


class BotExcludedRecommendationEvaluator:
    """Bot除外で学習されたモデルの推薦システム精度評価クラス"""

    def __init__(self, cfg: DictConfig, train_ratio: float = 0.7):
        self.cfg = cfg
        self.train_ratio = train_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # データの読み込み
        with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
            self.full_backlog = json.load(f)
        with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            self.full_dev_profiles = yaml.safe_load(f)

        # エキスパートデータの読み込み
        self.expert_trajectories = self.load_expert_data()

        # 学習時と同じ条件でデータ分割
        self.split_data()

        # 学習時と同じ条件で環境とコントローラーの初期化
        # 重要: 学習時と同じタスク空間を使用（観測・アクション空間の一致のため）
        self.env = OSSSimpleEnv(
            cfg, self.train_tasks, self.human_dev_profiles
        )  # Use train_tasks for same observation space
        self.controller = IndependentPPOController(self.env, cfg)

        # 学習済みモデルの読み込み
        self.load_trained_models()

    def load_expert_data(self) -> List[Dict]:
        """エキスパート軌跡データを読み込む（正解データとして使用）"""
        expert_path = Path(self.cfg.irl.expert_path)

        if not expert_path.exists():
            print(f"⚠️  Expert data not found: {expert_path}")
            return []

        with open(expert_path, "rb") as f:
            expert_data = pickle.load(f)

        print(f"Loaded {len(expert_data)} expert trajectories")
        return expert_data

    def split_data(self):
        """学習時と同じ条件でデータ分割（botタスクと開発者を除外）"""
        print("Splitting data using same conditions as training...")

        # Use the same filtering logic as training
        human_tasks, human_dev_profiles, filtered_trajectories = (
            filter_bot_tasks_and_developers(
                self.full_backlog, self.full_dev_profiles, self.expert_trajectories
            )
        )

        # Store human developer profiles
        self.human_dev_profiles = human_dev_profiles

        # Split tasks by time (same as training)
        train_tasks, test_tasks = split_tasks_by_time(
            human_tasks, train_ratio=self.train_ratio
        )

        # Store both train and test data (for completeness)
        self.train_tasks = train_tasks  # Add this line
        self.test_backlog = test_tasks

        # Create expert assignments for both train and test sets
        train_task_ids = {task["id"] for task in train_tasks}
        test_task_ids = {task["id"] for task in test_tasks}
        self.train_expert_assignments = {}
        self.test_expert_assignments = {}

        # Process expert trajectories to create assignments
        for trajectory_episode in self.expert_trajectories:
            if isinstance(trajectory_episode, list):
                for step in trajectory_episode:
                    if isinstance(step, dict) and "action_details" in step:
                        action_details = step["action_details"]
                        task_id = action_details.get("task_id")
                        assigned_dev = action_details.get("developer")
                        if task_id and assigned_dev:
                            # Skip bot assignments
                            if is_bot_user(assigned_dev):
                                continue
                            if task_id in train_task_ids:
                                self.train_expert_assignments[task_id] = assigned_dev
                            elif task_id in test_task_ids:
                                self.test_expert_assignments[task_id] = assigned_dev

        print(f"Data split completed (matching training conditions):")
        print(f"  Train tasks: {len(self.train_tasks)}")
        print(f"  Test tasks: {len(self.test_backlog)}")
        print(f"  Human developers: {len(self.human_dev_profiles)}")
        print(f"  Train expert assignments: {len(self.train_expert_assignments)}")
        print(f"  Test expert assignments: {len(self.test_expert_assignments)}")

    def load_trained_models(self):
        """学習済みPPOモデルを読み込む"""
        model_dir = Path(self.cfg.rl.output_model_dir)

        if not model_dir.exists():
            print(f"⚠️  Model directory not found: {model_dir}")
            return

        # 学習メタデータの確認
        metadata_path = model_dir / "training_metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            print(f"Loading models trained with:")
            print(f"  Human developers: {metadata.get('human_developers', 'unknown')}")
            print(f"  Train tasks: {metadata.get('train_tasks', 'unknown')}")
            print(f"  Training completed: {metadata.get('training_completed', False)}")

        print(f"Loading trained models from: {model_dir}")

        loaded_count = 0
        for agent_id in self.controller.agent_ids:
            model_path = model_dir / f"ppo_agent_{agent_id}.pth"
            if model_path.exists():
                try:
                    self.controller.agents[agent_id].load(str(model_path))
                    loaded_count += 1
                    print(f"✅ Loaded model for {agent_id}")
                except Exception as e:
                    print(f"❌ Failed to load model for {agent_id}: {e}")
            else:
                print(f"⚠️  Model not found for {agent_id}: {model_path}")

        print(
            f"Successfully loaded {loaded_count}/{len(self.controller.agent_ids)} models"
        )
        self.models_loaded = loaded_count > 0

    def get_model_recommendations(self, max_k: int = 5) -> List[Tuple[str, List[str]]]:
        """
        学習済みモデルを使用してテストデータに対する推薦を生成（Top-K対応）

        Args:
            max_k: 推薦する開発者の最大数

        Returns:
            List of (task_id, [recommended_developers]) tuples
        """
        print(f"Generating recommendations for {len(self.test_backlog)} test tasks")

        recommendations = []

        if not self.models_loaded:
            print(
                "⚠️  No models loaded successfully. Using random baseline for comparison."
            )
            # ランダム推薦を生成
            import random

            developer_list = list(self.human_dev_profiles.keys())
            for task in self.test_backlog:
                random_devs = random.sample(
                    developer_list, min(max_k, len(developer_list))
                )
                recommendations.append((task["id"], random_devs))
            return recommendations

        # 環境をリセット
        observations = self.env.reset()

        for step in range(len(self.test_backlog)):
            if not self.env.backlog:
                break

            # 現在のタスクを取得
            current_task = self.env.backlog[0]

            # 各開発者に対する行動確率を計算
            developer_scores = {}

            for agent_id in self.controller.agent_ids:
                if agent_id in observations:
                    obs = observations[agent_id]

                    try:
                        # モデルから行動確率を取得
                        with torch.no_grad():
                            obs_tensor = torch.FloatTensor(obs).to(
                                self.controller.agents[agent_id].device
                            )
                            action_probs = (
                                self.controller.agents[agent_id]
                                .policy.actor(obs_tensor)
                                .cpu()
                                .numpy()
                            )

                        # アクション2（タスクを受け入れる）の確率を使用
                        accept_prob = (
                            action_probs[2].item() if len(action_probs) > 2 else 0.0
                        )
                        developer_scores[agent_id] = accept_prob
                    except Exception as e:
                        print(f"⚠️  Model inference error for {agent_id}: {e}")
                        # モデルエラーの場合はランダム値を使用
                        developer_scores[agent_id] = np.random.random()

            # 確率の高い順に開発者をソートしてTop-K推薦を生成
            if developer_scores:
                sorted_developers = sorted(
                    developer_scores.items(), key=lambda x: x[1], reverse=True
                )
                top_k_developers = [dev for dev, score in sorted_developers[:max_k]]
                recommendations.append((current_task.id, top_k_developers))

                # 最も確率の高い開発者がタスクを受け入れる（シミュレーション）
                recommended_dev = top_k_developers[0] if top_k_developers else None
            else:
                recommendations.append((current_task.id, []))
                recommended_dev = None

            # 推薦を実行（シミュレーション）
            actions = {
                agent_id: 0 for agent_id in self.controller.agent_ids
            }  # 全員が待機
            if recommended_dev and recommended_dev in actions:
                actions[recommended_dev] = 2  # 推薦された開発者がタスクを受け入れ

            observations, rewards, terminateds, truncateds, infos = self.env.step(
                actions
            )

            # 終了条件チェック
            if all(terminateds.values()) or all(truncateds.values()):
                break

        print(f"Generated {len(recommendations)} recommendations for test data")
        return recommendations

    def calculate_accuracy(
        self,
        recommendations: List[Tuple[str, List[str]]],
        k_values: List[int] = [1, 3, 5],
    ) -> Dict[str, float]:
        """推薦精度を計算（Top-K対応）"""
        print(
            f"Human test expert assignments available for {len(self.test_expert_assignments)} tasks"
        )

        # 精度計算
        accuracies = {}

        for k in k_values:
            correct_predictions = 0
            valid_recommendations = 0

            for task_id, recommended_devs in recommendations:
                if task_id in self.test_expert_assignments:
                    valid_recommendations += 1
                    expert_dev = self.test_expert_assignments[task_id]

                    # Top-K精度：推薦リストのトップK個に正解が含まれているかチェック
                    top_k_recommendations = recommended_devs[:k]
                    if expert_dev in top_k_recommendations:
                        correct_predictions += 1

            if valid_recommendations > 0:
                accuracy = correct_predictions / valid_recommendations
                accuracies[f"top_{k}_accuracy"] = accuracy
            else:
                accuracies[f"top_{k}_accuracy"] = 0.0

        # 詳細情報の追加
        accuracies["total_valid_recommendations"] = valid_recommendations
        accuracies["total_recommendations"] = len(recommendations)
        accuracies["human_test_set_size"] = len(self.test_expert_assignments)

        return accuracies

    def analyze_baseline_performance(self):
        """ベースライン手法との比較分析"""
        print("\n🔍 Baseline Analysis (Human developers only):")

        num_developers = len(self.human_dev_profiles)

        # ランダムベースライン
        random_accuracy_1 = 1.0 / num_developers
        random_accuracy_3 = min(3.0 / num_developers, 1.0)
        random_accuracy_5 = min(5.0 / num_developers, 1.0)

        print(f"Random baseline ({num_developers} human developers):")
        print(f"  Top-1: {random_accuracy_1:.3f} ({random_accuracy_1*100:.1f}%)")
        print(f"  Top-3: {random_accuracy_3:.3f} ({random_accuracy_3*100:.1f}%)")
        print(f"  Top-5: {random_accuracy_5:.3f} ({random_accuracy_5*100:.1f}%)")

        # 最頻開発者ベースライン
        from collections import Counter

        dev_counts = Counter(self.train_expert_assignments.values())
        most_frequent_devs = [dev for dev, count in dev_counts.most_common(5)]

        print(f"\nMost frequent human developers in training data:")
        for i, (dev, count) in enumerate(dev_counts.most_common(5)):
            print(
                f"  {i+1}. {dev}: {count} assignments ({count/len(self.train_expert_assignments)*100:.1f}%)"
            )

        # 最頻開発者ベースラインの精度計算
        frequent_dev_accuracies = {}
        for k in [1, 3, 5]:
            correct = 0
            for task_id, expert_dev in self.test_expert_assignments.items():
                if expert_dev in most_frequent_devs[:k]:
                    correct += 1
            accuracy = (
                correct / len(self.test_expert_assignments)
                if self.test_expert_assignments
                else 0.0
            )
            frequent_dev_accuracies[f"frequent_dev_top_{k}"] = accuracy

        print(f"\nMost frequent developers baseline:")
        for k in [1, 3, 5]:
            acc = frequent_dev_accuracies[f"frequent_dev_top_{k}"]
            print(f"  Top-{k}: {acc:.3f} ({acc*100:.1f}%)")

        return frequent_dev_accuracies

    def evaluate(self):
        """推薦システムの総合評価を実行"""
        print("🎯 Starting Bot-Excluded Model Evaluation...")
        print(f"Train ratio: {self.train_ratio:.1%}")

        # ベースライン分析
        baseline_results = self.analyze_baseline_performance()

        # 推薦の生成
        print("\n📊 Generating recommendations for test data...")
        recommendations = self.get_model_recommendations(max_k=5)

        # サンプル推薦の表示
        print("\n📋 Sample test recommendations:")
        for i, (task_id, devs) in enumerate(recommendations[:5]):
            top_3_devs = devs[:3] if len(devs) >= 3 else devs
            expert_dev = self.test_expert_assignments.get(task_id, "Unknown")
            print(f"  {i+1}. Task {task_id} → {top_3_devs} (Expert: {expert_dev})")

        # 精度計算
        print("\n🎯 Calculating accuracy on test data...")
        accuracies = self.calculate_accuracy(recommendations)

        # 結果表示
        print("\n📈 Bot-Excluded Model Recommendation Accuracy Results:")
        print("=" * 70)

        model_type = "Trained Model" if self.models_loaded else "Random Baseline"
        print(f"Model Type: {model_type}")

        if accuracies:
            print("Model Performance:")
            for metric, value in accuracies.items():
                if metric.endswith("_accuracy"):
                    print(f"  {metric:20s}: {value:.3f} ({value*100:.1f}%)")

            print(f"\nBaseline Comparisons:")
            print(
                f"  Random Top-1       : {1.0/len(self.human_dev_profiles):.3f} ({100.0/len(self.human_dev_profiles):.1f}%)"
            )
            print(
                f"  Frequent Dev Top-1 : {baseline_results.get('frequent_dev_top_1', 0):.3f} ({baseline_results.get('frequent_dev_top_1', 0)*100:.1f}%)"
            )
        else:
            print("❌ Could not calculate accuracy")

        # 詳細統計
        print(f"\nDetailed Statistics:")
        print(f"  Human test set size: {accuracies.get('human_test_set_size', 0)}")
        print(
            f"  Total test recommendations: {accuracies.get('total_recommendations', len(recommendations))}"
        )
        print(
            f"  Valid human recommendations: {accuracies.get('total_valid_recommendations', 0)}"
        )
        print(f"  Human train set size: {len(self.train_expert_assignments)}")
        print(f"  Models loaded successfully: {self.models_loaded}")

        return accuracies, baseline_results


@hydra.main(version_base="1.3", config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """メイン実行関数"""

    print("🚀 Bot-Excluded Model Recommendation Accuracy Evaluation")
    print("=" * 70)

    try:
        evaluator = BotExcludedRecommendationEvaluator(cfg, train_ratio=0.7)
        model_results, baseline_results = evaluator.evaluate()

        print("\n✅ Evaluation completed successfully!")

        # 結果の保存
        results = {
            "model_performance": model_results,
            "baseline_performance": baseline_results,
            "train_ratio": 0.7,
            "bot_excluded": True,
            "models_loaded": evaluator.models_loaded,
        }

        return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
