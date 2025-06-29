#!/usr/bin/env python3
"""
Bot除外で学習されたモデルの推薦システム精度評価（修正版）

主な修正点:
1. 環境シミュレーションを使わずに直接モデル推論で推薦生成
2. 正しいアクション空間の理解
3. 全テストタスクに対する推薦生成を保証
"""

import json
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController


def filter_bot_tasks(tasks: List[Dict]) -> List[Dict]:
    """ボット関連のタスクを除外"""
    human_tasks = []
    for task in tasks:
        assigned_to = task.get("assigned_to", "")
        if assigned_to and "[bot]" not in assigned_to.lower():
            human_tasks.append(task)
    return human_tasks


def filter_bot_developers(dev_profiles: Dict) -> Dict:
    """ボット開発者を除外"""
    human_devs = {}
    for dev_name, profile in dev_profiles.items():
        if "[bot]" not in dev_name.lower():
            human_devs[dev_name] = profile
    return human_devs


def filter_bot_expert_trajectories(
    expert_trajectories: List, human_developers: set
) -> List:
    """ボット開発者が関与するエキスパート軌跡を除外"""
    human_trajectories = []
    for episode in expert_trajectories:
        human_episode = []
        for step in episode:
            if isinstance(step, dict):
                actions = step.get("actions", {})
                if any(dev in human_developers for dev in actions.keys()):
                    human_episode.append(step)
        if human_episode:
            human_trajectories.append(human_episode)
    return human_trajectories


def split_tasks_by_time(
    tasks: List[Dict], train_ratio: float = 0.7
) -> Tuple[List[Dict], List[Dict]]:
    """タスクを時系列で訓練/テストに分割"""
    tasks_with_dates = []

    for task in tasks:
        created_at = task.get("created_at") or task.get("createdAt")
        if created_at:
            try:
                if isinstance(created_at, str):
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    dt = created_at
                tasks_with_dates.append((dt, task))
            except:
                continue

    if not tasks_with_dates:
        print("Warning: No tasks with valid dates found")
        return tasks, []

    tasks_with_dates.sort(key=lambda x: x[0])
    split_idx = int(len(tasks_with_dates) * train_ratio)
    train_tasks = [task for _, task in tasks_with_dates[:split_idx]]
    test_tasks = [task for _, task in tasks_with_dates[split_idx:]]

    print(f"Time-based split: {len(train_tasks)} train, {len(test_tasks)} test")
    return train_tasks, test_tasks


class FixedBotExcludedRecommendationEvaluator:
    """修正版：Bot除外で学習されたモデルの推薦システム精度評価クラス"""

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

        # 学習時と同じ条件で環境とコントローラーの初期化（観測空間計算用）
        self.env = OSSSimpleEnv(cfg, self.train_tasks, self.human_dev_profiles)
        self.controller = IndependentPPOController(self.env, cfg)

        self.models_loaded = False

    def load_expert_data(self) -> List:
        """エキスパート軌跡データの読み込み"""
        try:
            with open(self.cfg.env.expert_trajectories_path, "rb") as f:
                expert_trajectories = pickle.load(f)
            print(f"Loaded {len(expert_trajectories)} expert trajectories")
            return expert_trajectories
        except Exception as e:
            print(f"Warning: Could not load expert trajectories: {e}")
            return []

    def split_data(self):
        """データを学習時と同じ条件で分割"""
        print("Splitting data using same conditions as training...")

        print("\n=== Filtering Bot Tasks and Developers ===")

        # ボット開発者の除外
        print(f"Original developers: {len(self.full_dev_profiles)}")
        self.human_dev_profiles = filter_bot_developers(self.full_dev_profiles)
        bot_devs = set(self.full_dev_profiles.keys()) - set(
            self.human_dev_profiles.keys()
        )
        print(f"Bot developers filtered out: {len(bot_devs)}")
        print(f"Human developers remaining: {len(self.human_dev_profiles)}")
        if bot_devs:
            print(
                f"Bot developers: {list(bot_devs)[:5]}{'...' if len(bot_devs) > 5 else ''}"
            )

        # エキスパート割り当ての処理
        expert_assignments = {}
        bot_expert_count = 0

        for episode in self.expert_trajectories:
            for step in episode:
                if isinstance(step, dict) and "actions" in step:
                    for dev, action in step["actions"].items():
                        if action != 0:  # 何らかのタスクを選択
                            task_id = step.get("task_id")
                            if task_id:
                                if dev in self.human_dev_profiles:
                                    expert_assignments[task_id] = dev
                                else:
                                    bot_expert_count += 1

        print(f"Expert assignments found: {len(expert_assignments)}")
        print(f"Bot expert assignments excluded: {bot_expert_count}")

        # ボットタスクの除外
        print(f"Original tasks: {len(self.full_backlog)}")
        self.human_backlog = filter_bot_tasks(self.full_backlog)
        print(
            f"Bot tasks filtered out: {len(self.full_backlog) - len(self.human_backlog)}"
        )
        print(f"Human tasks remaining: {len(self.human_backlog)}")

        # エキスパート軌跡の処理
        print(f"Original expert trajectory episodes: {len(self.expert_trajectories)}")
        self.human_expert_trajectories = filter_bot_expert_trajectories(
            self.expert_trajectories, set(self.human_dev_profiles.keys())
        )
        print(
            f"Human expert trajectory episodes remaining: {len(self.human_expert_trajectories)}"
        )

        # 時系列分割
        self.train_tasks, self.test_tasks = split_tasks_by_time(
            self.human_backlog, self.train_ratio
        )

        # 訓練・テスト用エキスパート割り当ての分離
        train_task_ids = {task["id"] for task in self.train_tasks}
        test_task_ids = {task["id"] for task in self.test_tasks}

        self.train_expert_assignments = {
            task_id: dev
            for task_id, dev in expert_assignments.items()
            if task_id in train_task_ids
        }
        self.test_expert_assignments = {
            task_id: dev
            for task_id, dev in expert_assignments.items()
            if task_id in test_task_ids
        }

        print(f"Data split completed (matching training conditions):")
        print(f"  Train tasks: {len(self.train_tasks)}")
        print(f"  Test tasks: {len(self.test_tasks)}")
        print(f"  Human developers: {len(self.human_dev_profiles)}")
        print(f"  Train expert assignments: {len(self.train_expert_assignments)}")
        print(f"  Test expert assignments: {len(self.test_expert_assignments)}")

    def load_models(self, model_dir: Path = Path("models")):
        """訓練済みモデルの読み込み"""
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return

        print(f"Loading models trained with:")
        print(f"  Human developers: unknown")
        print(f"  Train tasks: unknown")
        print(f"  Training completed: False")

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

    def get_task_observation(self, task: Dict, all_tasks: List[Dict]) -> np.ndarray:
        """単一タスクの観測ベクトルを生成（環境の観測空間に対応）"""
        # 環境の観測形式に合わせてタスク状態を作成
        task_states = []

        # 全初期タスクの順序に基づいて観測を作成
        initial_task_ids = [t["id"] for t in all_tasks]

        for task_id in initial_task_ids:
            # 現在評価中のタスクの場合は進行中として扱う
            if task_id == task["id"]:
                status_val = 1  # in_progress
            else:
                status_val = 0  # available

            # complexity, deadline は固定値（実際の環境と一致させる）
            task_states.extend([status_val, 0, 0])

        return np.array(task_states, dtype=np.float32)

    def get_model_recommendations_fixed(
        self, max_k: int = 5
    ) -> List[Tuple[str, List[str]]]:
        """
        修正版：学習済みモデルを使用してテストデータに対する推薦を生成
        環境シミュレーションを使わず、各タスクに対して直接推論
        """
        print(f"Generating recommendations for {len(self.test_tasks)} test tasks")

        if not self.models_loaded:
            print(
                "⚠️  No models loaded successfully. Using random baseline for comparison."
            )
            import random

            developer_list = list(self.human_dev_profiles.keys())
            recommendations = []
            for task in self.test_tasks:
                random_devs = random.sample(
                    developer_list, min(max_k, len(developer_list))
                )
                recommendations.append((task["id"], random_devs))
            return recommendations

        recommendations = []

        # 各テストタスクについて推薦を生成
        for task in self.test_tasks:
            # タスクの観測ベクトルを生成
            task_obs = self.get_task_observation(task, self.train_tasks)

            # 各開発者の推薦スコアを計算
            developer_scores = {}

            for agent_id in self.controller.agent_ids:
                try:
                    # モデルから行動確率を取得
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(task_obs).to(
                            self.controller.agents[agent_id].device
                        )
                        action_logits = self.controller.agents[agent_id].policy.actor(
                            obs_tensor
                        )
                        action_probs = (
                            torch.softmax(action_logits, dim=-1).cpu().numpy()
                        )

                    # タスクを受け入れる確率を計算
                    # 環境では action=0 がタスク選択、action=N（最後）がNO_OP
                    # タスクを受け入れる確率として、最初のタスク（index=0）の確率を使用
                    if len(action_probs) > 0:
                        # NO_OP以外のアクションの最大確率を取る
                        no_op_action = len(action_probs) - 1
                        task_action_probs = (
                            action_probs[:-1] if len(action_probs) > 1 else action_probs
                        )
                        accept_prob = (
                            np.max(task_action_probs)
                            if len(task_action_probs) > 0
                            else 0.0
                        )
                    else:
                        accept_prob = 0.0

                    developer_scores[agent_id] = accept_prob

                except Exception as e:
                    print(f"⚠️  Model inference error for {agent_id}: {e}")
                    developer_scores[agent_id] = np.random.random()

            # 確率の高い順に開発者をソートしてTop-K推薦を生成
            if developer_scores:
                sorted_developers = sorted(
                    developer_scores.items(), key=lambda x: x[1], reverse=True
                )
                top_k_developers = [dev for dev, score in sorted_developers[:max_k]]
                recommendations.append((task["id"], top_k_developers))
            else:
                recommendations.append((task["id"], []))

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

            if len(self.test_expert_assignments) > 0:
                accuracy = correct / len(self.test_expert_assignments)
                frequent_dev_accuracies[f"frequent_dev_top_{k}"] = accuracy
            else:
                frequent_dev_accuracies[f"frequent_dev_top_{k}"] = 0.0

        print(f"\nMost frequent developers baseline:")
        for k in [1, 3, 5]:
            acc = frequent_dev_accuracies.get(f"frequent_dev_top_{k}", 0)
            print(f"  Top-{k}: {acc:.3f} ({acc*100:.1f}%)")

        return frequent_dev_accuracies

    def evaluate(self):
        """推薦システムの総合評価を実行"""
        print("🎯 Starting Fixed Bot-Excluded Model Evaluation...")
        print(f"Train ratio: {self.train_ratio:.1%}")

        # モデルの読み込み
        self.load_models()

        # ベースライン分析
        baseline_results = self.analyze_baseline_performance()

        # 推薦の生成（修正版）
        print("\n📊 Generating recommendations for test data...")
        recommendations = self.get_model_recommendations_fixed(max_k=5)

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
        print("\n📈 Fixed Bot-Excluded Model Recommendation Accuracy Results:")
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

    print("🚀 Fixed Bot-Excluded Model Recommendation Accuracy Evaluation")
    print("=" * 70)

    try:
        evaluator = FixedBotExcludedRecommendationEvaluator(cfg, train_ratio=0.7)
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
