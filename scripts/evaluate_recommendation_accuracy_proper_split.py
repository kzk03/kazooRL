#!/usr/bin/env python3
"""
強化学習で学習したモデルを使用した推薦システムの精度評価（適切なデータ分割版）

このスクリプトは：
1. 時系列順にデータを分割（70%学習、30%テスト）
2. 学習済みPPOモデルを読み込み
3. テストデータに対してのみ推薦を実行
4. 正解データ（エキスパートトラジェクトリ）と比較
5. 推薦精度（Top-K accuracy）を計算
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController


class ProperSplitRecommendationEvaluator:
    """適切なデータ分割による推薦システムの精度評価クラス"""

    def __init__(self, cfg: DictConfig, train_ratio: float = 0.7):
        self.cfg = cfg
        self.train_ratio = train_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # データの読み込みと分割
        import json

        import yaml

        with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
            self.full_backlog = json.load(f)
        with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            self.dev_profiles = yaml.safe_load(f)

        # エキスパートデータの読み込み
        self.expert_trajectories = self.load_expert_data()

        # データ分割の実行
        self.split_data()

        # 環境とコントローラーの初期化（テストデータのみ使用）
        self.env = OSSSimpleEnv(cfg, self.test_backlog, self.dev_profiles)
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
        """時系列順にデータを学習・テスト用に分割（botタスクを除外）"""
        # エキスパートデータのタスクIDを取得（botを除外）
        expert_task_ids = set()
        bot_task_count = 0
        for trajectory_episode in self.expert_trajectories:
            for step in trajectory_episode:
                if isinstance(step, dict) and "action_details" in step:
                    task_id = step["action_details"].get("task_id")
                    assigned_dev = step["action_details"].get("developer")
                    if task_id and assigned_dev:
                        # botタスクを除外
                        if (
                            "bot" in assigned_dev.lower()
                            or assigned_dev == "stale[bot]"
                        ):
                            bot_task_count += 1
                            continue
                        expert_task_ids.add(task_id)

        print(f"Excluded {bot_task_count} bot tasks from evaluation")

        # エキスパートデータがあるタスクのみをフィルタリング
        expert_tasks = []
        for task in self.full_backlog:
            if task["id"] in expert_task_ids and "created_at" in task:
                task["created_at_dt"] = datetime.fromisoformat(
                    task["created_at"].replace("Z", "+00:00")
                )
                expert_tasks.append(task)

        # 時系列順にソート
        expert_tasks_sorted = sorted(expert_tasks, key=lambda x: x["created_at_dt"])

        # 分割点を計算
        split_index = int(len(expert_tasks_sorted) * self.train_ratio)

        self.train_tasks = expert_tasks_sorted[:split_index]
        self.test_tasks = expert_tasks_sorted[split_index:]

        # テスト用のバックログを作成（created_at_dtフィールドを削除）
        self.test_backlog = []
        for task in self.test_tasks:
            task_copy = task.copy()
            if "created_at_dt" in task_copy:
                del task_copy["created_at_dt"]
            self.test_backlog.append(task_copy)

        print(f"Data split completed:")
        print(f"  Total expert tasks: {len(expert_tasks_sorted)}")
        print(
            f"  Train tasks: {len(self.train_tasks)} (up to {self.train_tasks[-1]['created_at'] if self.train_tasks else 'N/A'})"
        )
        print(
            f"  Test tasks: {len(self.test_tasks)} (from {self.test_tasks[0]['created_at'] if self.test_tasks else 'N/A'})"
        )

        # トレイン・テスト分割されたエキスパートデータを作成
        train_task_ids = {task["id"] for task in self.train_tasks}
        test_task_ids = {task["id"] for task in self.test_tasks}

        self.train_expert_assignments = {}
        self.test_expert_assignments = {}

        for trajectory_episode in self.expert_trajectories:
            for step in trajectory_episode:
                if isinstance(step, dict) and "action_details" in step:
                    action_details = step["action_details"]
                    task_id = action_details.get("task_id")
                    assigned_dev = action_details.get("developer")
                    if task_id and assigned_dev:
                        # botタスクを除外
                        if (
                            "bot" in assigned_dev.lower()
                            or assigned_dev == "stale[bot]"
                        ):
                            continue
                        if task_id in train_task_ids:
                            self.train_expert_assignments[task_id] = assigned_dev
                        elif task_id in test_task_ids:
                            self.test_expert_assignments[task_id] = assigned_dev

        print(f"  Train expert assignments: {len(self.train_expert_assignments)}")
        print(f"  Test expert assignments: {len(self.test_expert_assignments)}")

    def load_trained_models(self):
        """学習済みPPOモデルを読み込む"""
        model_dir = Path(self.cfg.rl.output_model_dir)

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

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

    def get_model_recommendations(self, max_k: int = 5) -> List[Tuple[str, List[str]]]:
        """
        学習済みモデルを使用してテストデータに対する推薦を生成（Top-K対応）
        モデルが読み込めない場合はランダム推薦を生成

        Args:
            max_k: 推薦する開発者の最大数

        Returns:
            List of (task_id, [recommended_developers]) tuples
            推薦開発者リストは確率の高い順にソート済み
        """
        print(f"Generating recommendations for {len(self.test_backlog)} test tasks")

        recommendations = []

        # モデルが読み込まれているかチェック
        models_loaded = False
        for agent_id in self.controller.agent_ids:
            try:
                # モデルが使用可能かテスト
                test_obs = np.random.rand(451)  # 現在の観測次元
                obs_tensor = torch.FloatTensor(test_obs).to(
                    self.controller.agents[agent_id].device
                )
                with torch.no_grad():
                    action_probs = (
                        self.controller.agents[agent_id]
                        .policy.actor(obs_tensor)
                        .cpu()
                        .numpy()
                    )
                models_loaded = True
                break
            except:
                continue

        if not models_loaded:
            print(
                "⚠️  No models loaded successfully. Using random baseline for comparison."
            )
            # ランダム推薦を生成（botを除外）
            import random

            developer_list = [
                dev
                for dev in self.dev_profiles.keys()
                if not ("bot" in dev.lower() or dev == "stale[bot]")
            ]
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
                    except:
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
        """
        推薦精度を計算（Top-K対応、テストデータのみ使用、botタスクを除外）

        Args:
            recommendations: モデルの推薦結果 [(task_id, [recommended_devs]), ...]
            k_values: Top-K精度を計算するKの値のリスト

        Returns:
            各K値に対する精度の辞書
        """
        # botタスクを除外したテストデータのみを使用
        human_test_assignments = {
            task_id: dev
            for task_id, dev in self.test_expert_assignments.items()
            if not ("bot" in dev.lower() or dev == "stale[bot]")
        }

        print(
            f"Human test expert assignments available for {len(human_test_assignments)} tasks"
        )
        print(
            f"Bot test assignments excluded: {len(self.test_expert_assignments) - len(human_test_assignments)}"
        )

        # 精度計算
        accuracies = {}

        for k in k_values:
            correct_predictions = 0
            valid_recommendations = 0

            for task_id, recommended_devs in recommendations:
                if task_id in human_test_assignments:
                    valid_recommendations += 1
                    expert_dev = human_test_assignments[task_id]

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
        accuracies["human_test_set_size"] = len(human_test_assignments)
        accuracies["bot_test_set_size"] = len(self.test_expert_assignments) - len(
            human_test_assignments
        )

        return accuracies

    def analyze_baseline_performance(self):
        """ベースライン手法との比較分析（botを除外）"""
        print("\n🔍 Baseline Analysis (excluding bots):")

        # 開発者の数を取得（botを除外）
        human_developers = [
            dev
            for dev in self.dev_profiles.keys()
            if not ("bot" in dev.lower() or dev == "stale[bot]")
        ]
        num_developers = len(human_developers)

        # ランダムベースライン
        random_accuracy_1 = 1.0 / num_developers
        random_accuracy_3 = min(3.0 / num_developers, 1.0)
        random_accuracy_5 = min(5.0 / num_developers, 1.0)

        print(f"Random baseline (assuming {num_developers} human developers):")
        print(f"  Top-1: {random_accuracy_1:.3f} ({random_accuracy_1*100:.1f}%)")
        print(f"  Top-3: {random_accuracy_3:.3f} ({random_accuracy_3*100:.1f}%)")
        print(f"  Top-5: {random_accuracy_5:.3f} ({random_accuracy_5*100:.1f}%)")

        # 最頻開発者ベースライン（botを除外）
        from collections import Counter

        human_assignments = {
            task_id: dev
            for task_id, dev in self.train_expert_assignments.items()
            if not ("bot" in dev.lower() or dev == "stale[bot]")
        }
        dev_counts = Counter(human_assignments.values())
        most_frequent_devs = [dev for dev, count in dev_counts.most_common(5)]

        print(f"\nMost frequent human developers in training data:")
        total_human_assignments = len(human_assignments)
        for i, (dev, count) in enumerate(dev_counts.most_common(5)):
            print(
                f"  {i+1}. {dev}: {count} assignments ({count/total_human_assignments*100:.1f}%)"
            )

        # 最頻開発者ベースラインの精度計算（botを除外したテストデータ）
        human_test_assignments = {
            task_id: dev
            for task_id, dev in self.test_expert_assignments.items()
            if not ("bot" in dev.lower() or dev == "stale[bot]")
        }

        frequent_dev_accuracies = {}
        for k in [1, 3, 5]:
            correct = 0
            for task_id, expert_dev in human_test_assignments.items():
                if expert_dev in most_frequent_devs[:k]:
                    correct += 1
            accuracy = (
                correct / len(human_test_assignments) if human_test_assignments else 0.0
            )
            frequent_dev_accuracies[f"frequent_dev_top_{k}"] = accuracy

        print(f"\nMost frequent developers baseline (human tasks only):")
        for k in [1, 3, 5]:
            acc = frequent_dev_accuracies[f"frequent_dev_top_{k}"]
            print(f"  Top-{k}: {acc:.3f} ({acc*100:.1f}%)")

        print(f"\nHuman task statistics:")
        print(f"  Human training tasks: {total_human_assignments}")
        print(f"  Human test tasks: {len(human_test_assignments)}")
        print(
            f"  Bot training tasks excluded: {len(self.train_expert_assignments) - total_human_assignments}"
        )
        print(
            f"  Bot test tasks excluded: {len(self.test_expert_assignments) - len(human_test_assignments)}"
        )

        return frequent_dev_accuracies

    def evaluate(self):
        """推薦システムの総合評価を実行"""
        print(
            "🎯 Starting proper train/test split recommendation accuracy evaluation..."
        )
        print(f"Train ratio: {self.train_ratio:.1%}")

        # ベースライン分析
        baseline_results = self.analyze_baseline_performance()

        # 推薦の生成（テストデータのみ）
        print("\n📊 Generating recommendations for test data...")
        recommendations = self.get_model_recommendations(max_k=5)

        # サンプル推薦の表示（人間のタスクのみ）
        print("\n📋 Sample test recommendations (human tasks only):")
        human_test_assignments = {
            task_id: dev
            for task_id, dev in self.test_expert_assignments.items()
            if not ("bot" in dev.lower() or dev == "stale[bot]")
        }

        sample_count = 0
        for i, (task_id, devs) in enumerate(recommendations):
            if task_id in human_test_assignments and sample_count < 5:
                top_3_devs = devs[:3] if len(devs) >= 3 else devs
                expert_dev = human_test_assignments.get(task_id, "Unknown")
                print(
                    f"  {sample_count+1}. Task {task_id} → {top_3_devs} (Expert: {expert_dev})"
                )
                sample_count += 1

        # 精度計算
        print("\n🎯 Calculating accuracy on test data...")
        accuracies = self.calculate_accuracy(recommendations)

        # 結果表示
        print("\n📈 Test Set Recommendation Accuracy Results (Human Tasks Only):")
        print("=" * 70)

        if accuracies:
            print("Model Performance:")
            for metric, value in accuracies.items():
                if metric.endswith("_accuracy"):
                    print(f"  {metric:20s}: {value:.3f} ({value*100:.1f}%)")

            human_developers = [
                dev
                for dev in self.dev_profiles.keys()
                if not ("bot" in dev.lower() or dev == "stale[bot]")
            ]
            print(f"\nBaseline Comparisons:")
            print(
                f"  Random Top-1       : {1.0/len(human_developers):.3f} ({100.0/len(human_developers):.1f}%)"
            )
            print(
                f"  Frequent Dev Top-1 : {baseline_results.get('frequent_dev_top_1', 0):.3f} ({baseline_results.get('frequent_dev_top_1', 0)*100:.1f}%)"
            )
        else:
            print(
                "❌ Could not calculate accuracy (no test data or valid recommendations)"
            )

        # 詳細統計
        print(f"\nDetailed Statistics:")
        print(f"  Human test set size: {accuracies.get('human_test_set_size', 0)}")
        print(
            f"  Bot test set size (excluded): {accuracies.get('bot_test_set_size', 0)}"
        )
        print(
            f"  Total test recommendations: {accuracies.get('total_recommendations', len(recommendations))}"
        )
        print(
            f"  Valid human recommendations: {accuracies.get('total_valid_recommendations', 0)}"
        )
        print(
            f"  Human train set size: {len([task for task, dev in self.train_expert_assignments.items() if not ('bot' in dev.lower() or dev == 'stale[bot]')])}"
        )

        return accuracies, baseline_results


@hydra.main(version_base="1.3", config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """メイン実行関数"""

    print("🚀 Proper Train/Test Split Recommendation Accuracy Evaluation")
    print("=" * 70)

    try:
        evaluator = ProperSplitRecommendationEvaluator(cfg, train_ratio=0.7)
        model_results, baseline_results = evaluator.evaluate()

        print("\n✅ Evaluation completed successfully!")

        # 結果の保存
        results = {
            "model_performance": model_results,
            "baseline_performance": baseline_results,
            "train_ratio": 0.7,
        }

        return results

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
