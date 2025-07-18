#!/usr/bin/env python3
"""
実際の開発者行動とRLエージェントの予測比較評価

学習済みモデルが実際の開発者の選択をどの程度予測できるかを評価する
"""

import argparse
import json
# プロジェクトのモジュール
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
# Stable-Baselines3
from stable_baselines3 import PPO

sys.path.append("/Users/kazuki-h/rl/kazoo")
sys.path.append("/Users/kazuki-h/rl/kazoo/src")

from scripts.train_simple_unified_rl import SimpleTaskAssignmentEnv
from src.kazoo.envs.task import Task


class BehavioralComparison:
    """実際の開発者行動とRLエージェントの予測比較"""

    def __init__(self, config_path, model_path, test_data_path):
        """
        Args:
            config_path: 設定ファイルパス
            model_path: 学習済みモデルパス
            test_data_path: テストデータパス
        """
        self.config_path = config_path
        self.model_path = model_path
        self.test_data_path = test_data_path

        # 設定読み込み
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # テストデータ読み込み
        with open(test_data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # 2023年のデータのみ抽出（テスト用）
        self.test_data = []
        for task in all_data:
            created_at = task.get("created_at", "")
            if created_at.startswith("2023"):
                self.test_data.append(task)

        print(f"📊 全データ: {len(all_data):,} タスク")
        print(f"📊 2023年テストデータ: {len(self.test_data):,} タスク")

        # 環境初期化
        self._setup_environment()

        # モデル読み込み
        self.model = PPO.load(model_path)
        print(f"✅ モデル読み込み完了: {model_path}")

        # 結果格納
        self.results = {
            "predictions": [],
            "actuals": [],
            "tasks": [],
            "developers": [],
            "accuracies": {},
        }

    def _setup_environment(self):
        """環境の初期化"""
        print("🎮 環境初期化中...")

        # 開発者プロファイル読み込み
        dev_profiles_path = self.config["env"]["dev_profiles_path"]
        with open(dev_profiles_path, "r", encoding="utf-8") as f:
            self.dev_profiles = yaml.safe_load(f)

        # 学習に使用したバックログデータも読み込み（開発者プール確認のため）
        backlog_path = self.config["env"]["backlog_path"]
        with open(backlog_path, "r", encoding="utf-8") as f:
            training_backlog = json.load(f)

        print(f"   学習用バックログ: {len(training_backlog):,} タスク")

        # 設定オブジェクトの作成（dictベースで属性アクセス可能）
        class DictObj:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, DictObj(v))
                    else:
                        setattr(self, k, v)

            def get(self, key, default=None):
                return getattr(self, key, default)

        cfg = DictObj(self.config)

        # 環境作成（学習時のデータ構成を使用）
        self.env = SimpleTaskAssignmentEnv(
            cfg=cfg,
            backlog_data=training_backlog,  # 学習データを使用して開発者プールを確定
            dev_profiles_data=self.dev_profiles,
        )

        print(f"   開発者数: {self.env.num_developers}")
        print(f"   タスク数: {len(self.env.tasks)}")
        print(f"   特徴量次元: {self.env.observation_space.shape[0]}")

    def extract_actual_assignments(self):
        """実際の開発者割り当てを抽出"""
        print("📋 実際の割り当て抽出中...")

        actual_assignments = {}
        assignment_stats = defaultdict(int)
        developer_stats = Counter()

        for task_data in self.test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            # 実際の担当者を抽出
            assignee = None

            # assignees フィールドから抽出
            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")

            # events から ASSIGNED イベントを探す
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            # comments から @メンションを探す（簡易）
            elif "comments" in task_data:
                for comment in task_data["comments"]:
                    body = comment.get("body", "")
                    if "@" in body and "assign" in body.lower():
                        # より詳細な解析が必要
                        pass

            if assignee:
                actual_assignments[task_id] = assignee
                assignment_stats["assigned"] += 1
                developer_stats[assignee] += 1
            else:
                assignment_stats["unassigned"] += 1

        print(f"   割り当て済み: {assignment_stats['assigned']:,} タスク")
        print(f"   未割り当て: {assignment_stats['unassigned']:,} タスク")
        print(f"   ユニーク開発者: {len(developer_stats)} 人")

        # 上位開発者表示
        top_devs = developer_stats.most_common(10)
        print("   上位開発者:")
        for dev, count in top_devs:
            print(f"     {dev}: {count} タスク")

        return actual_assignments

    def predict_assignments(self, actual_assignments):
        """RLエージェントによる割り当て予測"""
        print("🤖 RLエージェント予測中...")

        predictions = {}
        prediction_scores = {}

        # 実際に割り当てられた2023年のタスクを対象に予測
        test_tasks_with_assignments = []
        for task in self.test_data:
            task_id = task.get("id") or task.get("number")
            if task_id and task_id in actual_assignments:
                test_tasks_with_assignments.append(task)

        print(f"   予測対象: {len(test_tasks_with_assignments)} タスク（2023年）")

        for task_data in test_tasks_with_assignments:
            try:
                # TaskオブジェクトとしてTaskを作成
                from src.kazoo.envs.task import Task

                task_obj = Task(task_data)
                task_id = (
                    task_obj.id
                    if hasattr(task_obj, "id")
                    else task_data.get("id", task_data.get("number"))
                )

                # 環境の最初の開発者を使用して特徴量を計算
                first_dev_name = self.env.developers[0]
                first_dev_profile = self.dev_profiles.get(first_dev_name, {})
                dev_obj = {"name": first_dev_name, "profile": first_dev_profile}

                # ダミー環境でベース特徴量を計算
                dummy_env = type(
                    "DummyEnv",
                    (),
                    {
                        "backlog": self.env.tasks,
                        "dev_profiles": self.dev_profiles,
                        "assignments": {},
                        "dev_action_history": {},
                    },
                )()

                # 各開発者に対する特徴量と予測確率を計算
                dev_predictions = []
                all_probabilities = np.zeros(len(self.env.developers))

                # タスクに対する基本特徴量を取得（開発者非依存部分）
                base_features = None

                for dev_idx, dev_name in enumerate(self.env.developers):
                    dev_profile = self.dev_profiles.get(dev_name, {})
                    dev_obj = {"name": dev_name, "profile": dev_profile}

                    # 特徴量を抽出
                    features = self.env.feature_extractor.get_features(
                        task_obj, dev_obj, dummy_env
                    )
                    obs = features.astype(np.float32)

                    # モデルで予測（確率分布）
                    obs_tensor = torch.tensor(obs).unsqueeze(0).float()
                    with torch.no_grad():
                        logits = self.model.policy.mlp_extractor.policy_net(
                            self.model.policy.features_extractor(obs_tensor)
                        )
                        probs = torch.softmax(logits, dim=-1).numpy()[0]
                        dev_prob = probs[dev_idx] if dev_idx < len(probs) else 0.0
                        dev_predictions.append((dev_idx, dev_name, dev_prob))
                        all_probabilities[dev_idx] = dev_prob

                # 最高確率の開発者を選択
                dev_predictions.sort(key=lambda x: x[2], reverse=True)
                top_dev_idx, top_dev_name, top_prob = dev_predictions[0]

                predictions[task_id] = top_dev_name
                prediction_scores[task_id] = {
                    "top_action": top_dev_idx,
                    "top_developer": top_dev_name,
                    "top_probability": top_prob,
                    "probabilities": all_probabilities,  # 全開発者の確率
                    "top_5_actions": [p[0] for p in dev_predictions[:5]],
                    "top_5_probs": [p[2] for p in dev_predictions[:5]],
                    "top_5_developers": [p[1] for p in dev_predictions[:5]],
                }

            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測でエラー: {e}")
                continue

        print(f"   予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores

    def calculate_metrics(self, actual_assignments, predictions, prediction_scores):
        """評価指標の計算"""
        print("📊 評価指標計算中...")

        # 共通のタスクのみ評価
        common_tasks = set(actual_assignments.keys()) & set(predictions.keys())
        print(f"   評価対象タスク: {len(common_tasks)}")

        if not common_tasks:
            print("⚠️ 評価可能なタスクがありません")
            return {}

        # 開発者名のマッピング
        all_actual_devs = set(actual_assignments.values())
        all_predicted_devs = set(predictions.values())
        all_env_devs = set(self.env.developers)

        print(f"   実際の開発者: {len(all_actual_devs)} 人")
        print(f"   予測開発者: {len(all_predicted_devs)} 人")
        print(f"   環境の開発者: {len(all_env_devs)} 人")

        # 開発者プールの重複チェック
        overlap_devs = all_actual_devs & all_env_devs
        print(f"   重複する開発者: {len(overlap_devs)} 人")

        if overlap_devs:
            print("   重複開発者:", sorted(list(overlap_devs)))

        # 正確に一致するケース
        exact_matches = 0
        dev_mapping_matches = 0

        actuals = []
        predicted_actions = []

        # 重複開発者のタスクのみで評価
        overlap_task_count = 0
        overlap_exact_matches = 0

        for task_id in common_tasks:
            actual_dev = actual_assignments[task_id]
            predicted_dev = predictions[task_id]

            # 正確な一致
            if actual_dev == predicted_dev:
                exact_matches += 1

            # 重複開発者のタスクでの評価
            if actual_dev in overlap_devs:
                overlap_task_count += 1
                if actual_dev == predicted_dev:
                    overlap_exact_matches += 1

            # 開発者インデックスベースの評価
            try:
                actual_idx = self.env.developers.index(actual_dev)
                predicted_idx = self.env.developers.index(predicted_dev)

                actuals.append(actual_idx)
                predicted_actions.append(predicted_idx)

                if actual_idx == predicted_idx:
                    dev_mapping_matches += 1

            except ValueError:
                # 環境に存在しない開発者の場合はスキップ
                continue

        metrics = {}

        # 基本的な精度指標
        metrics["exact_accuracy"] = exact_matches / len(common_tasks)
        metrics["mapping_accuracy"] = (
            dev_mapping_matches / len(actuals) if actuals else 0
        )

        # 重複開発者での精度（より公平な評価）
        if overlap_task_count > 0:
            metrics["overlap_accuracy"] = overlap_exact_matches / overlap_task_count
            print(
                f"   重複開発者精度: {overlap_exact_matches}/{overlap_task_count} = {metrics['overlap_accuracy']:.3f}"
            )
        else:
            metrics["overlap_accuracy"] = 0.0
            print("   重複開発者精度: N/A（重複開発者なし）")

        print(
            f"   全体正確一致: {exact_matches}/{len(common_tasks)} = {metrics['exact_accuracy']:.3f}"
        )
        print(
            f"   マッピング一致: {dev_mapping_matches}/{len(actuals)} = {metrics['mapping_accuracy']:.3f}"
        )

        # 推薦システムとしての評価指標
        recommendation_metrics = self._calculate_recommendation_metrics(
            actual_assignments, predictions, prediction_scores, common_tasks
        )
        metrics.update(recommendation_metrics)

        # Top-K精度の計算
        if actuals and len(actuals) == len(predicted_actions):
            try:
                # Top-K精度を手動計算
                top_k_scores = {}
                for k in [1, 3, 5, 10]:
                    if k <= len(self.env.developers):
                        top_k_correct = 0
                        top_k_overlap_correct = 0
                        top_k_overlap_total = 0

                        for task_id in common_tasks:
                            if task_id in prediction_scores:
                                actual_dev = actual_assignments[task_id]

                                # 全体でのTop-K精度
                                if actual_dev in self.env.developers:
                                    actual_idx = self.env.developers.index(actual_dev)
                                    top_k_actions = prediction_scores[task_id][
                                        "top_5_actions"
                                    ][:k]
                                    if actual_idx in top_k_actions:
                                        top_k_correct += 1

                                # 重複開発者でのTop-K精度
                                if actual_dev in overlap_devs:
                                    top_k_overlap_total += 1
                                    actual_idx = self.env.developers.index(actual_dev)
                                    top_k_actions = prediction_scores[task_id][
                                        "top_5_actions"
                                    ][:k]
                                    if actual_idx in top_k_actions:
                                        top_k_overlap_correct += 1

                        top_k_scores[f"top_{k}_accuracy"] = top_k_correct / len(
                            common_tasks
                        )
                        if top_k_overlap_total > 0:
                            top_k_scores[f"top_{k}_overlap_accuracy"] = (
                                top_k_overlap_correct / top_k_overlap_total
                            )
                        else:
                            top_k_scores[f"top_{k}_overlap_accuracy"] = 0.0

                        print(
                            f"   Top-{k}精度: {top_k_correct}/{len(common_tasks)} = {top_k_scores[f'top_{k}_accuracy']:.3f}"
                        )
                        if top_k_overlap_total > 0:
                            print(
                                f"   Top-{k}重複精度: {top_k_overlap_correct}/{top_k_overlap_total} = {top_k_scores[f'top_{k}_overlap_accuracy']:.3f}"
                            )

                metrics.update(top_k_scores)

            except Exception as e:
                print(f"⚠️ Top-K精度計算エラー: {e}")

        # 開発者レベルの統計
        dev_stats = self._calculate_developer_stats(
            actual_assignments, predictions, common_tasks
        )
        metrics["developer_stats"] = dev_stats

        return metrics

    def _calculate_recommendation_metrics(
        self, actual_assignments, predictions, prediction_scores, common_tasks
    ):
        """推薦システムとしての評価指標"""
        print("🎯 推薦システム指標計算中...")

        metrics = {}

        # 1. 信頼度分析（予測確率の分布）
        confidence_scores = []
        correct_confidences = []
        incorrect_confidences = []

        for task_id in common_tasks:
            if task_id in prediction_scores:
                actual_dev = actual_assignments[task_id]
                predicted_dev = predictions[task_id]

                # 予測の信頼度（最高確率）
                max_prob = max(prediction_scores[task_id]["top_5_probs"])
                confidence_scores.append(max_prob)

                if actual_dev == predicted_dev:
                    correct_confidences.append(max_prob)
                else:
                    incorrect_confidences.append(max_prob)

        if confidence_scores:
            metrics["avg_confidence"] = np.mean(confidence_scores)
            metrics["confidence_std"] = np.std(confidence_scores)

            if correct_confidences:
                metrics["correct_avg_confidence"] = np.mean(correct_confidences)
            if incorrect_confidences:
                metrics["incorrect_avg_confidence"] = np.mean(incorrect_confidences)

            print(
                f"   平均予測信頼度: {metrics['avg_confidence']:.3f} ± {metrics['confidence_std']:.3f}"
            )
            if correct_confidences and incorrect_confidences:
                print(f"   正解時信頼度: {metrics['correct_avg_confidence']:.3f}")
                print(f"   不正解時信頼度: {metrics['incorrect_avg_confidence']:.3f}")

        # 2. ランキング精度（実際の開発者が何位に予測されるか）
        ranking_positions = []
        ranking_similarities = []

        for task_id in common_tasks:
            if task_id in prediction_scores:
                actual_dev = actual_assignments[task_id]
                if actual_dev in self.env.developers:
                    actual_idx = self.env.developers.index(actual_dev)

                    # 全開発者に対する予測確率を取得
                    if "probabilities" in prediction_scores[task_id]:
                        all_probs = prediction_scores[task_id]["probabilities"]
                    else:
                        # top_5_probsから全体を推定
                        all_probs = np.zeros(len(self.env.developers))
                        top_actions = prediction_scores[task_id].get(
                            "top_5_actions", []
                        )
                        top_probs = prediction_scores[task_id].get("top_5_probs", [])
                        for act, prob in zip(top_actions, top_probs):
                            if act < len(all_probs):
                                all_probs[act] = prob

                    # ランキング位置を計算
                    sorted_indices = np.argsort(all_probs)[::-1]  # 降順
                    try:
                        position = (
                            np.where(sorted_indices == actual_idx)[0][0] + 1
                        )  # 1-based
                        ranking_positions.append(position)

                        # ランキング類似度計算（位置ベース）
                        # 1位なら1.0、最下位なら0.0に近づく
                        similarity = 1.0 - (position - 1) / (
                            len(self.env.developers) - 1
                        )
                        ranking_similarities.append(similarity)

                    except (IndexError, ValueError):
                        # 見つからない場合は最下位扱い
                        ranking_positions.append(len(self.env.developers))
                        ranking_similarities.append(0.0)

        if ranking_positions:
            metrics["avg_ranking_position"] = np.mean(ranking_positions)
            metrics["median_ranking_position"] = np.median(ranking_positions)
            metrics["avg_ranking_similarity"] = np.mean(ranking_similarities)
            metrics["ranking_top_1_ratio"] = sum(
                1 for p in ranking_positions if p == 1
            ) / len(ranking_positions)
            metrics["ranking_top_5_ratio"] = sum(
                1 for p in ranking_positions if p <= 5
            ) / len(ranking_positions)
            metrics["ranking_top_10_ratio"] = sum(
                1 for p in ranking_positions if p <= 10
            ) / len(ranking_positions)

            print(f"   平均ランキング位置: {metrics['avg_ranking_position']:.2f}")
            print(f"   中央値ランキング位置: {metrics['median_ranking_position']:.1f}")
            print(f"   平均ランキング類似度: {metrics['avg_ranking_similarity']:.3f}")
            print(f"   Top-1率: {metrics['ranking_top_1_ratio']:.3f}")
            print(f"   Top-5率: {metrics['ranking_top_5_ratio']:.3f}")
            print(f"   Top-10率: {metrics['ranking_top_10_ratio']:.3f}")

        # 3. スピアマン順位相関（開発者プール重複部分のみ）
        spearman_correlations = []
        overlap_devs = set(actual_assignments.values()) & set(self.env.developers)

        if len(overlap_devs) >= 2:  # 相関計算には最低2人必要
            for task_id in common_tasks:
                if task_id in prediction_scores:
                    actual_dev = actual_assignments[task_id]
                    if actual_dev in overlap_devs:
                        # 重複開発者のみでランキングを作成
                        overlap_indices = [
                            self.env.developers.index(dev) for dev in overlap_devs
                        ]

                        if "probabilities" in prediction_scores[task_id]:
                            all_probs = prediction_scores[task_id]["probabilities"]
                            overlap_probs = [
                                all_probs[idx] if idx < len(all_probs) else 0.0
                                for idx in overlap_indices
                            ]
                        else:
                            # デフォルト確率
                            overlap_probs = [0.1] * len(overlap_indices)

                        # 実際のランキング（実際の開発者が1位）
                        actual_ranking = [
                            1 if dev == actual_dev else 2 for dev in overlap_devs
                        ]

                        # 予測ランキング
                        predicted_ranking = (
                            np.argsort(np.argsort(overlap_probs)[::-1]) + 1
                        )

                        if (
                            len(set(actual_ranking)) > 1
                            and len(set(predicted_ranking)) > 1
                        ):
                            # シンプルなスピアマン順位相関の実装
                            def simple_spearman(x, y):
                                n = len(x)
                                if n != len(y):
                                    return 0.0

                                # 順位に変換
                                rank_x = np.argsort(np.argsort(x))
                                rank_y = np.argsort(np.argsort(y))

                                # スピアマン相関計算
                                d_squared = np.sum((rank_x - rank_y) ** 2)
                                correlation = 1 - (6 * d_squared) / (n * (n**2 - 1))
                                return correlation

                            correlation = simple_spearman(
                                actual_ranking, predicted_ranking
                            )
                            if not np.isnan(correlation):
                                spearman_correlations.append(correlation)

            if spearman_correlations:
                metrics["avg_spearman_correlation"] = np.mean(spearman_correlations)
                print(
                    f"   平均スピアマン順位相関: {metrics['avg_spearman_correlation']:.3f}"
                )

        # 4. 実際の開発者群との類似度（コサイン類似度）
        developer_similarity_scores = []

        for task_id in common_tasks:
            if task_id in prediction_scores:
                actual_dev = actual_assignments[task_id]

                # 実際の開発者ベクトル（ワンホット）
                actual_vector = np.zeros(len(self.env.developers))
                if actual_dev in self.env.developers:
                    actual_idx = self.env.developers.index(actual_dev)
                    actual_vector[actual_idx] = 1.0

                # 予測確率ベクトル
                if "probabilities" in prediction_scores[task_id]:
                    predicted_vector = np.array(
                        prediction_scores[task_id]["probabilities"]
                    )
                else:
                    predicted_vector = np.zeros(len(self.env.developers))
                    top_actions = prediction_scores[task_id].get("top_5_actions", [])
                    top_probs = prediction_scores[task_id].get("top_5_probs", [])
                    for act, prob in zip(top_actions, top_probs):
                        if act < len(predicted_vector):
                            predicted_vector[act] = prob

                # コサイン類似度計算
                if (
                    np.linalg.norm(actual_vector) > 0
                    and np.linalg.norm(predicted_vector) > 0
                ):
                    cosine_sim = np.dot(actual_vector, predicted_vector) / (
                        np.linalg.norm(actual_vector) * np.linalg.norm(predicted_vector)
                    )
                    developer_similarity_scores.append(cosine_sim)

        if developer_similarity_scores:
            metrics["avg_cosine_similarity"] = np.mean(developer_similarity_scores)
            metrics["cosine_similarity_std"] = np.std(developer_similarity_scores)
            print(
                f"   平均コサイン類似度: {metrics['avg_cosine_similarity']:.3f} ± {metrics['cosine_similarity_std']:.3f}"
            )

        # 5. 推薦多様性とカバレッジ（元のロジックを維持）
        predicted_dev_counts = Counter(predictions.values())
        total_predictions = len(predictions)

        if total_predictions > 0:
            # ジニ係数（不平等度）の計算
            counts = list(predicted_dev_counts.values())
            counts.sort()
            n = len(counts)
            gini = (2 * sum((i + 1) * x for i, x in enumerate(counts))) / (
                n * sum(counts)
            ) - (n + 1) / n

            metrics["prediction_diversity"] = 1 - gini  # 多様性は不平等度の逆
            metrics["unique_predicted_devs"] = len(predicted_dev_counts)
            metrics["prediction_concentration"] = (
                max(predicted_dev_counts.values()) / total_predictions
            )

            print(f"   予測多様性: {metrics['prediction_diversity']:.3f}")
            print(f"   ユニーク予測開発者数: {metrics['unique_predicted_devs']}")
            print(f"   予測集中度: {metrics['prediction_concentration']:.3f}")

        return metrics

    def _calculate_developer_stats(self, actual_assignments, predictions, common_tasks):
        """開発者レベルの統計"""
        dev_stats = {
            "actual_distribution": Counter(),
            "predicted_distribution": Counter(),
            "per_developer_accuracy": {},
        }

        for task_id in common_tasks:
            actual_dev = actual_assignments[task_id]
            predicted_dev = predictions[task_id]

            dev_stats["actual_distribution"][actual_dev] += 1
            dev_stats["predicted_distribution"][predicted_dev] += 1

        # 開発者別精度
        for dev in dev_stats["actual_distribution"]:
            dev_tasks = [tid for tid in common_tasks if actual_assignments[tid] == dev]
            correct = sum(1 for tid in dev_tasks if predictions.get(tid) == dev)
            dev_stats["per_developer_accuracy"][dev] = (
                correct / len(dev_tasks) if dev_tasks else 0
            )

        return dev_stats

    def generate_report(self, metrics, output_dir):
        """評価レポート生成"""
        print("📄 レポート生成中...")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # メトリクス保存
        metrics_path = output_dir / f"behavioral_comparison_metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            # NumPy配列を対応
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                return obj

            json.dump(metrics, f, indent=2, ensure_ascii=False, default=convert_numpy)

        print(f"✅ メトリクス保存: {metrics_path}")

        # 可視化
        self._create_visualizations(metrics, output_dir, timestamp)

        # テキストレポート
        report_path = output_dir / f"behavioral_comparison_report_{timestamp}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("実際の開発者行動 vs RLエージェント予測 比較評価レポート\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"評価日時: {datetime.now()}\n")
            f.write(f"モデル: {self.model_path}\n")
            f.write(f"テストデータ: {self.test_data_path}\n\n")

            # メトリクス
            f.write("📊 評価結果:\n")
            for key, value in metrics.items():
                if key != "developer_stats" and isinstance(value, (int, float)):
                    f.write(f"   {key}: {value:.3f}\n")

            f.write("\n🏆 主要指標:\n")
            if "exact_accuracy" in metrics:
                f.write(f"   正確一致率: {metrics['exact_accuracy']:.3f}\n")
            if "top_1_accuracy" in metrics:
                f.write(f"   Top-1精度: {metrics['top_1_accuracy']:.3f}\n")
            if "top_5_accuracy" in metrics:
                f.write(f"   Top-5精度: {metrics['top_5_accuracy']:.3f}\n")

        print(f"✅ レポート保存: {report_path}")
        return report_path

    def _create_visualizations(self, metrics, output_dir, timestamp):
        """可視化の作成"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 精度比較
            if any(key.startswith("top_") for key in metrics.keys()):
                fig, ax = plt.subplots(figsize=(10, 6))

                accuracy_metrics = {
                    k: v
                    for k, v in metrics.items()
                    if k.endswith("_accuracy") and isinstance(v, (int, float))
                }

                names = list(accuracy_metrics.keys())
                values = list(accuracy_metrics.values())

                bars = ax.bar(range(len(names)), values, color="steelblue", alpha=0.7)
                ax.set_xlabel("評価指標")
                ax.set_ylabel("精度")
                ax.set_title("RLエージェント vs 実際の開発者行動 - 精度比較")
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha="right")
                ax.set_ylim(0, 1)

                # 値をバーの上に表示
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()
                accuracy_path = output_dir / f"accuracy_comparison_{timestamp}.png"
                plt.savefig(accuracy_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"✅ 精度比較グラフ: {accuracy_path}")

        except ImportError:
            print(
                "⚠️ matplotlib/seabornがインストールされていません。可視化をスキップします。"
            )
        except Exception as e:
            print(f"⚠️ 可視化エラー: {e}")

    def run_evaluation(self, output_dir="outputs"):
        """評価実行"""
        print("🚀 実際の開発者行動との比較評価開始")
        print("=" * 60)

        # 実際の割り当て抽出
        actual_assignments = self.extract_actual_assignments()

        if not actual_assignments:
            print("❌ 実際の割り当てデータが見つかりません")
            return None

        # RLエージェント予測
        predictions, prediction_scores = self.predict_assignments(actual_assignments)

        # 評価指標計算
        metrics = self.calculate_metrics(
            actual_assignments, predictions, prediction_scores
        )

        # レポート生成
        report_path = self.generate_report(metrics, output_dir)

        print("\n✅ 評価完了!")
        print(f"📄 詳細レポート: {report_path}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="実際の開発者行動との比較評価")
    parser.add_argument(
        "--config", default="configs/unified_rl.yaml", help="設定ファイルパス"
    )
    parser.add_argument(
        "--model",
        default="models/simple_unified_rl_agent.zip",
        help="学習済みモデルパス",
    )
    parser.add_argument(
        "--test-data", default="data/backlog.json", help="テストデータパス（統合済み）"
    )
    parser.add_argument("--output", default="outputs", help="出力ディレクトリ")

    args = parser.parse_args()

    # パスの存在確認
    for path_name, path_value in [
        ("config", args.config),
        ("model", args.model),
        ("test-data", args.test_data),
    ]:
        if not Path(path_value).exists():
            print(f"❌ {path_name}ファイルが見つかりません: {path_value}")
            return 1

    # 評価実行
    evaluator = BehavioralComparison(args.config, args.model, args.test_data)
    metrics = evaluator.run_evaluation(args.output)

    if metrics:
        print("\n🎯 主要結果:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key.endswith("_accuracy"):
                print(f"   {key}: {value:.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
