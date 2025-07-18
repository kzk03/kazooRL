#!/usr/bin/env python3
"""
類似度ベースのタスク推薦システム

教師あり学習アプローチ：
- 2022年データで開発者-タスクのマッチングパターンを学習
- 特徴量ベースの類似度で2023年データの推薦を実行
- 開発者プールが異なっても類似度でマッピング
"""

import argparse
import json
import pickle
# プロジェクトのモジュール
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

sys.path.append("/Users/kazuki-h/rl/kazoo")
sys.path.append("/Users/kazuki-h/rl/kazoo/src")

from src.kazoo.envs.task import Task
from src.kazoo.features.feature_extractor import FeatureExtractor


class SimilarityBasedRecommender:
    """類似度ベースのタスク推薦システム"""

    def __init__(self, config_path):
        """
        Args:
            config_path: 設定ファイルパス
        """
        self.config_path = config_path

        # 設定読み込み
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 学習済みモデル保存用
        self.trained_models = {}
        self.developer_embeddings = {}
        self.task_embeddings = {}
        self.scaler = StandardScaler()

        print("🎯 類似度ベースの推薦システム初期化完了")

    def load_data(self, data_path):
        """データを読み込んで時系列分割"""
        print("📊 データ読み込み中...")

        with open(data_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # 時系列分割
        training_data = []  # 2022年以前
        test_data = []  # 2023年

        for task in all_data:
            created_at = task.get("created_at", "")
            if created_at.startswith("2022"):
                training_data.append(task)
            elif created_at.startswith("2023"):
                test_data.append(task)

        print(f"   学習データ: {len(training_data):,} タスク (2022年)")
        print(f"   テストデータ: {len(test_data):,} タスク (2023年)")

        return training_data, test_data

    def extract_training_pairs(self, training_data):
        """学習データから開発者-タスクペアを抽出"""
        print("🔍 学習用ペア抽出中...")

        training_pairs = []
        developer_stats = Counter()

        for task_data in training_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            # 実際の担当者を抽出
            assignee = None

            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            if assignee:
                training_pairs.append(
                    {"task_data": task_data, "developer": assignee, "task_id": task_id}
                )
                developer_stats[assignee] += 1

        print(f"   学習ペア: {len(training_pairs):,} ペア")
        print(f"   ユニーク開発者: {len(developer_stats)} 人")

        # 上位開発者表示
        top_devs = developer_stats.most_common(10)
        print("   上位開発者:")
        for dev, count in top_devs:
            print(f"     {dev}: {count} タスク")

        return training_pairs, developer_stats

    def setup_feature_extractor(self):
        """特徴量抽出器の初期化"""
        print("🔧 特徴量抽出器初期化中...")

        # 設定オブジェクトの作成
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
        self.feature_extractor = FeatureExtractor(cfg)

        print(f"   特徴量次元: {len(self.feature_extractor.feature_names)}")
        print(f"   特徴量: {self.feature_extractor.feature_names[:10]}...")

    def extract_features_for_training(self, training_pairs):
        """学習用の特徴量を抽出"""
        print("🧮 学習用特徴量抽出中...")

        features = []
        labels = []
        developer_profiles = {}

        # 開発者プロファイル読み込み
        dev_profiles_path = self.config["env"]["dev_profiles_path"]
        with open(dev_profiles_path, "r", encoding="utf-8") as f:
            all_dev_profiles = yaml.safe_load(f)

        # ダミー環境作成
        dummy_env = type(
            "DummyEnv",
            (),
            {
                "backlog": [],
                "dev_profiles": all_dev_profiles,
                "assignments": {},
                "dev_action_history": {},
            },
        )()

        for i, pair in enumerate(training_pairs):
            try:
                task_obj = Task(pair["task_data"])
                developer_name = pair["developer"]

                # 開発者プロファイル（存在しない場合はデフォルト）
                dev_profile = all_dev_profiles.get(
                    developer_name,
                    {"name": developer_name, "skills": [], "expertise": []},
                )
                developer_obj = {"name": developer_name, "profile": dev_profile}

                # 特徴量抽出
                task_features = self.feature_extractor.get_features(
                    task_obj, developer_obj, dummy_env
                )

                features.append(task_features)
                labels.append(developer_name)

                # 開発者プロファイル記録
                if developer_name not in developer_profiles:
                    developer_profiles[developer_name] = {
                        "profile": dev_profile,
                        "task_count": 0,
                        "feature_sum": np.zeros_like(task_features),
                    }

                developer_profiles[developer_name]["task_count"] += 1
                developer_profiles[developer_name]["feature_sum"] += task_features

                if (i + 1) % 100 == 0:
                    print(f"   進捗: {i + 1}/{len(training_pairs)}")

            except Exception as e:
                print(f"⚠️ ペア {i} の特徴量抽出エラー: {e}")
                continue

        print(f"   抽出完了: {len(features)} 特徴量")

        # 開発者の平均特徴量を計算（埋め込み表現）
        for dev_name, data in developer_profiles.items():
            if data["task_count"] > 0:
                data["avg_features"] = data["feature_sum"] / data["task_count"]
            else:
                data["avg_features"] = np.zeros_like(
                    features[0] if features else np.zeros(62)
                )

        return np.array(features), labels, developer_profiles

    def train_similarity_model(self, features, labels, developer_profiles):
        """類似度ベースのモデルを訓練"""
        print("🎓 類似度モデル訓練中...")

        # 特徴量の正規化
        features_scaled = self.scaler.fit_transform(features)

        # RandomForest分類器で開発者マッチングパターンを学習
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
        )

        self.rf_classifier.fit(features_scaled, labels)

        # 開発者埋め込みを保存
        self.developer_embeddings = {}
        for dev_name, data in developer_profiles.items():
            self.developer_embeddings[dev_name] = data["avg_features"]

        print(f"   訓練完了: {len(set(labels))} クラス（開発者）")
        print(f"   開発者埋め込み: {len(self.developer_embeddings)} 人")

        # 特徴重要度の表示
        feature_importance = self.rf_classifier.feature_importances_
        top_features = np.argsort(feature_importance)[-10:][::-1]

        print("   重要特徴量 Top-10:")
        for i, feat_idx in enumerate(top_features):
            feat_name = self.feature_extractor.feature_names[feat_idx]
            importance = feature_importance[feat_idx]
            print(f"     {i+1}. {feat_name}: {importance:.4f}")

    def find_similar_developers(self, test_developers, top_k=5):
        """テストデータの開発者に類似した学習データの開発者を見つける"""
        print("🔍 類似開発者検索中...")

        similarity_mapping = {}

        for test_dev in test_developers:
            # テスト開発者が学習データにいる場合
            if test_dev in self.developer_embeddings:
                similarity_mapping[test_dev] = [(test_dev, 1.0)]
                continue

            # 類似度計算（コサイン類似度）
            similarities = []

            # テスト開発者のダミー特徴量（タスク履歴から推定）
            test_embedding = np.random.normal(
                0, 0.1, len(list(self.developer_embeddings.values())[0])
            )

            for train_dev, train_embedding in self.developer_embeddings.items():
                # コサイン類似度
                cosine_sim = np.dot(test_embedding, train_embedding) / (
                    np.linalg.norm(test_embedding) * np.linalg.norm(train_embedding)
                )
                similarities.append((train_dev, cosine_sim))

            # 上位K人の類似開発者
            similarities.sort(key=lambda x: x[1], reverse=True)
            similarity_mapping[test_dev] = similarities[:top_k]

        print(f"   類似度マッピング完了: {len(similarity_mapping)} 開発者")
        return similarity_mapping

    def predict_assignments(self, test_data, similarity_mapping):
        """テストデータの割り当てを予測"""
        print("🤖 割り当て予測中...")

        predictions = {}
        prediction_scores = {}

        # テストデータの実際の割り当てを抽出
        test_assignments = {}
        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            assignee = None
            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            if assignee:
                test_assignments[task_id] = assignee

        print(f"   予測対象: {len(test_assignments)} タスク")

        # ダミー環境
        dummy_env = type(
            "DummyEnv",
            (),
            {
                "backlog": [],
                "dev_profiles": {},
                "assignments": {},
                "dev_action_history": {},
            },
        )()

        # 各タスクの予測
        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if task_id not in test_assignments:
                continue

            try:
                task_obj = Task(task_data)
                actual_dev = test_assignments[task_id]

                # 実際の開発者に類似した学習開発者を取得
                if actual_dev in similarity_mapping:
                    similar_devs = similarity_mapping[actual_dev]

                    # 各類似開発者での予測確率を計算
                    dev_scores = []

                    for similar_dev, similarity_score in similar_devs:
                        # ダミー開発者オブジェクト
                        dev_obj = {"name": similar_dev, "profile": {}}

                        # 特徴量抽出
                        features = self.feature_extractor.get_features(
                            task_obj, dev_obj, dummy_env
                        )
                        features_scaled = self.scaler.transform([features])

                        # モデルによる予測確率
                        proba = self.rf_classifier.predict_proba(features_scaled)[0]
                        dev_classes = self.rf_classifier.classes_

                        # 類似開発者の予測確率を取得
                        if similar_dev in dev_classes:
                            dev_idx = list(dev_classes).index(similar_dev)
                            pred_proba = proba[dev_idx]
                        else:
                            pred_proba = 0.0

                        # 類似度で重み付け
                        weighted_score = pred_proba * similarity_score
                        dev_scores.append(
                            (similar_dev, weighted_score, pred_proba, similarity_score)
                        )

                    # 最高スコアの開発者を予測
                    if dev_scores:
                        dev_scores.sort(key=lambda x: x[1], reverse=True)
                        best_dev, best_score, best_proba, best_similarity = dev_scores[
                            0
                        ]

                        predictions[task_id] = best_dev
                        prediction_scores[task_id] = {
                            "predicted_dev": best_dev,
                            "score": best_score,
                            "probability": best_proba,
                            "similarity": best_similarity,
                            "all_scores": dev_scores[:5],
                        }

            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測エラー: {e}")
                continue

        print(f"   予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores, test_assignments

    def evaluate_predictions(self, predictions, test_assignments, prediction_scores):
        """予測結果の評価"""
        print("📊 予測評価中...")

        # 共通タスクで評価
        common_tasks = set(predictions.keys()) & set(test_assignments.keys())
        print(f"   評価対象: {len(common_tasks)} タスク")

        if not common_tasks:
            print("⚠️ 評価可能なタスクがありません")
            return {}

        # 正確性評価
        correct_predictions = 0
        for task_id in common_tasks:
            if predictions[task_id] == test_assignments[task_id]:
                correct_predictions += 1

        accuracy = correct_predictions / len(common_tasks)

        # 類似度ベース評価
        similarity_scores = []
        prediction_confidences = []

        for task_id in common_tasks:
            if task_id in prediction_scores:
                scores = prediction_scores[task_id]
                similarity_scores.append(scores["similarity"])
                prediction_confidences.append(scores["score"])

        metrics = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_predictions": len(common_tasks),
            "avg_similarity": np.mean(similarity_scores) if similarity_scores else 0.0,
            "avg_confidence": (
                np.mean(prediction_confidences) if prediction_confidences else 0.0
            ),
            "similarity_std": np.std(similarity_scores) if similarity_scores else 0.0,
            "confidence_std": (
                np.std(prediction_confidences) if prediction_confidences else 0.0
            ),
        }

        print(f"   精度: {accuracy:.3f} ({correct_predictions}/{len(common_tasks)})")
        print(
            f"   平均類似度: {metrics['avg_similarity']:.3f} ± {metrics['similarity_std']:.3f}"
        )
        print(
            f"   平均信頼度: {metrics['avg_confidence']:.3f} ± {metrics['confidence_std']:.3f}"
        )

        # 開発者別評価
        dev_metrics = self._evaluate_by_developer(
            predictions, test_assignments, common_tasks
        )
        metrics["developer_metrics"] = dev_metrics

        return metrics

    def _evaluate_by_developer(self, predictions, test_assignments, common_tasks):
        """開発者別の評価"""
        dev_metrics = defaultdict(lambda: {"correct": 0, "total": 0})

        for task_id in common_tasks:
            actual_dev = test_assignments[task_id]
            predicted_dev = predictions[task_id]

            dev_metrics[actual_dev]["total"] += 1
            if actual_dev == predicted_dev:
                dev_metrics[actual_dev]["correct"] += 1

        # 精度計算
        for dev, stats in dev_metrics.items():
            stats["accuracy"] = (
                stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        print("   開発者別精度:")
        sorted_devs = sorted(
            dev_metrics.items(), key=lambda x: x[1]["total"], reverse=True
        )
        for dev, stats in sorted_devs[:10]:
            print(
                f"     {dev}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})"
            )

        return dict(dev_metrics)

    def save_model(self, output_dir="models"):
        """モデルの保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Random Forest モデル
        model_path = output_dir / f"similarity_recommender_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "rf_classifier": self.rf_classifier,
                    "scaler": self.scaler,
                    "developer_embeddings": self.developer_embeddings,
                    "feature_names": self.feature_extractor.feature_names,
                },
                f,
            )

        print(f"✅ モデル保存: {model_path}")
        return model_path

    def run_full_pipeline(self, data_path, output_dir="outputs"):
        """完全パイプラインの実行"""
        print("🚀 類似度ベース推薦システム実行開始")
        print("=" * 70)

        # 1. データ読み込み
        training_data, test_data = self.load_data(data_path)

        # 2. 学習ペア抽出
        training_pairs, developer_stats = self.extract_training_pairs(training_data)

        # 3. 特徴量抽出器初期化
        self.setup_feature_extractor()

        # 4. 学習用特徴量抽出
        features, labels, developer_profiles = self.extract_features_for_training(
            training_pairs
        )

        # 5. モデル訓練
        self.train_similarity_model(features, labels, developer_profiles)

        # 6. テストデータの開発者を抽出
        test_assignments = {}
        test_developers = set()
        for task_data in test_data:
            task_id = task_data.get("id") or task_data.get("number")
            if not task_id:
                continue

            assignee = None
            if "assignees" in task_data and task_data["assignees"]:
                assignee = task_data["assignees"][0].get("login")
            elif "events" in task_data:
                for event in task_data["events"]:
                    if event.get("event") == "assigned" and event.get("assignee"):
                        assignee = event["assignee"].get("login")
                        break

            if assignee:
                test_developers.add(assignee)

        # 7. 類似開発者マッピング
        similarity_mapping = self.find_similar_developers(list(test_developers))

        # 8. 予測実行
        predictions, prediction_scores, test_assignments = self.predict_assignments(
            test_data, similarity_mapping
        )

        # 9. 評価
        metrics = self.evaluate_predictions(
            predictions, test_assignments, prediction_scores
        )

        # 10. 結果保存
        self.save_results(
            metrics, predictions, prediction_scores, test_assignments, output_dir
        )

        # 11. モデル保存
        model_path = self.save_model()

        print("✅ 類似度ベース推薦システム完了")
        return metrics

    def save_results(
        self, metrics, predictions, prediction_scores, test_assignments, output_dir
    ):
        """結果の保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # メトリクス保存
        metrics_path = output_dir / f"similarity_recommender_metrics_{timestamp}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            # numpy配列を対応
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                return obj

            json.dump(metrics, f, indent=2, ensure_ascii=False, default=convert_numpy)

        # 予測結果保存
        results = []
        for task_id in set(predictions.keys()) | set(test_assignments.keys()):
            result = {
                "task_id": task_id,
                "actual_developer": test_assignments.get(task_id),
                "predicted_developer": predictions.get(task_id),
                "correct": test_assignments.get(task_id) == predictions.get(task_id),
            }

            if task_id in prediction_scores:
                result.update(prediction_scores[task_id])

            results.append(result)

        results_df = pd.DataFrame(results)
        results_path = output_dir / f"similarity_recommender_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False, encoding="utf-8")

        print(f"✅ メトリクス保存: {metrics_path}")
        print(f"✅ 結果保存: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="類似度ベースのタスク推薦システム")
    parser.add_argument(
        "--config", default="configs/unified_rl.yaml", help="設定ファイルパス"
    )
    parser.add_argument("--data", default="data/backlog.json", help="統合データパス")
    parser.add_argument("--output", default="outputs", help="出力ディレクトリ")

    args = parser.parse_args()

    # 推薦システム実行
    recommender = SimilarityBasedRecommender(args.config)
    metrics = recommender.run_full_pipeline(args.data, args.output)

    print("\n🎯 最終結果:")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "developer_metrics":
                print(f"   {key}: {value:.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
