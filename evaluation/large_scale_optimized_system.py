#!/usr/bin/env python3
"""
大規模評価対応の最適化アンサンブル推薦システム
全データを活用して統計的に信頼性の高い結果を得る
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from advanced_ensemble_system import AdvancedEnsembleSystem, PPOPolicyNetwork, is_bot


class LargeScaleOptimizedSystem(AdvancedEnsembleSystem):
    """大規模評価対応の最適化アンサンブル推薦システム"""

    def __init__(self, model_dir: str, test_data_path: str):
        super().__init__(model_dir, test_data_path)
        print("🔧 大規模最適化システム初期化完了")
        
        # 利用可能な全評価データを準備
        self._prepare_large_evaluation_data()

    def _prepare_large_evaluation_data(self):
        """大規模評価用データの準備"""
        print("📊 大規模評価データ準備中...")
        
        available_agents = set(self.models.keys())
        self.large_eval_tasks = []
        self.large_eval_ground_truth = []
        
        # 全データから利用可能なタスクを抽出
        for task, author in zip(self.tasks, self.ground_truth):
            if author in available_agents:
                self.large_eval_tasks.append(task)
                self.large_eval_ground_truth.append(author)
        
        print(f"   大規模評価データ: {len(self.large_eval_tasks):,}タスク")
        print(f"   対象開発者数: {len(available_agents)}人")
        
        # データをシャッフル（再現性のためseed固定）
        combined = list(zip(self.large_eval_tasks, self.large_eval_ground_truth))
        random.seed(42)
        random.shuffle(combined)
        self.large_eval_tasks, self.large_eval_ground_truth = zip(*combined)

    def feature_optimized_recommendation(
        self, task_features: torch.Tensor, task: Dict, k: int = 5
    ) -> List[Tuple[str, float]]:
        """🎯 Feature_Optimized推薦 - 実験で最高性能を示した手法"""

        methods_results = {}

        # 1. 基本アンサンブル
        basic_scores = {}
        for agent_name, model in self.models.items():
            try:
                ppo_score = model.get_action_score(task_features)
                contribution = self.author_contributions.get(agent_name, 0)
                contribution_score = min(contribution / 100.0, 1.0)
                similarity_score = self._calculate_task_similarity(task, agent_name)

                basic_score = (
                    0.4 * ppo_score + 0.4 * contribution_score + 0.2 * similarity_score
                )
                basic_scores[agent_name] = basic_score
            except:
                basic_scores[agent_name] = 0.0

        methods_results["basic"] = basic_scores

        # 2. 高度な類似度スコア（強化版）
        enhanced_similarity_scores = {}
        for agent_name in self.models.keys():
            base_similarity = self._calculate_task_similarity(task, agent_name)
            
            # より高度な類似度計算
            title_lower = (task.get("title", "") or "").lower()
            body_lower = (task.get("body", "") or "").lower()
            current_text = f"{title_lower} {body_lower}"
            
            # 開発者の過去タスクとの詳細類似度
            author_tasks = self.author_task_history.get(agent_name, [])
            if len(author_tasks) > 0:
                # 複数の類似度指標を統合
                text_similarities = []
                keyword_similarities = []
                
                for past_task in author_tasks[-20:]:  # 最新20件
                    past_title = (past_task.get("title", "") or "").lower()
                    past_body = (past_task.get("body", "") or "").lower()
                    past_text = f"{past_title} {past_body}"
                    
                    # 1. 単語重複度
                    current_words = set(current_text.split())
                    past_words = set(past_text.split())
                    if len(current_words) > 0 and len(past_words) > 0:
                        jaccard = len(current_words & past_words) / len(current_words | past_words)
                        text_similarities.append(jaccard)
                    
                    # 2. キーワード類似度
                    important_keywords = [
                        "bug", "fix", "error", "feature", "enhancement", "doc", 
                        "test", "ui", "api", "performance", "security"
                    ]
                    current_keywords = [kw for kw in important_keywords if kw in current_text]
                    past_keywords = [kw for kw in important_keywords if kw in past_text]
                    
                    if len(current_keywords) > 0 or len(past_keywords) > 0:
                        keyword_match = len(set(current_keywords) & set(past_keywords))
                        keyword_total = len(set(current_keywords) | set(past_keywords))
                        if keyword_total > 0:
                            keyword_similarities.append(keyword_match / keyword_total)
                
                # 最高類似度を採用（開発者の最も類似したタスク経験）
                max_text_sim = max(text_similarities) if text_similarities else 0.0
                max_keyword_sim = max(keyword_similarities) if keyword_similarities else 0.0
                
                # 基本類似度と統合
                enhanced_similarity = (
                    0.4 * base_similarity + 
                    0.4 * max_text_sim + 
                    0.2 * max_keyword_sim
                )
            else:
                enhanced_similarity = base_similarity
            
            enhanced_similarity_scores[agent_name] = enhanced_similarity

        methods_results["enhanced_similarity"] = enhanced_similarity_scores

        # 3. 貢献量スコア
        contribution_scores = {}
        for agent_name in self.models.keys():
            contribution = self.author_contributions.get(agent_name, 0)
            contribution_scores[agent_name] = min(contribution / 200.0, 1.0)

        methods_results["contribution"] = contribution_scores

        # 4. 時間的パターンスコア
        temporal_scores = {}
        for agent_name in self.models.keys():
            temporal_score = self._calculate_temporal_match(task, agent_name)
            temporal_scores[agent_name] = temporal_score

        methods_results["temporal"] = temporal_scores

        # Feature_Optimized統合 [0.25, 0.15, 0.45, 0.15]
        final_scores = {}

        for agent_name in self.models.keys():
            basic_score = methods_results["basic"].get(agent_name, 0.0)
            enhanced_sim_score = methods_results["enhanced_similarity"].get(agent_name, 0.0)
            contrib_score = methods_results["contribution"].get(agent_name, 0.0)
            temp_score = methods_results["temporal"].get(agent_name, 0.0)

            # Feature_Optimized重み（実験で最高性能）
            feature_optimized_score = (
                0.25 * basic_score +
                0.45 * enhanced_sim_score +  # 類似度最重視
                0.15 * contrib_score +
                0.15 * temp_score
            )

            # 適度なブースト（過度にならないよう調整）
            contribution = self.author_contributions.get(agent_name, 0)
            if contribution >= 200:
                feature_optimized_score *= 1.1
            elif contribution >= 100:
                feature_optimized_score *= 1.05

            if enhanced_sim_score > 0.7:
                feature_optimized_score *= 1.05

            final_scores[agent_name] = min(feature_optimized_score, 1.0)

        # スコア順にソート
        sorted_agents = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:k]

    def comprehensive_evaluation(
        self, 
        methods: List[str] = None, 
        sample_sizes: List[int] = None,
        cross_validation: bool = True
    ):
        """🎯 大規模総合評価"""
        print("🚀 大規模総合評価開始")
        print("=" * 60)

        if methods is None:
            methods = [
                "feature_optimized",
                "meta_ensemble", 
                "ultra_advanced"
            ]

        if sample_sizes is None:
            sample_sizes = [500, 1000, len(self.large_eval_tasks)]

        all_results = {}

        for method in methods:
            print(f"\n## 📊 {method}手法の評価")
            method_results = {}

            for sample_size in sample_sizes:
                if sample_size > len(self.large_eval_tasks):
                    sample_size = len(self.large_eval_tasks)

                print(f"\n### サンプルサイズ: {sample_size:,}")

                if cross_validation and sample_size < len(self.large_eval_tasks):
                    # クロスバリデーション（3分割）
                    cv_results = []
                    fold_size = sample_size // 3

                    for fold in range(3):
                        start_idx = fold * fold_size
                        end_idx = min((fold + 1) * fold_size, sample_size)
                        
                        fold_tasks = self.large_eval_tasks[start_idx:end_idx]
                        fold_authors = self.large_eval_ground_truth[start_idx:end_idx]
                        
                        fold_result = self._evaluate_method_on_data(
                            method, fold_tasks, fold_authors, f"Fold-{fold+1}"
                        )
                        cv_results.append(fold_result)

                    # 平均結果を計算
                    avg_results = {}
                    for k in [1, 3, 5]:
                        accuracies = [r[f"top_{k}"]["accuracy"] for r in cv_results]
                        diversities = [r[f"top_{k}"]["diversity_score"] for r in cv_results]
                        
                        avg_results[f"top_{k}"] = {
                            "accuracy": np.mean(accuracies),
                            "accuracy_std": np.std(accuracies),
                            "diversity_score": np.mean(diversities),
                            "diversity_std": np.std(diversities),
                        }

                    method_results[sample_size] = avg_results

                    # 結果表示
                    for k in [1, 3, 5]:
                        acc = avg_results[f"top_{k}"]["accuracy"]
                        acc_std = avg_results[f"top_{k}"]["accuracy_std"]
                        print(f"   Top-{k}精度: {acc:.3f} ± {acc_std:.3f} ({acc*100:.1f}% ± {acc_std*100:.1f}%)")

                else:
                    # 単一評価
                    eval_tasks = self.large_eval_tasks[:sample_size]
                    eval_authors = self.large_eval_ground_truth[:sample_size]
                    
                    single_result = self._evaluate_method_on_data(
                        method, eval_tasks, eval_authors, f"Single-{sample_size}"
                    )
                    method_results[sample_size] = single_result

                    # 結果表示
                    for k in [1, 3, 5]:
                        acc = single_result[f"top_{k}"]["accuracy"]
                        print(f"   Top-{k}精度: {acc:.3f} ({acc*100:.1f}%)")

            all_results[method] = method_results

        # 最終比較
        self._display_comprehensive_results(all_results, sample_sizes)
        return all_results

    def _evaluate_method_on_data(
        self, method: str, eval_tasks: List, eval_authors: List, desc: str
    ) -> Dict:
        """指定された手法でデータを評価"""
        results = {}

        for k in [1, 3, 5]:
            correct_predictions = 0
            all_recommendations = []

            for task, actual_author in tqdm(
                zip(eval_tasks, eval_authors),
                desc=f"{desc} Top-{k}",
                total=len(eval_tasks),
            ):
                try:
                    task_features = self._extract_task_features(task)

                    if method == "feature_optimized":
                        recommendations = self.feature_optimized_recommendation(
                            task_features, task, k
                        )
                    elif method == "meta_ensemble":
                        recommendations = self.meta_ensemble_recommendation(
                            task_features, task, k
                        )
                    elif method == "ultra_advanced":
                        recommendations = self.ultra_advanced_ensemble_recommendation(
                            task_features, task, k
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    recommended_agents = [agent for agent, _ in recommendations]
                    all_recommendations.extend(recommended_agents)

                    if actual_author in recommended_agents:
                        correct_predictions += 1

                except Exception:
                    continue

            accuracy = correct_predictions / len(eval_tasks) if eval_tasks else 0
            diversity_score = (
                len(set(all_recommendations)) / len(all_recommendations)
                if all_recommendations else 0
            )

            results[f"top_{k}"] = {
                "accuracy": accuracy,
                "diversity_score": diversity_score,
            }

        return results

    def _display_comprehensive_results(self, all_results: Dict, sample_sizes: List[int]):
        """総合結果の表示"""
        print(f"\n🏆 総合評価結果")
        print("=" * 80)

        # 最大サンプルサイズでの比較
        max_sample_size = max(sample_sizes)
        print(f"\n### 📊 最大サンプルサイズ({max_sample_size:,})での比較")
        
        print("| 手法 | Top-1精度 | Top-3精度 | Top-5精度 | 多様性 |")
        print("|------|-----------|-----------|-----------|--------|")

        best_top1_method = None
        best_top1_accuracy = 0

        for method, method_results in all_results.items():
            if max_sample_size in method_results:
                results = method_results[max_sample_size]
                top1_acc = results["top_1"]["accuracy"]
                top3_acc = results["top_3"]["accuracy"]
                top5_acc = results["top_5"]["accuracy"]
                diversity = results["top_1"]["diversity_score"]

                print(f"| {method} | {top1_acc*100:.1f}% | {top3_acc*100:.1f}% | {top5_acc*100:.1f}% | {diversity:.3f} |")

                if top1_acc > best_top1_accuracy:
                    best_top1_accuracy = top1_acc
                    best_top1_method = method

        print(f"\n🎯 最優秀手法: {best_top1_method}")
        print(f"🏆 最高Top-1精度: {best_top1_accuracy*100:.1f}%")

        if best_top1_accuracy >= 0.5:
            print(f"🎉 50%突破達成！")
        elif best_top1_accuracy >= 0.45:
            print(f"✅ 45%以上達成！")

        # サンプルサイズ別精度推移
        print(f"\n### 📈 サンプルサイズ別Top-1精度推移")
        print("| サンプルサイズ | " + " | ".join(all_results.keys()) + " |")
        print("|" + "---|" * (len(all_results) + 1))

        for sample_size in sample_sizes:
            row = f"| {sample_size:,} |"
            for method in all_results.keys():
                if sample_size in all_results[method]:
                    acc = all_results[method][sample_size]["top_1"]["accuracy"]
                    if "accuracy_std" in all_results[method][sample_size]["top_1"]:
                        std = all_results[method][sample_size]["top_1"]["accuracy_std"]
                        row += f" {acc*100:.1f}%±{std*100:.1f}% |"
                    else:
                        row += f" {acc*100:.1f}% |"
                else:
                    row += " - |"
            print(row)


def main():
    """メイン実行関数"""
    print("🚀 大規模最適化アンサンブル推薦システム")
    print("=" * 60)

    try:
        # システム初期化
        system = LargeScaleOptimizedSystem(
            model_dir="models/improved_rl/final_models",
            test_data_path="data/backlog_test_2023.json",
        )

        # 大規模総合評価
        results = system.comprehensive_evaluation(
            methods=["feature_optimized", "meta_ensemble"],
            sample_sizes=[500, 1000, len(system.large_eval_tasks)],
            cross_validation=True
        )

        print(f"\n🎉 大規模評価完了！")

    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()