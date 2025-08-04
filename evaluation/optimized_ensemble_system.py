#!/usr/bin/env python3
"""
最適化されたアンサンブル推薦システム
feature_optimizedパターンを基に45%→50%を目指す
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from advanced_ensemble_system import AdvancedEnsembleSystem, PPOPolicyNetwork, is_bot
from tqdm import tqdm


class OptimizedEnsembleSystem(AdvancedEnsembleSystem):
    """最適化されたアンサンブル推薦システム"""

    def __init__(self, model_dir: str, test_data_path: str):
        super().__init__(model_dir, test_data_path)
        print("🔧 最適化システム初期化完了")

    def ultra_optimized_recommendation(
        self, task_features: torch.Tensor, task: Dict, k: int = 5
    ) -> List[Tuple[str, float]]:
        """🚀 超最適化推薦 - 45%→50%を目指す"""

        methods_results = {}

        # 1. 基本アンサンブル（改良版）
        basic_scores = {}
        for agent_name, model in self.models.items():
            try:
                ppo_score = model.get_action_score(task_features)
                contribution = self.author_contributions.get(agent_name, 0)
                
                # 改良された貢献量スコア（より細かい段階）
                if contribution >= 500:
                    contribution_score = 1.0
                elif contribution >= 300:
                    contribution_score = 0.95
                elif contribution >= 200:
                    contribution_score = 0.9
                elif contribution >= 100:
                    contribution_score = 0.8
                elif contribution >= 50:
                    contribution_score = 0.65
                elif contribution >= 20:
                    contribution_score = 0.5
                elif contribution >= 10:
                    contribution_score = 0.35
                else:
                    contribution_score = 0.2
                
                similarity_score = self._calculate_task_similarity(task, agent_name)
                
                # 改良された基本スコア重み
                basic_score = (
                    0.35 * ppo_score + 
                    0.3 * contribution_score + 
                    0.35 * similarity_score
                )
                basic_scores[agent_name] = basic_score
            except:
                basic_scores[agent_name] = 0.0

        methods_results["basic"] = basic_scores

        # 2. 高度な類似度スコア
        enhanced_similarity_scores = {}
        for agent_name in self.models.keys():
            base_similarity = self._calculate_task_similarity(task, agent_name)
            
            # タスク特徴との詳細マッチング
            title_lower = (task.get("title", "") or "").lower()
            body_lower = (task.get("body", "") or "").lower()
            
            # 追加の類似度ブースト
            author_tasks = self.author_task_history.get(agent_name, [])
            if len(author_tasks) > 0:
                # 最近のタスクとの類似度（時間重み付き）
                recent_similarity = 0.0
                for i, past_task in enumerate(author_tasks[-10:]):  # 最新10件
                    past_title = (past_task.get("title", "") or "").lower()
                    past_body = (past_task.get("body", "") or "").lower()
                    
                    # 簡易文字列類似度
                    title_overlap = len(set(title_lower.split()) & set(past_title.split()))
                    body_overlap = len(set(body_lower.split()) & set(past_body.split()))
                    
                    task_similarity = (title_overlap + body_overlap) / (len(title_lower.split()) + len(body_lower.split()) + 1)
                    
                    # 時間重み（新しいほど重要）
                    time_weight = (i + 1) / 10
                    recent_similarity += task_similarity * time_weight
                
                recent_similarity /= min(len(author_tasks), 10)
                
                # 基本類似度と最近タスク類似度を統合
                enhanced_similarity = 0.7 * base_similarity + 0.3 * recent_similarity
            else:
                enhanced_similarity = base_similarity
            
            enhanced_similarity_scores[agent_name] = enhanced_similarity

        methods_results["enhanced_similarity"] = enhanced_similarity_scores

        # 3. 貢献量スコア（標準）
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

        # 5. 専門性スコア（改良版）
        specialization_scores = {}
        for agent_name in self.models.keys():
            author_tasks = self.author_task_history.get(agent_name, [])
            if len(author_tasks) > 0:
                # より詳細なタスクタイプ分類
                task_type_counts = defaultdict(int)
                
                for t in author_tasks:
                    title_lower = (t.get("title", "") or "").lower()
                    body_lower = (t.get("body", "") or "").lower()
                    full_text = f"{title_lower} {body_lower}"
                    
                    if any(kw in full_text for kw in ["bug", "fix", "error", "issue"]):
                        task_type_counts["bug"] += 1
                    elif any(kw in full_text for kw in ["feature", "enhancement", "new"]):
                        task_type_counts["feature"] += 1
                    elif any(kw in full_text for kw in ["doc", "readme", "guide", "documentation"]):
                        task_type_counts["doc"] += 1
                    elif any(kw in full_text for kw in ["test", "spec", "coverage"]):
                        task_type_counts["test"] += 1
                    elif any(kw in full_text for kw in ["ui", "ux", "design", "frontend"]):
                        task_type_counts["ui"] += 1
                    elif any(kw in full_text for kw in ["api", "backend", "server"]):
                        task_type_counts["api"] += 1
                    else:
                        task_type_counts["other"] += 1
                
                # 現在のタスクのタイプを判定
                current_title = (task.get("title", "") or "").lower()
                current_body = (task.get("body", "") or "").lower()
                current_full = f"{current_title} {current_body}"
                
                current_type = "other"
                if any(kw in current_full for kw in ["bug", "fix", "error", "issue"]):
                    current_type = "bug"
                elif any(kw in current_full for kw in ["feature", "enhancement", "new"]):
                    current_type = "feature"
                elif any(kw in current_full for kw in ["doc", "readme", "guide", "documentation"]):
                    current_type = "doc"
                elif any(kw in current_full for kw in ["test", "spec", "coverage"]):
                    current_type = "test"
                elif any(kw in current_full for kw in ["ui", "ux", "design", "frontend"]):
                    current_type = "ui"
                elif any(kw in current_full for kw in ["api", "backend", "server"]):
                    current_type = "api"
                
                # 該当タイプでの経験値
                type_experience = task_type_counts[current_type] / len(author_tasks)
                specialization_scores[agent_name] = min(type_experience * 2, 1.0)  # 2倍にして上限1.0
            else:
                specialization_scores[agent_name] = 0.0

        methods_results["specialization"] = specialization_scores

        # 超最適化統合
        final_scores = {}

        for agent_name in self.models.keys():
            basic_score = methods_results["basic"].get(agent_name, 0.0)
            enhanced_sim_score = methods_results["enhanced_similarity"].get(agent_name, 0.0)
            contrib_score = methods_results["contribution"].get(agent_name, 0.0)
            temp_score = methods_results["temporal"].get(agent_name, 0.0)
            spec_score = methods_results["specialization"].get(agent_name, 0.0)

            # タスクタイプ別の最適重み（実験結果に基づく）
            title_lower = (task.get("title", "") or "").lower()
            body_lower = (task.get("body", "") or "").lower()
            full_text = f"{title_lower} {body_lower}"

            if any(kw in full_text for kw in ["bug", "fix", "error", "issue"]):
                # バグ修正：経験と専門性重視
                weights = [0.2, 0.3, 0.25, 0.1, 0.15]
            elif any(kw in full_text for kw in ["feature", "enhancement", "new"]):
                # 新機能：類似度最重視（実験結果より）
                weights = [0.2, 0.5, 0.1, 0.05, 0.15]
            elif any(kw in full_text for kw in ["doc", "readme", "guide"]):
                # ドキュメント：類似度超重視
                weights = [0.15, 0.55, 0.1, 0.05, 0.15]
            elif any(kw in full_text for kw in ["test", "spec", "coverage"]):
                # テスト：専門性重視
                weights = [0.25, 0.35, 0.15, 0.05, 0.2]
            else:
                # 一般：バランス型（feature_optimizedベース）
                weights = [0.25, 0.45, 0.15, 0.05, 0.1]

            # 超最適化スコア計算
            ultra_score = (
                weights[0] * basic_score +
                weights[1] * enhanced_sim_score +
                weights[2] * contrib_score +
                weights[3] * temp_score +
                weights[4] * spec_score
            )

            # 強化ブースト（段階的）
            contribution = self.author_contributions.get(agent_name, 0)
            if contribution >= 500:
                ultra_score *= 1.25
            elif contribution >= 300:
                ultra_score *= 1.2
            elif contribution >= 200:
                ultra_score *= 1.15
            elif contribution >= 100:
                ultra_score *= 1.1
            elif contribution >= 50:
                ultra_score *= 1.05

            # 類似度ブースト（段階的）
            if enhanced_sim_score > 0.9:
                ultra_score *= 1.15
            elif enhanced_sim_score > 0.8:
                ultra_score *= 1.1
            elif enhanced_sim_score > 0.6:
                ultra_score *= 1.05

            # 専門性ブースト
            if spec_score > 0.8:
                ultra_score *= 1.1
            elif spec_score > 0.6:
                ultra_score *= 1.05

            final_scores[agent_name] = min(ultra_score, 1.0)

        # スコア順にソート
        sorted_agents = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_agents[:k]

    def evaluate_ultra_optimized(self, sample_size: int = 200):
        """超最適化システムの評価"""
        print("🚀 超最適化推薦システムの評価")
        print("=" * 50)

        available_agents = set(self.models.keys())
        eval_tasks = []
        eval_ground_truth = []

        for task, author in zip(
            self.tasks[:sample_size * 3], self.ground_truth[:sample_size * 3]
        ):
            if author in available_agents and len(eval_tasks) < sample_size:
                eval_tasks.append(task)
                eval_ground_truth.append(author)

        print(f"   評価タスク数: {len(eval_tasks)}")

        results = {}

        for k in [1, 3, 5]:
            correct_predictions = 0
            all_recommendations = []

            for task, actual_author in tqdm(
                zip(eval_tasks, eval_ground_truth),
                desc=f"Top-{k}評価中",
                total=len(eval_tasks),
            ):
                try:
                    task_features = self._extract_task_features(task)
                    recommendations = self.ultra_optimized_recommendation(
                        task_features, task, k
                    )

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

            print(f"   Top-{k}精度: {accuracy:.3f} ({accuracy*100:.1f}%)")
            print(f"   多様性スコア: {diversity_score:.3f}")

        return results


def main():
    """メイン実行関数"""
    print("🚀 超最適化アンサンブル推薦システム")
    print("=" * 60)

    try:
        # システム初期化
        system = OptimizedEnsembleSystem(
            model_dir="models/improved_rl/final_models",
            test_data_path="data/backlog_test_2023.json",
        )

        # 超最適化手法の評価
        print(f"\n## 超最適化手法の評価")
        ultra_results = system.evaluate_ultra_optimized(sample_size=200)

        # 従来手法との比較
        print(f"\n## 従来手法との比較")
        meta_results = system.evaluate_system("meta_ensemble", sample_size=200)

        print(f"\n🎉 比較結果")
        print("=" * 60)
        
        ultra_top1 = ultra_results["top_1"]["accuracy"]
        meta_top1 = meta_results["top_1"]["accuracy"]
        
        print(f"🏆 超最適化手法:")
        print(f"   Top-1精度: {ultra_top1*100:.1f}%")
        print(f"   Top-3精度: {ultra_results['top_3']['accuracy']*100:.1f}%")
        print(f"   Top-5精度: {ultra_results['top_5']['accuracy']*100:.1f}%")
        
        print(f"📊 従来メタアンサンブル:")
        print(f"   Top-1精度: {meta_top1*100:.1f}%")
        print(f"   Top-3精度: {meta_results['top_3']['accuracy']*100:.1f}%")
        print(f"   Top-5精度: {meta_results['top_5']['accuracy']*100:.1f}%")

        if ultra_top1 > meta_top1:
            improvement = (ultra_top1 - meta_top1) / meta_top1 * 100
            print(f"\n🚀 改善達成: +{improvement:.1f}%")
            print(f"🎯 最終Top-1精度: {ultra_top1*100:.1f}%")
            
            if ultra_top1 >= 0.5:
                print(f"🎉 50%突破達成！")
            elif ultra_top1 >= 0.47:
                print(f"✅ 47%以上達成！")
        else:
            print(f"📊 従来手法が優秀")

    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()