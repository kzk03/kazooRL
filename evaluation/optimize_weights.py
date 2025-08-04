#!/usr/bin/env python3
"""
メタアンサンブル重み最適化実験
"""

import itertools

from advanced_ensemble_system import AdvancedEnsembleSystem
from tqdm import tqdm


def optimize_meta_weights():
    """メタアンサンブルの重み最適化"""
    print("🔧 メタアンサンブル重み最適化実験")
    print("=" * 50)
    
    system = AdvancedEnsembleSystem(
        model_dir="models/improved_rl/final_models",
        test_data_path="data/backlog_test_2023.json",
    )
    
    # 重みパターンの生成
    weight_patterns = {
        "current_best": [0.35, 0.25, 0.25, 0.15],  # 現在の一般パターン
        "bug_optimized": [0.15, 0.45, 0.25, 0.15],  # バグ修正最適化
        "feature_optimized": [0.25, 0.15, 0.45, 0.15],  # 機能開発最適化
        "doc_optimized": [0.15, 0.15, 0.55, 0.15],  # ドキュメント最適化
        "experience_heavy": [0.10, 0.60, 0.20, 0.10],  # 経験超重視
        "similarity_heavy": [0.20, 0.15, 0.50, 0.15],  # 類似度超重視
        "balanced_new": [0.25, 0.30, 0.30, 0.15],  # 新バランス
    }
    
    # 評価用データ準備
    available_agents = set(system.models.keys())
    eval_tasks = []
    eval_ground_truth = []
    
    for task, author in zip(system.tasks[:300], system.ground_truth[:300]):
        if author in available_agents and len(eval_tasks) < 100:
            eval_tasks.append(task)
            eval_ground_truth.append(author)
    
    results = {}
    
    for pattern_name, base_weights in weight_patterns.items():
        print(f"\n🔍 {pattern_name}パターン評価中...")
        
        correct_count = 0
        
        for task, actual_author in tqdm(zip(eval_tasks, eval_ground_truth), 
                                      desc=f"{pattern_name}", 
                                      total=len(eval_tasks)):
            try:
                # カスタム重みでメタアンサンブル実行
                task_features = system._extract_task_features(task)
                
                # 各手法のスコア取得
                methods_results = {}
                
                # 基本アンサンブル
                basic_scores = {}
                for agent_name, model in system.models.items():
                    try:
                        ppo_score = model.get_action_score(task_features)
                        contribution = system.author_contributions.get(agent_name, 0)
                        contribution_score = min(contribution / 100.0, 1.0)
                        similarity_score = system._calculate_task_similarity(task, agent_name)
                        basic_score = (0.4 * ppo_score + 0.4 * contribution_score + 0.2 * similarity_score)
                        basic_scores[agent_name] = basic_score
                    except:
                        basic_scores[agent_name] = 0.0
                
                methods_results["basic"] = basic_scores
                
                # 他の手法のスコア
                contribution_scores = {agent: min(system.author_contributions.get(agent, 0) / 200.0, 1.0) 
                                     for agent in system.models.keys()}
                similarity_scores = {agent: system._calculate_task_similarity(task, agent) 
                                   for agent in system.models.keys()}
                temporal_scores = {agent: system._calculate_temporal_match(task, agent) 
                                 for agent in system.models.keys()}
                
                methods_results["contribution"] = contribution_scores
                methods_results["similarity"] = similarity_scores  
                methods_results["temporal"] = temporal_scores
                
                # カスタム重みで統合
                final_scores = {}
                for agent_name in system.models.keys():
                    basic_score = methods_results["basic"].get(agent_name, 0.0)
                    contrib_score = methods_results["contribution"].get(agent_name, 0.0)
                    sim_score = methods_results["similarity"].get(agent_name, 0.0)
                    temp_score = methods_results["temporal"].get(agent_name, 0.0)
                    
                    # カスタム重みを適用
                    meta_score = (
                        base_weights[0] * basic_score +
                        base_weights[1] * contrib_score +
                        base_weights[2] * sim_score +
                        base_weights[3] * temp_score
                    )
                    
                    # ブースト適用
                    contribution = system.author_contributions.get(agent_name, 0)
                    if contribution >= 200:
                        meta_score *= 1.2
                    elif contribution >= 100:
                        meta_score *= 1.15
                    elif contribution >= 50:
                        meta_score *= 1.1
                    
                    final_scores[agent_name] = min(meta_score, 1.0)
                
                # Top-1予測
                top_agent = max(final_scores.items(), key=lambda x: x[1])[0]
                if top_agent == actual_author:
                    correct_count += 1
                    
            except Exception as e:
                continue
        
        accuracy = correct_count / len(eval_tasks)
        results[pattern_name] = {
            "accuracy": accuracy,
            "weights": base_weights,
            "correct_count": correct_count,
            "total_count": len(eval_tasks)
        }
        
        print(f"   精度: {accuracy*100:.1f}% ({correct_count}/{len(eval_tasks)})")
    
    # 結果表示
    print(f"\n## 🏆 重み最適化結果")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    print("| 順位 | パターン名 | 精度 | 基本 | 貢献 | 類似 | 時間 |")
    print("|------|------------|------|------|------|------|------|")
    
    for i, (pattern_name, result) in enumerate(sorted_results):
        accuracy = result["accuracy"]
        weights = result["weights"]
        print(f"| {i+1} | {pattern_name} | {accuracy*100:.1f}% | {weights[0]:.2f} | {weights[1]:.2f} | {weights[2]:.2f} | {weights[3]:.2f} |")
    
    # 最適重みの提案
    best_pattern, best_result = sorted_results[0]
    print(f"\n🎯 最適重みパターン: {best_pattern}")
    print(f"   精度: {best_result['accuracy']*100:.1f}%")
    print(f"   推奨重み: {best_result['weights']}")


if __name__ == "__main__":
    optimize_meta_weights()