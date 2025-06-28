#!/usr/bin/env python3
"""
強化学習で学習したモデルを使用した推薦システムの精度評価

このスクリプトは：
1. 学習済みPPOモデルを読み込み
2. テストデータに対して推薦を実行
3. 正解データ（エキスパートトラジェクトリ）と比較
4. 推薦精度（Top-K accuracy）を計算
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from kazoo.envs.oss_simple import OSSSimpleEnv
from kazoo.learners.independent_ppo_controller import IndependentPPOController


class RecommendationEvaluator:
    """推薦システムの精度評価クラス"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # データの読み込み
        import json
        import yaml
        
        with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
            backlog = json.load(f)
        with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            dev_profiles = yaml.safe_load(f)
        
        # 環境とコントローラーの初期化
        self.env = OSSSimpleEnv(cfg, backlog, dev_profiles)
        self.controller = IndependentPPOController(self.env, cfg)
        
        # 学習済みモデルの読み込み
        self.load_trained_models()
        
        # エキスパートデータの読み込み（正解データとして使用）
        self.expert_trajectories = self.load_expert_data()
    
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
        
        print(f"Successfully loaded {loaded_count}/{len(self.controller.agent_ids)} models")
    
    def load_expert_data(self) -> List[Dict]:
        """エキスパート軌跡データを読み込む（正解データとして使用）"""
        expert_path = Path(self.cfg.irl.expert_path)
        
        if not expert_path.exists():
            print(f"⚠️  Expert data not found: {expert_path}")
            return []
        
        with open(expert_path, 'rb') as f:
            expert_data = pickle.load(f)
        
        print(f"Loaded {len(expert_data)} expert trajectories")
        return expert_data
    
    def get_model_recommendations(self, num_tasks: int = 50) -> List[Tuple[str, str]]:
        """
        学習済みモデルを使用してタスク推薦を生成
        
        Returns:
            List of (task_id, recommended_developer) tuples
        """
        recommendations = []
        
        # 環境をリセット
        observations = self.env.reset()
        
        for step in range(num_tasks):
            if not self.env.backlog:
                break
            
            # 現在のタスクを取得
            current_task = self.env.backlog[0]  # 最初のタスクを評価対象とする
            
            # 各開発者に対する行動確率を計算
            developer_scores = {}
            
            for agent_id in self.controller.agent_ids:
                if agent_id in observations:
                    obs = observations[agent_id]
                    
                    # モデルから行動確率を取得
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).to(self.controller.agents[agent_id].device)
                        action_probs = self.controller.agents[agent_id].policy.actor(obs_tensor).cpu().numpy()
                    
                    # アクション2（タスクを受け入れる）の確率を使用
                    accept_prob = action_probs[2].item() if len(action_probs) > 2 else 0.0
                    developer_scores[agent_id] = accept_prob
            
            # 最も確率の高い開発者を推薦
            recommended_dev = None
            if developer_scores:
                recommended_dev = max(developer_scores, key=developer_scores.get)
                recommendations.append((current_task.id, recommended_dev))
            
            # 推薦を実行（シミュレーション）
            actions = {agent_id: 0 for agent_id in self.controller.agent_ids}  # 全員が待機
            if recommended_dev and recommended_dev in actions:
                actions[recommended_dev] = 2  # 推薦された開発者がタスクを受け入れ
            
            observations, rewards, terminateds, truncateds, infos = self.env.step(actions)
            
            # 終了条件チェック
            if all(terminateds.values()) or all(truncateds.values()):
                break
        
        return recommendations
    
    def calculate_accuracy(self, recommendations: List[Tuple[str, str]], k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        推薦精度を計算
        
        Args:
            recommendations: モデルの推薦結果 [(task_id, recommended_dev), ...]
            k_values: Top-K精度を計算するKの値のリスト
        
        Returns:
            各K値に対する精度の辞書
        """
        if not self.expert_trajectories:
            print("❌ No expert data available for accuracy calculation")
            return {}
        
        # エキスパートデータから正解を作成
        expert_assignments = {}
        for trajectory_episode in self.expert_trajectories:  # 外側のリスト（エピソード）
            for step in trajectory_episode:  # 各エピソード内のステップ
                if isinstance(step, dict) and 'action_details' in step:
                    action_details = step['action_details']
                    task_id = action_details.get('task_id')
                    assigned_dev = action_details.get('developer')
                    if task_id and assigned_dev:
                        expert_assignments[task_id] = assigned_dev
        
        print(f"Expert assignments available for {len(expert_assignments)} tasks")
        
        # 精度計算
        accuracies = {}
        valid_recommendations = 0
        
        for k in k_values:
            correct_predictions = 0
            
            for task_id, recommended_dev in recommendations:
                if task_id in expert_assignments:
                    valid_recommendations += 1
                    expert_dev = expert_assignments[task_id]
                    
                    # Top-K精度の場合、ここでは単純にTop-1として計算
                    # より複雑なTop-K計算が必要な場合は、複数の推薦を生成する必要がある
                    if recommended_dev == expert_dev:
                        correct_predictions += 1
            
            if valid_recommendations > 0:
                accuracy = correct_predictions / valid_recommendations
                accuracies[f"top_{k}_accuracy"] = accuracy
            else:
                accuracies[f"top_{k}_accuracy"] = 0.0
        
        return accuracies
    
    def evaluate(self, num_tasks: int = 50):
        """推薦システムの総合評価を実行"""
        print("🎯 Starting recommendation accuracy evaluation...")
        print(f"Evaluating {num_tasks} tasks")
        
        # 推薦の生成
        print("\n📊 Generating recommendations...")
        recommendations = self.get_model_recommendations(num_tasks)
        
        print(f"Generated {len(recommendations)} recommendations")
        
        # サンプル推薦の表示
        print("\n📋 Sample recommendations:")
        for i, (task_id, dev) in enumerate(recommendations[:5]):
            print(f"  {i+1}. Task {task_id} → {dev}")
        
        # 精度計算
        print("\n🎯 Calculating accuracy...")
        accuracies = self.calculate_accuracy(recommendations)
        
        # 結果表示
        print("\n📈 Recommendation Accuracy Results:")
        print("=" * 50)
        
        if accuracies:
            for metric, value in accuracies.items():
                print(f"{metric:20s}: {value:.3f} ({value*100:.1f}%)")
        else:
            print("❌ Could not calculate accuracy (no expert data or valid recommendations)")
        
        # 詳細統計
        print(f"\nTotal recommendations: {len(recommendations)}")
        print(f"Expert data points: {len(self.expert_trajectories)}")
        
        return accuracies


@hydra.main(version_base="1.3", config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """メイン実行関数"""
    
    print("🚀 Recommendation Accuracy Evaluation")
    print("=" * 50)
    
    try:
        evaluator = RecommendationEvaluator(cfg)
        results = evaluator.evaluate(num_tasks=30)  # 30タスクで評価
        
        print("\n✅ Evaluation completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
