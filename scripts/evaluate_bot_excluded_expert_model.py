#!/usr/bin/env python3
"""
Bot除外済みExpert軌跡で学習したモデルの評価スクリプト

このスクリプトは：
1. Bot除外済みのデータで学習したモデルを読み込み
2. 同じデータ分割条件で評価を実行
3. Bot除外済みexpert軌跡を使用した学習効果を検証

使用例:
    python scripts/evaluate_bot_excluded_expert_model.py
"""

import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    bot_indicators = ['bot', 'stale', 'dependabot', 'renovate', 'greenkeeper']
    return any(indicator in username_lower for indicator in bot_indicators)


def filter_bot_tasks_and_developers(backlog, dev_profiles):
    """Filter out bot tasks and bot developers from data"""
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
    
    # Filter tasks - remove bot-assigned tasks
    human_backlog = []
    bot_tasks_filtered = 0
    
    for task in backlog:
        assigned_to = task.get('assigned_to', '')
        
        # Check if assigned to bot
        if assigned_to and is_bot_user(assigned_to):
            bot_tasks_filtered += 1
            continue
        
        # Check if developer exists in human profiles
        if assigned_to and assigned_to not in human_dev_profiles:
            bot_tasks_filtered += 1
            continue
        
        human_backlog.append(task)
    
    print(f"Original tasks: {len(backlog)}")
    print(f"Bot tasks filtered out: {bot_tasks_filtered}")
    print(f"Human tasks remaining: {len(human_backlog)}")
    
    return human_backlog, human_dev_profiles


def split_tasks_by_time(tasks, train_ratio=0.7):
    """Split tasks into train/test by time"""
    tasks_with_dates = []
    
    for task in tasks:
        created_at = task.get('created_at') or task.get('createdAt')
        if created_at:
            try:
                if isinstance(created_at, str):
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
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


def load_bot_excluded_expert_data(expert_trajectories_path):
    """Load bot-excluded expert trajectories and extract assignments"""
    try:
        with open(expert_trajectories_path, "rb") as f:
            expert_trajectories = pickle.load(f)
        
        print(f"Loaded {len(expert_trajectories)} bot-excluded expert trajectory episodes")
        
        # Extract expert assignments
        expert_assignments = {}
        for episode in expert_trajectories:
            for step in episode:
                if isinstance(step, dict):
                    action_details = step.get('action_details', {})
                    if action_details:
                        task_id = action_details.get('task_id')
                        developer = action_details.get('developer')
                        
                        if task_id and developer:
                            expert_assignments[task_id] = developer
        
        print(f"Expert assignments found: {len(expert_assignments)}")
        return expert_trajectories, expert_assignments
        
    except Exception as e:
        print(f"Error loading bot-excluded expert trajectories: {e}")
        return [], {}


class BotExcludedExpertModelEvaluator:
    """Bot除外済みExpert軌跡で学習したモデルの評価クラス"""
    
    def __init__(self, cfg: DictConfig, train_ratio: float = 0.7):
        self.cfg = cfg
        self.train_ratio = train_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # データの読み込み
        print("Loading data...")
        with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
            self.full_backlog = json.load(f)
        
        with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
            self.full_dev_profiles = yaml.safe_load(f)
        
        print(f"Loaded {len(self.full_backlog)} tasks and {len(self.full_dev_profiles)} developer profiles")
        
        # Bot除外済みexpert軌跡の読み込み
        expert_trajectories_path = "data/expert_trajectories_bot_excluded.pkl"
        self.expert_trajectories, self.expert_assignments = load_bot_excluded_expert_data(expert_trajectories_path)
        
        # Bot除外処理
        self.human_backlog, self.human_dev_profiles = filter_bot_tasks_and_developers(
            self.full_backlog, self.full_dev_profiles
        )
        
        # 時系列分割
        self.train_tasks, self.test_tasks = split_tasks_by_time(self.human_backlog, train_ratio)
        
        # 訓練・テスト用エキスパート割り当ての分離
        train_task_ids = {task['id'] for task in self.train_tasks}
        test_task_ids = {task['id'] for task in self.test_tasks}
        
        self.train_expert_assignments = {
            task_id: dev for task_id, dev in self.expert_assignments.items()
            if task_id in train_task_ids
        }
        self.test_expert_assignments = {
            task_id: dev for task_id, dev in self.expert_assignments.items()
            if task_id in test_task_ids
        }
        
        print(f"\nData split summary:")
        print(f"  Train tasks: {len(self.train_tasks)}")
        print(f"  Test tasks: {len(self.test_tasks)}")
        print(f"  Human developers: {len(self.human_dev_profiles)}")
        print(f"  Train expert assignments: {len(self.train_expert_assignments)}")
        print(f"  Test expert assignments: {len(self.test_expert_assignments)}")
        
        # 環境の初期化（訓練時と同じ条件）
        print("\nInitializing environment (matching training conditions)...")
        self.env = OSSSimpleEnv(cfg, self.train_tasks, self.human_dev_profiles)
        
        # コントローラーの初期化
        self.controller = IndependentPPOController(self.env, cfg)
        self.models_loaded = False
    
    def load_models(self, model_dir: Path = Path("models")):
        """Bot除外済みexpert軌跡で学習したモデルを読み込み"""
        if not model_dir.exists():
            print(f"❌ Model directory not found: {model_dir}")
            return
        
        # メタデータの読み込み
        metadata_path = model_dir / "training_metadata_bot_excluded.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"Loading bot-excluded expert models:")
            print(f"  Training completed: {metadata.get('training_completed', False)}")
            print(f"  Human developers: {metadata.get('human_developers', [])[:5]}...")
            print(f"  Train tasks: {metadata.get('train_tasks_count', 0)}")
            print(f"  Expert episodes used: {metadata.get('human_expert_episodes', 0)}")
        else:
            print("Warning: No bot-excluded training metadata found")
        
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
        self.models_loaded = loaded_count > 0
    
    def get_task_observation(self, task: Dict) -> np.ndarray:
        """単一タスクの観測ベクトルを生成"""
        # 訓練時と同じ観測空間を使用するため、train_tasksを基に観測を生成
        task_states = []
        
        for train_task in self.train_tasks:
            if train_task["id"] == task["id"]:
                status_val = 1  # current task in progress
            else:
                status_val = 0  # other tasks available
            
            # Same format as training: [status, complexity, deadline]
            task_states.extend([status_val, 0, 0])
        
        # GNN embedding placeholder (64 dimensions)
        gnn_embedding = np.zeros(64, dtype=np.float32)
        
        # Combine task states and GNN embedding
        obs = np.concatenate([np.array(task_states, dtype=np.float32), gnn_embedding])
        
        return obs
    
    def generate_recommendations(self, max_k: int = 5) -> List[Tuple[str, List[str]]]:
        """テストタスクに対する推薦を生成"""
        print(f"Generating recommendations for {len(self.test_tasks)} test tasks")
        
        if not self.models_loaded:
            print("⚠️  No models loaded. Using random baseline.")
            import random
            recommendations = []
            developer_list = list(self.human_dev_profiles.keys())
            for task in self.test_tasks:
                random_devs = random.sample(developer_list, min(max_k, len(developer_list)))
                recommendations.append((task['id'], random_devs))
            return recommendations
        
        recommendations = []
        
        for task in self.test_tasks:
            # タスクの観測を生成
            task_obs = self.get_task_observation(task)
            
            # 各開発者の推薦スコアを計算
            developer_scores = {}
            
            for agent_id in self.controller.agent_ids:
                try:
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(task_obs).to(self.controller.agents[agent_id].device)
                        action_logits = self.controller.agents[agent_id].policy.actor(obs_tensor)
                        action_probs = torch.softmax(action_logits, dim=-1).cpu().numpy()
                    
                    # タスク選択の確率（NO_OP以外の最大確率）
                    if len(action_probs) > 1:
                        task_action_probs = action_probs[:-1]  # Exclude NO_OP
                        accept_prob = np.max(task_action_probs)
                    else:
                        accept_prob = 0.0
                    
                    developer_scores[agent_id] = accept_prob
                    
                except Exception as e:
                    print(f"⚠️  Model inference error for {agent_id}: {e}")
                    developer_scores[agent_id] = np.random.random()
            
            # Top-K推薦の生成
            if developer_scores:
                sorted_developers = sorted(developer_scores.items(), key=lambda x: x[1], reverse=True)
                top_k_developers = [dev for dev, score in sorted_developers[:max_k]]
                recommendations.append((task['id'], top_k_developers))
            else:
                recommendations.append((task['id'], []))
        
        print(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def calculate_accuracy(self, recommendations: List[Tuple[str, List[str]]], k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """推薦精度を計算"""
        print(f"Test expert assignments available: {len(self.test_expert_assignments)}")
        
        accuracies = {}
        
        for k in k_values:
            correct_predictions = 0
            valid_recommendations = 0
            
            for task_id, recommended_devs in recommendations:
                if task_id in self.test_expert_assignments:
                    valid_recommendations += 1
                    expert_dev = self.test_expert_assignments[task_id]
                    
                    top_k_recommendations = recommended_devs[:k]
                    if expert_dev in top_k_recommendations:
                        correct_predictions += 1
            
            if valid_recommendations > 0:
                accuracy = correct_predictions / valid_recommendations
                accuracies[f"top_{k}_accuracy"] = accuracy
            else:
                accuracies[f"top_{k}_accuracy"] = 0.0
        
        accuracies['total_valid_recommendations'] = valid_recommendations
        accuracies['total_recommendations'] = len(recommendations)
        
        return accuracies
    
    def analyze_baseline_performance(self):
        """ベースライン手法の分析"""
        print("\n🔍 Baseline Analysis:")
        
        num_developers = len(self.human_dev_profiles)
        
        # ランダムベースライン
        random_results = {}
        for k in [1, 3, 5]:
            random_results[f"random_top_{k}"] = min(k / num_developers, 1.0)
        
        print(f"Random baseline ({num_developers} human developers):")
        for k in [1, 3, 5]:
            acc = random_results[f"random_top_{k}"]
            print(f"  Top-{k}: {acc:.3f} ({acc*100:.1f}%)")
        
        # 最頻開発者ベースライン
        from collections import Counter
        if self.train_expert_assignments:
            dev_counts = Counter(self.train_expert_assignments.values())
            most_frequent_devs = [dev for dev, count in dev_counts.most_common(5)]
            
            print(f"\nMost frequent developers in training:")
            for i, (dev, count) in enumerate(dev_counts.most_common(5)):
                print(f"  {i+1}. {dev}: {count} assignments")
            
            # 最頻開発者ベースライン精度
            frequent_results = {}
            for k in [1, 3, 5]:
                correct = 0
                for task_id, expert_dev in self.test_expert_assignments.items():
                    if expert_dev in most_frequent_devs[:k]:
                        correct += 1
                
                if len(self.test_expert_assignments) > 0:
                    accuracy = correct / len(self.test_expert_assignments)
                    frequent_results[f"frequent_top_{k}"] = accuracy
                else:
                    frequent_results[f"frequent_top_{k}"] = 0.0
            
            print(f"\nMost frequent developers baseline:")
            for k in [1, 3, 5]:
                acc = frequent_results[f"frequent_top_{k}"]
                print(f"  Top-{k}: {acc:.3f} ({acc*100:.1f}%)")
        else:
            print("\nNo training expert assignments available for baseline analysis")
            frequent_results = {f"frequent_top_{k}": 0.0 for k in [1, 3, 5]}
        
        return {**random_results, **frequent_results}
    
    def evaluate(self):
        """推薦システムの評価を実行"""
        print("🚀 Bot-Excluded Expert Model Evaluation")
        print("=" * 60)
        
        # モデルの読み込み
        self.load_models()
        
        # ベースライン分析
        baseline_results = self.analyze_baseline_performance()
        
        # 推薦の生成
        print("\n📊 Generating recommendations...")
        recommendations = self.generate_recommendations(max_k=5)
        
        # サンプル推薦の表示
        print("\n📋 Sample recommendations:")
        for i, (task_id, devs) in enumerate(recommendations[:5]):
            expert_dev = self.test_expert_assignments.get(task_id, "Unknown")
            print(f"  {i+1}. Task {task_id} → {devs[:3]} (Expert: {expert_dev})")
        
        # 精度計算
        print("\n🎯 Calculating accuracy...")
        accuracies = self.calculate_accuracy(recommendations)
        
        # 結果表示
        print("\n📈 Bot-Excluded Expert Model Results:")
        print("=" * 50)
        
        model_type = "Bot-Excluded Expert Model" if self.models_loaded else "Random Baseline"
        print(f"Model Type: {model_type}")
        
        print("\nModel Performance:")
        for metric, value in accuracies.items():
            if metric.endswith('_accuracy'):
                print(f"  {metric:20s}: {value:.3f} ({value*100:.1f}%)")
        
        print(f"\nBaseline Comparisons:")
        print(f"  Random Top-1       : {baseline_results.get('random_top_1', 0):.3f} ({baseline_results.get('random_top_1', 0)*100:.1f}%)")
        print(f"  Frequent Dev Top-1 : {baseline_results.get('frequent_top_1', 0):.3f} ({baseline_results.get('frequent_top_1', 0)*100:.1f}%)")
        
        print(f"\nDetailed Statistics:")
        print(f"  Test expert assignments: {len(self.test_expert_assignments)}")
        print(f"  Total recommendations: {accuracies.get('total_recommendations', 0)}")
        print(f"  Valid recommendations: {accuracies.get('total_valid_recommendations', 0)}")
        print(f"  Models loaded: {self.models_loaded}")
        
        return accuracies, baseline_results


@hydra.main(version_base="1.3", config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """メイン実行関数"""
    
    try:
        evaluator = BotExcludedExpertModelEvaluator(cfg, train_ratio=0.7)
        model_results, baseline_results = evaluator.evaluate()
        
        print("\n✅ Bot-excluded expert model evaluation completed!")
        
        return {
            'model_performance': model_results,
            'baseline_performance': baseline_results,
            'bot_excluded': True,
            'expert_trajectories_used': True,
            'models_loaded': evaluator.models_loaded
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
