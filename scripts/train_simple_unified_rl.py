#!/usr/bin/env python3
"""
シンプル統合強化学習システム - 簡単なGym環境版
複雑なDict観測空間を避けて、Stable-Baselines3で動作する版
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# パス設定
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kazoo.features.feature_extractor import FeatureExtractor


class SimpleTaskAssignmentEnv(gym.Env):
    """
    シンプルなタスク割り当て環境
    - Dictではなく単純なBox観測空間を使用
    - IRL重みベースの報酬計算
    """
    
    def __init__(self, cfg, backlog_data, dev_profiles_data):
        super().__init__()
        
        self.cfg = cfg
        self.backlog_data = backlog_data
        self.dev_profiles_data = dev_profiles_data
        
        # 特徴量抽出器
        self.feature_extractor = FeatureExtractor(cfg)
        
        # IRL重みを読み込み
        self.irl_weights = self._load_irl_weights()
        
        # 環境設定
        self.num_developers = min(len(dev_profiles_data), 
                                cfg.optimization.get('max_developers', 50))
        self.max_tasks = min(len(backlog_data), 
                           cfg.optimization.get('max_tasks', 200))
        self.max_steps = cfg.env.get('max_steps', 100)
        
        # 行動・観測空間
        self.action_space = gym.spaces.Discrete(self.num_developers)
        
        feature_dim = len(self.feature_extractor.feature_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(feature_dim,), 
            dtype=np.float32
        )
        
        # 開発者・タスクリストを作成
        self.developers = list(dev_profiles_data.keys())[:self.num_developers]
        
        # タスクをTaskオブジェクトに変換
        from kazoo.envs.task import Task
        self.Task = Task  # クラス参照を保存
        self.tasks = []
        conversion_errors = 0
        
        for i, task_dict in enumerate(backlog_data[:self.max_tasks]):
            try:
                task_obj = Task(task_dict)
                # デバッグ: updated_atが正しく設定されているかチェック
                if task_obj.updated_at is None:
                    print(f"⚠️ Task {i} has None updated_at: {task_dict.get('updated_at')}")
                self.tasks.append(task_obj)
            except Exception as e:
                print(f"⚠️ Failed to create Task object for task {i}: {e}")
                print(f"   Task data keys: {list(task_dict.keys()) if isinstance(task_dict, dict) else 'Not a dict'}")
                conversion_errors += 1
                # エラーが発生した場合は、デフォルト値で Task オブジェクトを作成
                default_task = {
                    'id': task_dict.get('id', i),
                    'number': task_dict.get('number', i),
                    'title': task_dict.get('title', 'Unknown Task'),
                    'body': task_dict.get('body', ''),
                    'state': task_dict.get('state', 'open'),
                    'created_at': task_dict.get('created_at', '2022-01-01T00:00:00Z'),
                    'updated_at': task_dict.get('updated_at', '2022-01-01T00:00:00Z'),
                    'labels': task_dict.get('labels', []),
                    'user': task_dict.get('user', {'login': 'unknown'}),
                    'comments': task_dict.get('comments', 0)
                }
                self.tasks.append(Task(default_task))
        
        if conversion_errors > 0:
            print(f"⚠️ Task conversion errors: {conversion_errors}/{self.max_tasks}")
        
        print(f"✅ All tasks converted to Task objects: {len(self.tasks)}")
        
        # 状態管理
        self.current_task_idx = 0
        self.step_count = 0
        self.assignments = {}
        
        print(f"🎮 シンプル環境初期化完了")
        print(f"   開発者数: {self.num_developers}")
        print(f"   タスク数: {len(self.tasks)}")
        print(f"   特徴量次元: {feature_dim}")
        print(f"   最大ステップ数: {self.max_steps}")
        
    def _load_irl_weights(self):
        """IRL学習済み重みを読み込み"""
        weights_path = self.cfg.irl.get('output_weights_path')
        
        if weights_path and Path(weights_path).exists():
            try:
                weights = np.load(weights_path)
                print(f"✅ IRL重みを読み込み: {weights_path} ({weights.shape})")
                return torch.tensor(weights, dtype=torch.float32)
            except Exception as e:
                print(f"⚠️ IRL重み読み込みエラー: {e}")
        
        # フォールバック: ランダム重み
        feature_dim = len(self.feature_extractor.feature_names)
        weights = torch.randn(feature_dim, dtype=torch.float32)
        print(f"⚠️ ランダム重みを使用: {weights.shape}")
        return weights
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        self.current_task_idx = 0
        self.step_count = 0
        self.assignments = {}
        
        # 最初の観測を取得
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """ステップ実行"""
        if (self.current_task_idx >= len(self.tasks) or 
            action >= self.num_developers):
            # 終了条件
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}
        
        # 現在のタスクと選択された開発者
        current_task = self.tasks[self.current_task_idx]
        
        # タスクオブジェクトの型をチェック
        if not hasattr(current_task, 'updated_at'):
            print(f"⚠️ Task {self.current_task_idx} is not a Task object in step: {type(current_task)}")
            # 辞書の場合は Task オブジェクトに変換
            if isinstance(current_task, dict):
                current_task = self.Task(current_task)
                self.tasks[self.current_task_idx] = current_task
        
        developer_name = self.developers[action]
        developer_profile = self.dev_profiles_data.get(developer_name, {})
        developer_obj = {"name": developer_name, "profile": developer_profile}
        
        # 報酬計算
        reward = self._calculate_reward(current_task, developer_obj)
        
        # 割り当てを記録
        task_id = current_task.id if hasattr(current_task, 'id') else current_task.get('id', self.current_task_idx)
        self.assignments[task_id] = developer_name
        
        # 次のタスクに移行
        self.current_task_idx += 1
        self.step_count += 1
        
        # 終了条件をチェック
        terminated = (self.current_task_idx >= len(self.tasks) or 
                     self.step_count >= self.max_steps)
        
        # 次の観測
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()
        
        info = {
            'task_id': task_id,
            'developer': developer_name,
            'step': self.step_count,
            'assignments_count': len(self.assignments)
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self):
        """現在の観測を取得"""
        if self.current_task_idx >= len(self.tasks):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        current_task = self.tasks[self.current_task_idx]
        
        # タスクオブジェクトの型をチェック
        if not hasattr(current_task, 'updated_at'):
            print(f"⚠️ Task {self.current_task_idx} is not a Task object: {type(current_task)}")
            # 辞書の場合は Task オブジェクトに変換
            if isinstance(current_task, dict):
                current_task = self.Task(current_task)
                self.tasks[self.current_task_idx] = current_task
        
        # 最初の開発者の特徴量を代表として使用
        try:
            developer_name = self.developers[0]
            developer_profile = self.dev_profiles_data.get(developer_name, {})
            developer_obj = {"name": developer_name, "profile": developer_profile}
            
            # ダミー環境オブジェクトを作成
            dummy_env = type('DummyEnv', (), {
                'backlog': [t if hasattr(t, 'updated_at') else self.Task(t) for t in self.backlog_data[:self.max_tasks]],
                'dev_profiles': self.dev_profiles_data,
                'assignments': {},
                'dev_action_history': {}  # 開発者のアクション履歴
            })()
            
            features = self.feature_extractor.get_features(
                current_task, developer_obj, dummy_env
            )
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ 観測取得エラー: {e}")
            print(f"   Task type: {type(current_task)}")
            if hasattr(current_task, 'updated_at'):
                print(f"   Task updated_at: {current_task.updated_at}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _calculate_reward(self, task, developer):
        """IRL重みを使用した報酬計算"""
        try:
            # タスクオブジェクトの型をチェック
            if not hasattr(task, 'updated_at'):
                print(f"⚠️ Task is not a Task object in reward calculation: {type(task)}")
                # 辞書の場合は Task オブジェクトに変換
                if isinstance(task, dict):
                    task = self.Task(task)
            
            # ダミー環境オブジェクトを作成
            dummy_env = type('DummyEnv', (), {
                'backlog': [t if hasattr(t, 'updated_at') else self.Task(t) for t in self.backlog_data[:self.max_tasks]],
                'dev_profiles': self.dev_profiles_data,
                'assignments': {},
                'dev_action_history': {}  # 開発者のアクション履歴
            })()
            
            # 特徴量を抽出
            features = self.feature_extractor.get_features(task, developer, dummy_env)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # IRL重みとの内積で報酬を計算
            reward = torch.dot(self.irl_weights, features_tensor).item()
            
            # 報酬を正規化
            reward = np.clip(reward, -10.0, 10.0)
            
            return reward
            
        except Exception as e:
            print(f"⚠️ 報酬計算エラー: {e}")
            print(f"   Task type: {type(task)}")
            if hasattr(task, 'updated_at'):
                print(f"   Task updated_at: {task.updated_at}")
            return 0.0


@hydra.main(config_path="../configs", config_name="unified_rl", version_base=None)
def main(cfg: DictConfig):
    """メイン実行関数"""
    
    print("🚀 シンプル統合強化学習システム開始")
    print("=" * 60)
    
    # 1. データ読み込み
    print("1. データ読み込み...")
    with open(cfg.env.backlog_path, "r", encoding="utf-8") as f:
        backlog_data = json.load(f)
    with open(cfg.env.dev_profiles_path, "r", encoding="utf-8") as f:
        dev_profiles_data = yaml.safe_load(f)
    
    print(f"   バックログ: {len(backlog_data)} タスク")
    print(f"   開発者: {len(dev_profiles_data)} 人")
    
    # 2. 環境の作成
    print("2. 環境初期化...")
    def make_env():
        return SimpleTaskAssignmentEnv(cfg, backlog_data, dev_profiles_data)
    
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    # 3. PPOモデル作成
    print("3. PPOモデル作成...")
    
    # 学習率スケジューリング関数
    def linear_schedule(initial_value: float):
        """
        Linear learning rate schedule.
        初期値から最終的に初期値の10%まで線形減衰
        """
        def func(progress_remaining: float) -> float:
            # progress_remaining は 1.0 (開始) から 0.0 (終了) へ
            return progress_remaining * initial_value + (1 - progress_remaining) * 0.1 * initial_value
        return func
    
    # 学習率を動的に設定
    learning_rate = cfg.rl.get('learning_rate', 3e-4)
    use_lr_schedule = cfg.rl.get('use_lr_schedule', True)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(learning_rate) if use_lr_schedule else learning_rate,
        n_steps=cfg.rl.get('n_steps', 2048),
        batch_size=cfg.rl.get('batch_size', 64),
        n_epochs=cfg.rl.get('n_epochs', 10),
        gamma=cfg.rl.get('gamma', 0.99),
        gae_lambda=cfg.rl.get('gae_lambda', 0.95),
        clip_range=cfg.rl.get('clip_range', 0.2),
        clip_range_vf=cfg.rl.get('clip_range_vf', None),
        ent_coef=cfg.rl.get('ent_coef', 0.0),
        vf_coef=cfg.rl.get('vf_coef', 0.5),
        max_grad_norm=cfg.rl.get('max_grad_norm', 0.5),
        device="auto",
        seed=cfg.rl.get('seed', None)
    )
    
    # 4. 訓練実行
    total_timesteps = cfg.rl.get('total_timesteps', 500000)
    print("4. 訓練開始...")
    print(f"   総ステップ数: {total_timesteps:,}")
    print(f"   推定実行時間: {total_timesteps / 1000:.1f}分")
    
    # 評価コールバック
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/simple_unified_best/",
        log_path="./logs/simple_unified_eval/",
        eval_freq=cfg.rl.get('eval_freq', 5000),
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 訓練
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=False  # プログレスバーを無効化
    )
    
    # 5. モデル保存
    print("5. モデル保存...")
    
    # メインモデル保存
    model_path = "models/simple_unified_rl_agent.zip"
    model.save(model_path)
    print(f"✅ モデル保存: {model_path}")
    
    # 訓練統計保存
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    training_info = {
        'timestamp': timestamp,
        'total_timesteps': total_timesteps,
        'final_mean_reward': 'TBD',  # 評価で更新
        'config': dict(cfg.rl)
    }
    
    # JSON形式で訓練情報を保存
    info_path = f"models/training_info_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    print(f"✅ 訓練情報保存: {info_path}")
    
    # 6. 評価
    print("6. 評価実行...")
    evaluate_simple_model(model, make_env(), cfg)
    
    print("✅ シンプル統合強化学習システム完了")


def evaluate_simple_model(model, env, cfg):
    """シンプルモデルの評価 - 推薦システムとしての性能評価も含む"""
    print("📊 性能評価...")
    
    num_episodes = cfg.rl.get('eval_episodes', 10)
    rewards = []
    assignment_counts = []
    
    # 推薦性能評価のため
    all_predictions = {}
    all_confidences = []
    developer_distribution = Counter()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        assignments = 0
        episode_predictions = {}
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            
            # 予測確率分布も取得
            obs_tensor = torch.tensor(obs).unsqueeze(0).float()
            with torch.no_grad():
                # モデルの政策ネットワークから確率分布を取得
                features = model.policy.features_extractor(obs_tensor)
                logits = model.policy.mlp_extractor.policy_net(features)
                probs = torch.softmax(logits, dim=-1).numpy()[0]
            
            # 現在のタスク情報を記録
            if env.current_task_idx < len(env.tasks):
                task = env.tasks[env.current_task_idx]
                task_id = task.id if hasattr(task, 'id') else task.get('id', env.current_task_idx)
                predicted_dev = env.developers[action] if action < len(env.developers) else 'unknown'
                
                episode_predictions[task_id] = {
                    'developer': predicted_dev,
                    'confidence': float(probs[action]),
                    'top_3_probs': sorted(probs, reverse=True)[:3],
                    'episode': episode + 1
                }
                
                developer_distribution[predicted_dev] += 1
                all_confidences.append(probs[action])
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            assignments += 1
            
            if terminated or truncated:
                break
        
        rewards.append(episode_reward)
        assignment_counts.append(assignments)
        all_predictions.update(episode_predictions)
        
        print(f"   Episode {episode + 1}: 報酬={episode_reward:.4f}, 割り当て数={assignments}")
    
    # 基本統計計算
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_assignments = np.mean(assignment_counts)
    
    # 推薦システム評価指標
    recommendation_metrics = calculate_recommendation_metrics(
        all_predictions, all_confidences, developer_distribution, env.developers
    )
    
    print(f"\n🎯 基本評価結果:")
    print(f"   平均報酬: {avg_reward:.4f} ± {std_reward:.4f}")
    print(f"   平均割り当て数: {avg_assignments:.1f}")
    print(f"   最大報酬: {max(rewards):.4f}")
    print(f"   最小報酬: {min(rewards):.4f}")
    
    print(f"\n🎯 推薦システム評価:")
    for metric, value in recommendation_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    # 結果保存
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 基本評価結果
    results_df = pd.DataFrame({
        'episode': range(1, num_episodes + 1),
        'reward': rewards,
        'assignments': assignment_counts
    })
    
    csv_path = f"outputs/simple_unified_evaluation_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"✅ 評価結果保存: {csv_path}")
    
    # 推薦詳細データ保存
    predictions_df = pd.DataFrame([
        {
            'task_id': task_id,
            'predicted_developer': pred['developer'],
            'confidence': pred['confidence'],
            'top_3_avg_prob': np.mean(pred['top_3_probs']),
            'episode': pred['episode']
        }
        for task_id, pred in all_predictions.items()
    ])
    
    predictions_path = f"outputs/simple_unified_predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✅ 予測詳細保存: {predictions_path}")
    
    # 統合サマリー保存
    all_metrics = {
        'avg_reward': avg_reward,
        'std_reward': std_reward, 
        'avg_assignments': avg_assignments,
        'max_reward': max(rewards),
        'min_reward': min(rewards),
        **recommendation_metrics
    }
    
    summary_df = pd.DataFrame([
        {'metric': k, 'value': v} for k, v in all_metrics.items()
    ])
    
    summary_path = f"outputs/simple_unified_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ 統合サマリー保存: {summary_path}")


def calculate_recommendation_metrics(predictions, confidences, dev_distribution, available_developers):
    """推薦システムとしての評価指標を計算"""
    metrics = {}
    
    if not predictions:
        return metrics
    
    # 1. 予測信頼度の統計
    if confidences:
        metrics['avg_confidence'] = np.mean(confidences)
        metrics['confidence_std'] = np.std(confidences)
        metrics['min_confidence'] = np.min(confidences)
        metrics['max_confidence'] = np.max(confidences)
    
    # 2. 開発者推薦の多様性
    total_predictions = len(predictions)
    unique_developers = len(dev_distribution)
    
    metrics['unique_recommended_developers'] = unique_developers
    metrics['total_available_developers'] = len(available_developers)
    metrics['recommendation_coverage'] = unique_developers / len(available_developers) if available_developers else 0
    
    # 3. 推薦集中度（上位開発者への集中度）
    if dev_distribution:
        max_assignments = max(dev_distribution.values())
        metrics['max_assignments_ratio'] = max_assignments / total_predictions
        
        # ジニ係数（推薦の偏り）
        counts = list(dev_distribution.values())
        counts.sort()
        n = len(counts)
        if n > 1:
            gini = (2 * sum((i + 1) * x for i, x in enumerate(counts))) / (n * sum(counts)) - (n + 1) / n
            metrics['recommendation_gini'] = gini
            metrics['recommendation_diversity'] = 1 - gini
    
    # 4. 高信頼度予測の割合
    if confidences:
        high_confidence_threshold = 0.7
        medium_confidence_threshold = 0.5
        
        high_confidence_count = sum(1 for c in confidences if c >= high_confidence_threshold)
        medium_confidence_count = sum(1 for c in confidences if c >= medium_confidence_threshold)
        
        metrics['high_confidence_ratio'] = high_confidence_count / len(confidences)
        metrics['medium_confidence_ratio'] = medium_confidence_count / len(confidences)
    
    return metrics


if __name__ == "__main__":
    main()
