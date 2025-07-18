#!/usr/bin/env python3
"""
GAT特徴量統合強化学習推薦システム

- 既存のFeatureExtractor（GAT+IRL重み）を活用
- 62次元の高次元特徴量で強化学習
- シンプル類似度ベースとの性能比較
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# プロジェクトのモジュール
sys.path.append('/Users/kazuki-h/rl/kazoo')
sys.path.append('/Users/kazuki-h/rl/kazoo/src')

from src.kazoo.envs.task import Task
from src.kazoo.features.feature_extractor import FeatureExtractor


class GATEnhancedRLEnvironment(gym.Env):
    """
    GAT特徴量統合強化学習環境
    - 62次元GAT特徴量を状態表現に使用
    - 複雑な開発者-タスク関係をモデル化
    - IRL重みを報酬計算に活用
    """
    
    def __init__(self, config, training_data, dev_profiles):
        super().__init__()
        
        self.config = config
        self.training_data = training_data
        self.dev_profiles = dev_profiles
        
        # 特徴量抽出器の初期化
        self.setup_feature_extractor()
        
        # 開発者リスト
        self.developers = list(dev_profiles.keys())
        self.num_developers = len(self.developers)
        
        # 行動空間: 開発者選択
        self.action_space = gym.spaces.Discrete(self.num_developers)
        
        # 観測空間: GAT特徴量 (62次元)
        feature_dim = len(self.feature_extractor.feature_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(feature_dim,),
            dtype=np.float32
        )
        
        # 学習データからペアを抽出
        self.training_pairs = self._extract_training_pairs()
        
        # 現在のエピソード状態
        self.current_pair = None
        self.episode_count = 0
        
        print(f"🤖 GAT統合強化学習環境初期化完了")
        print(f"   開発者数: {self.num_developers}")
        print(f"   特徴量次元: {feature_dim} (GAT含む)")
        print(f"   学習ペア数: {len(self.training_pairs)}")
        print(f"   行動空間: {self.action_space}")
    
    def setup_feature_extractor(self):
        """GAT特徴量抽出器の初期化"""
        print("🔧 GAT特徴量抽出器初期化中...")
        
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
        
        # IRL重みを読み込み
        self.irl_weights = self._load_irl_weights()
        
        print(f"   特徴量次元: {len(self.feature_extractor.feature_names)}")
        print(f"   GAT特徴量: {sum(1 for name in self.feature_extractor.feature_names if 'gat_' in name)}")
    
    def _load_irl_weights(self):
        """IRL学習済み重みを読み込み"""
        weights_path = self.config.get('irl', {}).get('output_weights_path')
        
        if weights_path and Path(weights_path).exists():
            try:
                weights = np.load(weights_path)
                print(f"✅ IRL重み読み込み: {weights_path} ({weights.shape})")
                return torch.tensor(weights, dtype=torch.float32)
            except Exception as e:
                print(f"⚠️ IRL重み読み込みエラー: {e}")
        
        # フォールバック: ランダム重み
        feature_dim = len(self.feature_extractor.feature_names)
        weights = torch.randn(feature_dim, dtype=torch.float32)
        print(f"⚠️ ランダム重みを使用: {weights.shape}")
        return weights
    
    def _extract_training_pairs(self):
        """学習データからタスク-開発者ペアを抽出"""
        pairs = []
        
        for task_data in self.training_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            # 実際の担当者を抽出
            assignee = None
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            
            if assignee and assignee in self.developers:
                pairs.append({
                    'task_data': task_data,
                    'developer': assignee,
                    'task_id': task_id
                })
        
        return pairs
    
    def reset(self, seed=None, options=None):
        """エピソードをリセット"""
        super().reset(seed=seed)
        
        # ランダムにペアを選択
        if self.training_pairs:
            self.current_pair = np.random.choice(self.training_pairs)
        else:
            # フォールバック
            self.current_pair = {
                'task_data': self.training_data[0] if self.training_data else {},
                'developer': self.developers[0] if self.developers else 'unknown',
                'task_id': 'dummy'
            }
        
        # 観測を生成
        obs = self._get_observation()
        
        self.episode_count += 1
        return obs, {}
    
    def step(self, action):
        """アクションを実行"""
        if action >= self.num_developers:
            # 無効なアクション
            reward = -10.0
            terminated = True
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, reward, terminated, False, {}
        
        selected_dev = self.developers[action]
        actual_dev = self.current_pair['developer']
        
        # 報酬計算（GAT特徴量ベース）
        reward = self._calculate_gat_reward(selected_dev, actual_dev)
        
        # エピソード終了
        terminated = True
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        info = {
            'selected_dev': selected_dev,
            'actual_dev': actual_dev,
            'task_id': self.current_pair['task_id'],
            'correct': selected_dev == actual_dev,
            'reward_breakdown': getattr(self, 'last_reward_breakdown', {})
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self):
        """GAT特徴量による観測を取得"""
        try:
            # タスクオブジェクト作成
            task_obj = Task(self.current_pair['task_data'])
            
            # 実際の開発者オブジェクト
            actual_dev_name = self.current_pair['developer']
            dev_profile = self.dev_profiles.get(actual_dev_name, {})
            developer_obj = {"name": actual_dev_name, "profile": dev_profile}
            
            # ダミー環境作成
            dummy_env = type('DummyEnv', (), {
                'backlog': [task_obj],
                'dev_profiles': self.dev_profiles,
                'assignments': {},
                'dev_action_history': {}
            })()
            
            # GAT特徴量抽出
            features = self.feature_extractor.get_features(
                task_obj, developer_obj, dummy_env
            )
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ 観測取得エラー: {e}")
            # フォールバック: ゼロベクトル
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _calculate_gat_reward(self, selected_dev, actual_dev):
        """GAT特徴量とIRL重みを活用した報酬計算"""
        total_reward = 0.0
        reward_breakdown = {}
        
        # 1. 正解報酬 (最重要)
        if selected_dev == actual_dev:
            correct_reward = 10.0
            total_reward += correct_reward
            reward_breakdown['correct'] = correct_reward
        else:
            reward_breakdown['correct'] = 0.0
        
        # 2. GAT特徴量ベース報酬
        try:
            # 選択された開発者での特徴量抽出
            task_obj = Task(self.current_pair['task_data'])
            selected_dev_profile = self.dev_profiles.get(selected_dev, {})
            selected_dev_obj = {"name": selected_dev, "profile": selected_dev_profile}
            
            dummy_env = type('DummyEnv', (), {
                'backlog': [task_obj],
                'dev_profiles': self.dev_profiles,
                'assignments': {},
                'dev_action_history': {}
            })()
            
            # GAT特徴量抽出
            features = self.feature_extractor.get_features(
                task_obj, selected_dev_obj, dummy_env
            )
            features_tensor = torch.tensor(features, dtype=torch.float32)
            
            # IRL重みとの内積で適合度計算
            compatibility_score = torch.dot(self.irl_weights, features_tensor).item()
            
            # 正規化して報酬に変換
            gat_reward = np.tanh(compatibility_score) * 5.0  # -5.0 to +5.0
            total_reward += gat_reward
            reward_breakdown['gat_compatibility'] = gat_reward
            
            # 3. GAT特定特徴量ボーナス
            gat_indices = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                          if 'gat_' in name]
            
            if gat_indices:
                gat_features = features[gat_indices]
                
                # GAT類似度特徴量
                gat_similarity_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                                    if name == 'gat_similarity']
                if gat_similarity_idx:
                    similarity_score = features[gat_similarity_idx[0]]
                    similarity_reward = similarity_score * 3.0
                    total_reward += similarity_reward
                    reward_breakdown['gat_similarity'] = similarity_reward
                
                # GAT専門性特徴量
                expertise_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                               if name == 'gat_dev_expertise']
                if expertise_idx:
                    expertise_score = features[expertise_idx[0]]
                    expertise_reward = expertise_score * 2.0
                    total_reward += expertise_reward
                    reward_breakdown['gat_expertise'] = expertise_reward
                
                # GAT協力強度特徴量
                collaboration_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                                   if name == 'gat_collaboration_strength']
                if collaboration_idx:
                    collab_score = features[collaboration_idx[0]]
                    collab_reward = collab_score * 2.0
                    total_reward += collab_reward
                    reward_breakdown['gat_collaboration'] = collab_reward
        
        except Exception as e:
            print(f"⚠️ GAT報酬計算エラー: {e}")
            reward_breakdown['gat_compatibility'] = 0.0
        
        # 4. 開発者経験報酬
        dev_profile = self.dev_profiles.get(selected_dev, {})
        if 'expertise' in dev_profile:
            expertise_count = len(dev_profile['expertise'])
            exp_reward = min(expertise_count * 0.1, 1.0)  # 最大1.0
            total_reward += exp_reward
            reward_breakdown['dev_experience'] = exp_reward
        
        # 報酬詳細を保存
        self.last_reward_breakdown = reward_breakdown
        
        return total_reward


class GATRLRecommender:
    """GAT特徴量統合強化学習推薦システム"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🚀 GAT統合強化学習推薦システム初期化完了")
    
    def load_data(self, data_path):
        """データを読み込んで時系列分割"""
        print("📊 データ読み込み中...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # 時系列分割
        training_data = []  # 2022年以前
        test_data = []      # 2023年
        
        for task in all_data:
            created_at = task.get('created_at', '')
            if created_at.startswith('2022'):
                training_data.append(task)
            elif created_at.startswith('2023'):
                test_data.append(task)
        
        print(f"   学習データ: {len(training_data):,} タスク (2022年)")
        print(f"   テストデータ: {len(test_data):,} タスク (2023年)")
        
        return training_data, test_data
    
    def load_dev_profiles(self):
        """開発者プロファイルを読み込み"""
        dev_profiles_path = self.config['env']['dev_profiles_path']
        with open(dev_profiles_path, 'r', encoding='utf-8') as f:
            dev_profiles = yaml.safe_load(f)
        
        print(f"📋 開発者プロファイル読み込み: {len(dev_profiles)} 人")
        return dev_profiles
    
    def train_rl_agent(self, training_data, dev_profiles, total_timesteps=100000):
        """GAT特徴量で強化学習エージェントを訓練"""
        print("🎓 GAT統合強化学習エージェント訓練開始...")
        
        # 環境作成
        def make_env():
            return GATEnhancedRLEnvironment(self.config, training_data, dev_profiles)
        
        env = DummyVecEnv([make_env])
        
        # PPOエージェント（高次元特徴量対応）
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # 探索促進
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="auto",
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # 大きなネットワーク
                activation_fn=torch.nn.ReLU
            )
        )
        
        # 評価コールバック
        eval_env = DummyVecEnv([make_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/gat_rl_best/",
            log_path="./logs/gat_rl_eval/",
            eval_freq=5000,
            deterministic=True,
            render=False,
            verbose=1
        )
        
        # 訓練実行
        print(f"   訓練ステップ: {total_timesteps:,}")
        print(f"   推定時間: {total_timesteps / 1000:.1f}分")
        
        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False  # プログレスバー無効化
        )
        
        print("✅ GAT統合強化学習エージェント訓練完了")
    
    def predict_with_gat_rl(self, test_data, dev_profiles):
        """GAT特徴量強化学習で予測"""
        print("🤖 GAT強化学習予測実行中...")
        
        predictions = {}
        prediction_scores = {}
        
        # テスト環境
        test_env = GATEnhancedRLEnvironment(self.config, test_data, dev_profiles)
        
        # テストデータの実際の割り当てを抽出
        test_assignments = {}
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
                
            assignee = None
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            
            if assignee:
                test_assignments[task_id] = assignee
        
        print(f"   予測対象: {len(test_assignments)} タスク")
        
        prediction_count = 0
        
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if task_id not in test_assignments:
                continue
            
            try:
                # テスト環境でタスクを設定
                test_env.current_pair = {
                    'task_data': task_data,
                    'developer': test_assignments[task_id],
                    'task_id': task_id
                }
                
                # 観測取得
                obs = test_env._get_observation()
                
                # 強化学習エージェントで予測
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                
                if action < len(test_env.developers):
                    predicted_dev = test_env.developers[action]
                    
                    predictions[task_id] = predicted_dev
                    prediction_scores[task_id] = {
                        'predicted_dev': predicted_dev,
                        'action': int(action),
                        'method': 'gat_reinforcement_learning'
                    }
                    
                    prediction_count += 1
                    
                    if prediction_count % 50 == 0:
                        print(f"   進捗: {prediction_count}/{len(test_assignments)}")
            
            except Exception as e:
                print(f"⚠️ タスク {task_id} のGAT-RL予測エラー: {e}")
                continue
        
        print(f"   GAT-RL予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores, test_assignments
    
    def compare_with_baseline(self, gat_predictions, test_assignments, baseline_path=None):
        """シンプル類似度ベースとの比較"""
        print("📊 ベースライン比較実行中...")
        
        # GAT-RL評価
        gat_common_tasks = set(gat_predictions.keys()) & set(test_assignments.keys())
        gat_correct = sum(1 for task_id in gat_common_tasks 
                         if gat_predictions[task_id] == test_assignments[task_id])
        gat_accuracy = gat_correct / len(gat_common_tasks) if gat_common_tasks else 0.0
        
        print(f"🤖 GAT強化学習システム:")
        print(f"   精度: {gat_accuracy:.3f} ({gat_correct}/{len(gat_common_tasks)})")
        
        # ベースライン比較（可能であれば）
        if baseline_path and Path(baseline_path).exists():
            baseline_results = pd.read_csv(baseline_path)
            baseline_accuracy = baseline_results['correct'].mean()
            
            print(f"📝 シンプル類似度ベース:")
            print(f"   精度: {baseline_accuracy:.3f}")
            
            improvement = gat_accuracy - baseline_accuracy
            print(f"📈 改善度: {improvement:+.3f} ({improvement/baseline_accuracy*100:+.1f}%)")
        
        return {
            'gat_rl_accuracy': gat_accuracy,
            'gat_rl_correct': gat_correct,
            'gat_rl_total': len(gat_common_tasks)
        }
    
    def save_gat_rl_model(self, output_dir="models"):
        """GAT統合強化学習モデルの保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RLエージェント保存
        rl_model_path = output_dir / f"gat_rl_recommender_{timestamp}.zip"
        self.rl_agent.save(rl_model_path)
        
        print(f"✅ GAT統合RLモデル保存: {rl_model_path}")
        return rl_model_path
    
    def run_full_pipeline(self, data_path, baseline_path=None, total_timesteps=100000):
        """完全パイプライン実行"""
        print("🚀 GAT統合強化学習推薦システム実行開始")
        print("=" * 70)
        
        # 1. データ読み込み
        training_data, test_data = self.load_data(data_path)
        
        # 2. 開発者プロファイル読み込み
        dev_profiles = self.load_dev_profiles()
        
        # 3. 強化学習エージェント訓練
        self.train_rl_agent(training_data, dev_profiles, total_timesteps)
        
        # 4. 予測実行
        predictions, scores, test_assignments = self.predict_with_gat_rl(test_data, dev_profiles)
        
        # 5. ベースラインとの比較
        metrics = self.compare_with_baseline(predictions, test_assignments, baseline_path)
        
        # 6. モデル保存
        self.save_gat_rl_model()
        
        print("✅ GAT統合強化学習推薦システム完了")
        return metrics


def main():
    parser = argparse.ArgumentParser(description='GAT統合強化学習推薦システム')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--data', default='data/backlog.json',
                       help='統合データパス')
    parser.add_argument('--baseline', 
                       help='ベースライン結果CSVパス')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='訓練ステップ数')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # GAT統合強化学習システム実行
    recommender = GATRLRecommender(args.config)
    metrics = recommender.run_full_pipeline(
        args.data, args.baseline, args.timesteps
    )
    
    print("\n🎯 最終結果:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
