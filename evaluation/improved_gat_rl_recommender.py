#!/usr/bin/env python3
"""
改善版GAT特徴量統合強化学習推薦システム

修正点:
1. 行動空間を実際の訓練開発者に制限
2. より密な報酬設計
3. GAT特徴量の効果的活用
4. シンプル類似度ベースとの正確な比較
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


class ImprovedGATRLEnvironment(gym.Env):
    """
    改善版GAT特徴量統合強化学習環境
    - 実際の訓練開発者のみを行動空間に含める
    - GAT特徴量を最大限活用
    - 密な報酬設計
    """
    
    def __init__(self, config, training_data, dev_profiles):
        super().__init__()
        
        self.config = config
        self.training_data = training_data
        self.dev_profiles = dev_profiles
        
        # 特徴量抽出器の初期化
        self.setup_feature_extractor()
        
        # 実際の訓練開発者のみ抽出
        self.active_developers = self._extract_active_developers()
        self.num_developers = len(self.active_developers)
        
        print(f"🎯 行動空間を実訓練開発者に制限: {self.num_developers}人")
        
        # 行動空間: 実際の開発者のみ
        self.action_space = gym.spaces.Discrete(self.num_developers)
        
        # 観測空間: GAT特徴量のみ（開発者選択肢は除外）
        feature_dim = len(self.feature_extractor.feature_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(feature_dim,),  # GAT特徴量のみ
            dtype=np.float32
        )
        
        # 学習データからペアを抽出
        self.training_pairs = self._extract_training_pairs()
        
        # 現在のエピソード状態
        self.current_pair = None
        self.episode_count = 0
        
        print(f"🤖 改善版GAT-RL環境初期化完了")
        print(f"   アクティブ開発者: {self.num_developers}")
        print(f"   観測次元: {self.observation_space.shape[0]} (GAT特徴量のみ)")
        print(f"   学習ペア数: {len(self.training_pairs)}")
    
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
    
    def _extract_active_developers(self):
        """実際に訓練データに登場する開発者を抽出"""
        active_devs = set()
        
        for task_data in self.training_data:
            # 実際の担当者を抽出
            assignee = None
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            
            if assignee and assignee in self.dev_profiles:
                active_devs.add(assignee)
        
        active_list = sorted(list(active_devs))
        print(f"📋 アクティブ開発者抽出: {len(active_list)}人")
        
        # 開発者統計
        dev_stats = Counter()
        for task_data in self.training_data:
            assignee = None
            if 'assignees' in task_data and task_data['assignees']:
                assignee = task_data['assignees'][0].get('login')
            elif 'events' in task_data:
                for event in task_data['events']:
                    if event.get('event') == 'assigned' and event.get('assignee'):
                        assignee = event['assignee'].get('login')
                        break
            if assignee in active_list:
                dev_stats[assignee] += 1
        
        print("   上位アクティブ開発者:")
        for dev, count in dev_stats.most_common(10):
            print(f"     {dev}: {count} タスク")
        
        return active_list
    
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
            
            if assignee and assignee in self.active_developers:
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
                'developer': self.active_developers[0] if self.active_developers else 'unknown',
                'task_id': 'dummy'
            }
        
        # 観測を生成
        obs = self._get_observation()
        
        self.episode_count += 1
        return obs, {}
    
    def step(self, action):
        """アクションを実行"""
        if action >= self.num_developers:
            # 無効なアクションに対する重いペナルティ
            reward = -50.0  # さらに重いペナルティ
            terminated = True
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            return obs, reward, terminated, False, {}
        
        selected_dev = self.active_developers[action]
        actual_dev = self.current_pair['developer']
        
        # 報酬計算（改善版）
        reward = self._calculate_improved_reward(selected_dev, actual_dev)
        
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
        """改善版観測生成（GAT特徴量のみ）"""
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
            
            # GAT特徴量抽出（62次元固定）
            gat_features = self.feature_extractor.get_features(
                task_obj, developer_obj, dummy_env
            )
            
            return gat_features.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ 観測取得エラー: {e}")
            # フォールバック: ゼロベクトル
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
    
    def _calculate_improved_reward(self, selected_dev, actual_dev):
        """改善版報酬計算 - 正解報酬を圧倒的に重要に"""
        total_reward = 0.0
        reward_breakdown = {}
        
        # 1. 正解報酬 (圧倒的に重要)
        if selected_dev == actual_dev:
            correct_reward = 100.0  # 大幅増加: 正解が最優先
            total_reward += correct_reward
            reward_breakdown['correct'] = correct_reward
        else:
            # 間違いに対する明確なペナルティ
            wrong_penalty = -20.0  # ペナルティ強化
            total_reward += wrong_penalty
            reward_breakdown['correct'] = wrong_penalty
        
        # 2. GAT特徴量報酬 (補助的役割のみ)
        try:
            # 選択された開発者でのGAT特徴量抽出
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
            
            # GAT類似度特徴量 (補助的)
            gat_similarity_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                                if name == 'gat_similarity']
            if gat_similarity_idx:
                similarity_score = features[gat_similarity_idx[0]]
                similarity_reward = similarity_score * 2.0  # 大幅削減: 補助的役割
                total_reward += similarity_reward
                reward_breakdown['gat_similarity'] = similarity_reward
            
            # GAT専門性報酬 (補助的)
            expertise_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                           if name == 'gat_dev_expertise']
            if expertise_idx:
                expertise_score = features[expertise_idx[0]]
                expertise_reward = expertise_score * 1.5  # 大幅削減
                total_reward += expertise_reward
                reward_breakdown['gat_expertise'] = expertise_reward
            
            # GAT協力強度報酬 (補助的)
            collaboration_idx = [i for i, name in enumerate(self.feature_extractor.feature_names) 
                               if name == 'gat_collaboration_strength']
            if collaboration_idx:
                collab_score = features[collaboration_idx[0]]
                collab_reward = collab_score * 1.0  # 大幅削減
                total_reward += collab_reward
                reward_breakdown['gat_collaboration'] = collab_reward
            
            # IRL重みとの統合報酬 (補助的)
            features_tensor = torch.tensor(features, dtype=torch.float32)
            irl_score = torch.dot(self.irl_weights, features_tensor).item()
            irl_reward = np.tanh(irl_score) * 1.0  # 大幅削減
            total_reward += irl_reward
            reward_breakdown['irl_compatibility'] = irl_reward
            
        except Exception as e:
            print(f"⚠️ GAT報酬計算エラー: {e}")
            # エラー時はGAT報酬を0に
            reward_breakdown['gat_similarity'] = 0.0
            reward_breakdown['gat_expertise'] = 0.0
            reward_breakdown['gat_collaboration'] = 0.0
            reward_breakdown['irl_compatibility'] = 0.0
        
        # 報酬詳細を保存
        self.last_reward_breakdown = reward_breakdown
        
        return total_reward


class ImprovedGATRLRecommender:
    """改善版GAT特徴量統合強化学習推薦システム"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🚀 改善版GAT統合強化学習推薦システム初期化完了")
    
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
    
    def train_improved_rl_agent(self, training_data, dev_profiles, total_timesteps=50000):
        """改善版GAT強化学習エージェントを訓練"""
        print("🎓 改善版GAT統合強化学習エージェント訓練開始...")
        
        # 環境作成
        def make_env():
            return ImprovedGATRLEnvironment(self.config, training_data, dev_profiles)
        
        env = DummyVecEnv([make_env])
        
        # PPOエージェント（改善版）
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-3,  # 学習率上昇
            n_steps=1024,        # ステップ数削減
            batch_size=128,      # バッチサイズ増加
            n_epochs=20,         # エポック数増加
            gamma=0.95,          # 割引率調整
            gae_lambda=0.9,      # GAE調整
            clip_range=0.3,      # クリップ範囲拡大
            ent_coef=0.05,       # エントロピー係数増加
            vf_coef=1.0,         # 価値関数係数増加
            max_grad_norm=1.0,   # 勾配クリッピング増加
            device="auto",
            policy_kwargs=dict(
                net_arch=[128, 128, 64],  # ネットワーク調整
                activation_fn=torch.nn.Tanh  # 活性化関数変更
            )
        )
        
        # 評価コールバック
        eval_env = DummyVecEnv([make_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/improved_gat_rl_best/",
            log_path="./logs/improved_gat_rl_eval/",
            eval_freq=2000,
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
            progress_bar=False
        )
        
        print("✅ 改善版GAT統合強化学習エージェント訓練完了")
    
    def predict_with_improved_gat_rl(self, test_data, dev_profiles):
        """改善版GAT強化学習で予測"""
        print("🤖 改善版GAT強化学習予測実行中...")
        
        # テスト環境作成
        test_env = ImprovedGATRLEnvironment(self.config, test_data, dev_profiles)
        
        predictions = {}
        prediction_scores = {}
        
        # テストデータの実際の割り当てを抽出（同じ開発者制限）
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
            
            # アクティブ開発者のみ対象
            if assignee and assignee in test_env.active_developers:
                test_assignments[task_id] = assignee
        
        print(f"   予測対象: {len(test_assignments)} タスク（アクティブ開発者のみ）")
        
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
                
                if action < len(test_env.active_developers):
                    predicted_dev = test_env.active_developers[action]
                    
                    predictions[task_id] = predicted_dev
                    prediction_scores[task_id] = {
                        'predicted_dev': predicted_dev,
                        'action': int(action),
                        'method': 'improved_gat_reinforcement_learning'
                    }
                    
                    prediction_count += 1
                    
                    if prediction_count % 20 == 0:
                        print(f"   進捗: {prediction_count}/{len(test_assignments)}")
            
            except Exception as e:
                print(f"⚠️ タスク {task_id} の改善GAT-RL予測エラー: {e}")
                continue
        
        print(f"   改善GAT-RL予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores, test_assignments
    
    def compare_with_baseline(self, gat_predictions, test_assignments, baseline_path=None):
        """シンプル類似度ベースとの詳細比較"""
        print("📊 詳細ベースライン比較実行中...")
        
        # GAT-RL評価
        gat_common_tasks = set(gat_predictions.keys()) & set(test_assignments.keys())
        gat_correct = sum(1 for task_id in gat_common_tasks 
                         if gat_predictions[task_id] == test_assignments[task_id])
        gat_accuracy = gat_correct / len(gat_common_tasks) if gat_common_tasks else 0.0
        
        print(f"🤖 改善版GAT強化学習システム:")
        print(f"   精度: {gat_accuracy:.3f} ({gat_correct}/{len(gat_common_tasks)})")
        
        # ベースライン読み込み
        baseline_accuracy = 0.0
        baseline_correct = 0
        baseline_total = 0
        
        if baseline_path and Path(baseline_path).exists():
            try:
                baseline_df = pd.read_csv(baseline_path)
                baseline_accuracy = baseline_df['correct'].mean()
                baseline_correct = baseline_df['correct'].sum()
                baseline_total = len(baseline_df)
                
                print(f"📝 シンプル類似度ベースライン:")
                print(f"   精度: {baseline_accuracy:.3f} ({baseline_correct}/{baseline_total})")
                
                improvement = gat_accuracy - baseline_accuracy
                print(f"📈 改善度: {improvement:+.3f} ({improvement/baseline_accuracy*100:+.1f}%)")
                
            except Exception as e:
                print(f"⚠️ ベースライン読み込みエラー: {e}")
        
        return {
            'improved_gat_rl_accuracy': gat_accuracy,
            'improved_gat_rl_correct': gat_correct,
            'improved_gat_rl_total': len(gat_common_tasks),
            'baseline_accuracy': baseline_accuracy,
            'baseline_correct': baseline_correct,
            'baseline_total': baseline_total,
            'improvement': gat_accuracy - baseline_accuracy
        }
    
    def save_improved_gat_rl_model(self, output_dir="models"):
        """改善版GAT統合強化学習モデルの保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RLエージェント保存
        rl_model_path = output_dir / f"improved_gat_rl_recommender_{timestamp}.zip"
        self.rl_agent.save(rl_model_path)
        
        print(f"✅ 改善版GAT統合RLモデル保存: {rl_model_path}")
        return rl_model_path
    
    def run_full_pipeline(self, data_path, baseline_path=None, total_timesteps=50000):
        """完全パイプライン実行"""
        print("🚀 改善版GAT統合強化学習推薦システム実行開始")
        print("=" * 70)
        
        # 1. データ読み込み
        training_data, test_data = self.load_data(data_path)
        
        # 2. 開発者プロファイル読み込み
        dev_profiles = self.load_dev_profiles()
        
        # 3. 改善版強化学習エージェント訓練
        self.train_improved_rl_agent(training_data, dev_profiles, total_timesteps)
        
        # 4. 予測実行
        predictions, scores, test_assignments = self.predict_with_improved_gat_rl(test_data, dev_profiles)
        
        # 5. ベースラインとの比較
        metrics = self.compare_with_baseline(predictions, test_assignments, baseline_path)
        
        # 6. モデル保存
        self.save_improved_gat_rl_model()
        
        print("✅ 改善版GAT統合強化学習推薦システム完了")
        return metrics


def main():
    parser = argparse.ArgumentParser(description='改善版GAT統合強化学習推薦システム')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--data', default='data/backlog.json',
                       help='統合データパス')
    parser.add_argument('--baseline', 
                       help='ベースライン結果CSVパス')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='訓練ステップ数')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 改善版GAT統合強化学習システム実行
    recommender = ImprovedGATRLRecommender(args.config)
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
