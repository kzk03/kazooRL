#!/usr/bin/env python3
"""
強化学習ベースのタスク推薦システム

- 既存の類似度ベースシステムを強化学習で改良
- RandomForestの予測を報酬として活用
- オフライン強化学習でポリシー最適化
"""

import argparse
import json
import pickle

# プロジェクトのモジュール
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 強化学習関連
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append('/Users/kazuki-h/rl/kazoo')
sys.path.append('/Users/kazuki-h/rl/kazoo/src')

from src.kazoo.envs.task import Task
from src.kazoo.features.feature_extractor import FeatureExtractor


class HybridRecommenderEnvironment(gym.Env):
    """
    ハイブリッド推薦環境
    - 類似度ベースの推薦を強化学習で改良
    - オフライン学習データでポリシーを訓練
    """
    
    def __init__(self, similarity_model, training_data, config):
        super().__init__()
        
        self.similarity_model = similarity_model
        self.training_data = training_data
        self.config = config
        
        # 開発者リスト
        self.developers = list(similarity_model.developer_embeddings.keys())
        self.num_developers = len(self.developers)
        
        # 行動空間: 開発者選択 + 信頼度調整
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_developers,  # 開発者選択
            5                     # 信頼度レベル (0-4)
        ])
        
        # 観測空間: 特徴量 + 類似度スコア
        feature_dim = len(similarity_model.feature_extractor.feature_names)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(feature_dim + self.num_developers,),  # 特徴量 + 各開発者への類似度
            dtype=np.float32
        )
        
        # エピソード管理
        self.current_episode = 0
        self.current_task_idx = 0
        self.tasks = []
        self.ground_truth = {}
        
        print(f"🤖 ハイブリッド推薦環境初期化")
        print(f"   開発者数: {self.num_developers}")
        print(f"   観測次元: {self.observation_space.shape[0]}")
        print(f"   行動空間: {self.action_space}")
    
    def reset(self, seed=None, options=None):
        """エピソードをリセット"""
        super().reset(seed=seed)
        
        # ランダムにタスクを選択
        task_data = np.random.choice(self.training_data)
        self.current_task = Task(task_data)
        
        # 実際の担当者を取得
        self.ground_truth_dev = None
        if 'assignees' in task_data and task_data['assignees']:
            self.ground_truth_dev = task_data['assignees'][0].get('login')
        
        # 観測を生成
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """アクションを実行"""
        dev_idx, confidence_level = action
        
        if dev_idx >= len(self.developers):
            # 無効なアクション
            reward = -1.0
            terminated = True
            obs = np.zeros(self.observation_space.shape[0])
            return obs, reward, terminated, False, {}
        
        selected_dev = self.developers[dev_idx]
        confidence_weight = (confidence_level + 1) / 5.0  # 0.2-1.0
        
        # 報酬計算
        reward = self._calculate_reward(selected_dev, confidence_weight)
        
        # エピソード終了
        terminated = True
        obs = np.zeros(self.observation_space.shape[0])
        
        info = {
            'selected_dev': selected_dev,
            'ground_truth_dev': self.ground_truth_dev,
            'confidence': confidence_weight,
            'similarity_scores': getattr(self, 'last_similarity_scores', {})
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self):
        """現在の観測を取得"""
        # タスクの特徴量
        dummy_env = type('DummyEnv', (), {
            'backlog': [],
            'dev_profiles': {},
            'assignments': {},
            'dev_action_history': {}
        })()
        
        # 代表開発者での特徴量抽出
        representative_dev = {"name": self.developers[0], "profile": {}}
        task_features = self.similarity_model.feature_extractor.get_features(
            self.current_task, representative_dev, dummy_env
        )
        
        # 各開発者への類似度スコア
        similarity_scores = []
        self.last_similarity_scores = {}
        
        for dev_name in self.developers:
            # 開発者の平均特徴量
            dev_embedding = self.similarity_model.developer_embeddings[dev_name]
            
            # コサイン類似度
            similarity = np.dot(task_features, dev_embedding) / (
                np.linalg.norm(task_features) * np.linalg.norm(dev_embedding)
            )
            similarity_scores.append(similarity)
            self.last_similarity_scores[dev_name] = similarity
        
        # 観測ベクトル: 特徴量 + 類似度スコア
        obs = np.concatenate([task_features, similarity_scores])
        return obs.astype(np.float32)
    
    def _calculate_reward(self, selected_dev, confidence_weight):
        """報酬を計算"""
        base_reward = 0.0
        
        # 1. 正解報酬
        if selected_dev == self.ground_truth_dev:
            base_reward += 10.0
        
        # 2. 類似度報酬
        if hasattr(self, 'last_similarity_scores'):
            similarity = self.last_similarity_scores.get(selected_dev, 0.0)
            base_reward += similarity * 5.0
        
        # 3. RandomForest予測報酬
        try:
            dummy_env = type('DummyEnv', (), {
                'backlog': [],
                'dev_profiles': {},
                'assignments': {},
                'dev_action_history': {}
            })()
            
            dev_obj = {"name": selected_dev, "profile": {}}
            features = self.similarity_model.feature_extractor.get_features(
                self.current_task, dev_obj, dummy_env
            )
            features_scaled = self.similarity_model.scaler.transform([features])
            
            # 予測確率
            proba = self.similarity_model.rf_classifier.predict_proba(features_scaled)[0]
            dev_classes = self.similarity_model.rf_classifier.classes_
            
            if selected_dev in dev_classes:
                dev_idx = list(dev_classes).index(selected_dev)
                rf_score = proba[dev_idx]
                base_reward += rf_score * 3.0
        
        except Exception:
            pass
        
        # 4. 信頼度ボーナス
        base_reward *= confidence_weight
        
        return base_reward


class RLEnhancedRecommender:
    """強化学習で拡張された推薦システム"""
    
    def __init__(self, similarity_model, config_path):
        self.similarity_model = similarity_model
        self.config_path = config_path
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("🚀 強化学習拡張推薦システム初期化完了")
    
    def train_rl_policy(self, training_data, total_timesteps=50000):
        """強化学習ポリシーを訓練"""
        print("🎓 強化学習ポリシー訓練開始...")
        
        # 環境作成
        def make_env():
            return HybridRecommenderEnvironment(
                self.similarity_model, training_data, self.config
            )
        
        env = DummyVecEnv([make_env])
        
        # PPOエージェント
        self.rl_agent = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="auto"
        )
        
        # 訓練実行
        print(f"   訓練ステップ: {total_timesteps:,}")
        self.rl_agent.learn(total_timesteps=total_timesteps)
        
        print("✅ 強化学習ポリシー訓練完了")
    
    def predict_with_rl(self, test_data):
        """強化学習ポリシーで予測"""
        print("🤖 強化学習予測実行中...")
        
        predictions = {}
        prediction_scores = {}
        
        # テスト環境
        test_env = HybridRecommenderEnvironment(
            self.similarity_model, test_data, self.config
        )
        
        for i, task_data in enumerate(test_data):
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            try:
                # タスクセット
                test_env.current_task = Task(task_data)
                obs = test_env._get_observation()
                
                # 強化学習エージェントで予測
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                dev_idx, confidence_level = action
                
                if dev_idx < len(test_env.developers):
                    predicted_dev = test_env.developers[dev_idx]
                    confidence = (confidence_level + 1) / 5.0
                    
                    predictions[task_id] = predicted_dev
                    prediction_scores[task_id] = {
                        'predicted_dev': predicted_dev,
                        'rl_confidence': confidence,
                        'dev_idx': dev_idx,
                        'confidence_level': confidence_level
                    }
                
                if (i + 1) % 100 == 0:
                    print(f"   進捗: {i + 1}/{len(test_data)}")
            
            except Exception as e:
                print(f"⚠️ タスク {task_id} のRL予測エラー: {e}")
                continue
        
        print(f"   RL予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores
    
    def hybrid_predict(self, test_data, alpha=0.7):
        """ハイブリッド予測: 類似度ベース + 強化学習"""
        print("🔗 ハイブリッド予測実行中...")
        
        # 類似度ベース予測
        similarity_predictions = {}
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            # シンプルな類似度ベース予測（例）
            best_dev = list(self.similarity_model.developer_embeddings.keys())[0]
            similarity_predictions[task_id] = {
                'dev': best_dev,
                'score': 0.5
            }
        
        # 強化学習予測
        rl_predictions, rl_scores = self.predict_with_rl(test_data)
        
        # ハイブリッド統合
        final_predictions = {}
        final_scores = {}
        
        for task_id in set(similarity_predictions.keys()) & set(rl_predictions.keys()):
            sim_pred = similarity_predictions[task_id]
            rl_pred = rl_predictions[task_id]
            
            # 重み付き統合
            sim_score = sim_pred['score']
            rl_confidence = rl_scores[task_id]['rl_confidence']
            
            combined_score = alpha * sim_score + (1 - alpha) * rl_confidence
            
            # より高いスコアの予測を選択
            if sim_score > rl_confidence:
                final_predictions[task_id] = sim_pred['dev']
                final_scores[task_id] = {
                    'predicted_dev': sim_pred['dev'],
                    'method': 'similarity',
                    'combined_score': combined_score,
                    'sim_score': sim_score,
                    'rl_confidence': rl_confidence
                }
            else:
                final_predictions[task_id] = rl_pred
                final_scores[task_id] = {
                    'predicted_dev': rl_pred,
                    'method': 'reinforcement_learning',
                    'combined_score': combined_score,
                    'sim_score': sim_score,
                    'rl_confidence': rl_confidence
                }
        
        print(f"   ハイブリッド予測完了: {len(final_predictions)} タスク")
        return final_predictions, final_scores
    
    def save_rl_model(self, output_dir="models"):
        """強化学習モデルの保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RLエージェント保存
        rl_model_path = output_dir / f"rl_enhanced_recommender_{timestamp}.zip"
        self.rl_agent.save(rl_model_path)
        
        print(f"✅ RL拡張モデル保存: {rl_model_path}")
        return rl_model_path


def main():
    parser = argparse.ArgumentParser(description='強化学習拡張タスク推薦システム')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--data', default='data/backlog.json',
                       help='統合データパス')
    parser.add_argument('--similarity_model', required=True,
                       help='学習済み類似度モデルパス')
    parser.add_argument('--train_rl', action='store_true',
                       help='強化学習ポリシーを訓練')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # 類似度モデル読み込み
    print("📚 類似度モデル読み込み中...")
    with open(args.similarity_model, 'rb') as f:
        similarity_model_data = pickle.load(f)
    
    # モデル復元（簡略化）
    class SimilarityModel:
        def __init__(self, data):
            self.__dict__.update(data)
    
    similarity_model = SimilarityModel(similarity_model_data)
    
    # データ読み込み
    with open(args.data, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    training_data = [task for task in all_data if task.get('created_at', '').startswith('2022')]
    test_data = [task for task in all_data if task.get('created_at', '').startswith('2023')]
    
    # 強化学習拡張システム
    rl_recommender = RLEnhancedRecommender(similarity_model, args.config)
    
    if args.train_rl:
        # 強化学習訓練
        rl_recommender.train_rl_policy(training_data)
        
        # モデル保存
        rl_recommender.save_rl_model()
    
    # ハイブリッド予測
    predictions, scores = rl_recommender.hybrid_predict(test_data)
    
    print("\n🎯 強化学習拡張システム完了")
    print(f"   予測数: {len(predictions)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
