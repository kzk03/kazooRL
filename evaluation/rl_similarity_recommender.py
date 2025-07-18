#!/usr/bin/env python3
"""
強化学習ベースのシンプル推薦システム

現在のSimpleSimilarityRecommenderの手法を強化学習環境に統合：
- TF-IDFとRandomForestの予測を報酬として活用
- 類似度スコアを状態表現に組み込み
- PPOエージェントで推薦ポリシーを学習
"""

import argparse
import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml

# シンプル類似度推薦システムをインポート
from simple_similarity_recommender import SimpleSimilarityRecommender
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class RLSimilarityEnvironment(gym.Env):
    """強化学習ベースの類似度推薦環境"""
    
    def __init__(self, similarity_recommender, training_pairs, config):
        super().__init__()
        
        self.similarity_recommender = similarity_recommender
        self.training_pairs = training_pairs
        self.config = config
        
        # 開発者リスト
        self.developers = list(similarity_recommender.developer_profiles.keys())
        self.num_developers = len(self.developers)
        
        # 行動空間: 開発者選択
        self.action_space = gym.spaces.Discrete(self.num_developers)
        
        # 観測空間: TF-IDFスコア + 特徴量スコア + 基本特徴量
        # - TF-IDFスコア: 各開発者への類似度 (num_developers次元)
        # - 特徴量スコア: RandomForestの予測確率 (num_developers次元)  
        # - 基本特徴量: タスクの基本特徴 (11次元)
        obs_dim = self.num_developers * 2 + 11
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # エピソード管理
        self.current_episode = 0
        self.reset_count = 0
        
        print(f"🎮 強化学習類似度推薦環境初期化完了")
        print(f"   開発者数: {self.num_developers}")
        print(f"   観測次元: {obs_dim}")
        print(f"   訓練ペア数: {len(training_pairs)}")
    
    def reset(self, seed=None, options=None):
        """エピソードをリセット"""
        super().reset(seed=seed)
        
        # ランダムに訓練ペアを選択
        pair = np.random.choice(self.training_pairs)
        self.current_task_data = pair['task_data']
        self.ground_truth_dev = pair['developer']
        
        # 観測を生成
        obs = self._get_observation()
        self.reset_count += 1
        
        return obs, {}
    
    def step(self, action):
        """アクションを実行"""
        if action >= self.num_developers:
            # 無効なアクション
            reward = -5.0
            terminated = True
            obs = np.zeros(self.observation_space.shape[0])
            return obs, reward, terminated, False, {'invalid_action': True}
        
        selected_dev = self.developers[action]
        
        # 報酬計算
        reward = self._calculate_reward(selected_dev)
        
        # エピソード終了
        terminated = True
        obs = np.zeros(self.observation_space.shape[0])
        
        info = {
            'selected_dev': selected_dev,
            'ground_truth_dev': self.ground_truth_dev,
            'correct_prediction': selected_dev == self.ground_truth_dev,
            'tfidf_scores': getattr(self, 'last_tfidf_scores', {}),
            'feature_scores': getattr(self, 'last_feature_scores', {})
        }
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self):
        """現在の観測を生成"""
        # 1. TF-IDFによる類似度スコア
        tfidf_scores = self.similarity_recommender._predict_by_text_similarity(self.current_task_data)
        tfidf_vector = []
        for dev in self.developers:
            tfidf_vector.append(tfidf_scores.get(dev, 0.0))
        self.last_tfidf_scores = tfidf_scores
        
        # 2. RandomForestによる特徴量スコア
        feature_scores = self.similarity_recommender._predict_by_features(self.current_task_data)
        feature_vector = []
        for dev in self.developers:
            feature_vector.append(feature_scores.get(dev, 0.0))
        self.last_feature_scores = feature_scores
        
        # 3. タスクの基本特徴量
        basic_features = self.similarity_recommender.extract_basic_features(self.current_task_data)
        basic_vector = [
            basic_features.get('title_length', 0),
            basic_features.get('body_length', 0),
            basic_features.get('comments_count', 0),
            basic_features.get('is_bug', 0),
            basic_features.get('is_enhancement', 0),
            basic_features.get('is_documentation', 0),
            basic_features.get('is_question', 0),
            basic_features.get('is_help_wanted', 0),
            basic_features.get('label_count', 0),
            basic_features.get('is_open', 0),
            len(basic_features)  # 特徴量の総数
        ]
        
        # 観測ベクトルの結合
        obs = np.concatenate([tfidf_vector, feature_vector, basic_vector])
        return obs.astype(np.float32)
    
    def _calculate_reward(self, selected_dev):
        """多面的な報酬計算"""
        total_reward = 0.0
        
        # 1. 正解報酬 (最重要)
        if selected_dev == self.ground_truth_dev:
            total_reward += 10.0
        
        # 2. TF-IDF類似度報酬
        if hasattr(self, 'last_tfidf_scores'):
            tfidf_score = self.last_tfidf_scores.get(selected_dev, 0.0)
            total_reward += tfidf_score * 3.0
        
        # 3. RandomForest確信度報酬
        if hasattr(self, 'last_feature_scores'):
            rf_score = self.last_feature_scores.get(selected_dev, 0.0)
            total_reward += rf_score * 5.0
        
        # 4. 統合スコア報酬（既存手法との一致）
        if hasattr(self, 'last_tfidf_scores') and hasattr(self, 'last_feature_scores'):
            text_score = self.last_tfidf_scores.get(selected_dev, 0.0)
            feature_score = self.last_feature_scores.get(selected_dev, 0.0)
            
            # 既存手法と同じ重み付け
            combined_score = 0.6 * text_score + 0.4 * feature_score
            total_reward += combined_score * 2.0
        
        # 5. 開発者の専門性報酬
        dev_profile = self.similarity_recommender.developer_profiles.get(selected_dev, {})
        task_count = dev_profile.get('task_count', 0)
        if task_count > 0:
            # より多くのタスクを担当した開発者には小さなボーナス
            experience_bonus = min(task_count / 100.0, 1.0)  # 最大1.0
            total_reward += experience_bonus
        
        # 6. ラベル一致報酬
        task_labels = [label.get('name', '').lower() for label in self.current_task_data.get('labels', [])]
        if task_labels and selected_dev in self.similarity_recommender.developer_profiles:
            dev_label_prefs = self.similarity_recommender.developer_profiles[selected_dev].get('label_preferences', {})
            label_match_score = 0.0
            
            for label in task_labels:
                if label in dev_label_prefs:
                    label_match_score += dev_label_prefs[label] / dev_profile.get('task_count', 1)
            
            total_reward += min(label_match_score, 2.0)  # 最大2.0のボーナス
        
        return total_reward


class RLSimilarityRecommender:
    """強化学習ベースの類似度推薦システム"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ベースとなる類似度推薦システム
        self.similarity_recommender = SimpleSimilarityRecommender(config_path)
        
        print("🤖 強化学習類似度推薦システム初期化完了")
    
    def train_base_models(self, data_path):
        """ベースとなる類似度モデルを訓練"""
        print("📚 ベース類似度モデル訓練中...")
        
        # データ読み込み
        training_data, test_data = self.similarity_recommender.load_data(data_path)
        
        # 学習ペア抽出
        training_pairs, developer_stats = self.similarity_recommender.extract_training_pairs(training_data)
        
        if not training_pairs:
            raise ValueError("学習ペアが見つかりませんでした")
        
        # 開発者プロファイル構築
        self.similarity_recommender.developer_profiles = self.similarity_recommender.build_developer_profiles(training_pairs)
        
        # TF-IDFモデル訓練
        self.similarity_recommender.train_text_similarity_model(self.similarity_recommender.developer_profiles)
        
        # RandomForestモデル訓練
        self.similarity_recommender.train_feature_model(training_pairs)
        
        print("✅ ベース類似度モデル訓練完了")
        return training_pairs, test_data
    
    def train_rl_policy(self, training_pairs, total_timesteps=100000):
        """強化学習ポリシーを訓練"""
        print("🎯 強化学習ポリシー訓練開始...")
        
        # 環境作成
        def make_env():
            return RLSimilarityEnvironment(
                self.similarity_recommender, training_pairs, self.config
            )
        
        env = DummyVecEnv([make_env])
        
        # PPOエージェント作成
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
            ent_coef=0.01,  # 探索を促進
            device="auto",
            seed=42
        )
        
        # 訓練実行
        print(f"   訓練ステップ数: {total_timesteps:,}")
        print(f"   推定訓練時間: {total_timesteps / 10000:.1f}分")
        
        # コールバック設定
        def callback(locals_, globals_):
            if locals_['self'].num_timesteps % 10000 == 0:
                print(f"   進捗: {locals_['self'].num_timesteps:,}/{total_timesteps:,} ステップ")
            return True
        
        self.rl_agent.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        print("✅ 強化学習ポリシー訓練完了")
    
    def predict_with_rl(self, test_data):
        """強化学習ポリシーで予測"""
        print("🤖 強化学習予測実行中...")
        
        predictions = {}
        prediction_scores = {}
        
        # テスト環境設定
        test_env = RLSimilarityEnvironment(
            self.similarity_recommender, [], self.config
        )
        
        prediction_count = 0
        
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            try:
                # タスクを環境に設定
                test_env.current_task_data = task_data
                obs = test_env._get_observation()
                
                # 強化学習エージェントで予測
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                
                if action < len(test_env.developers):
                    predicted_dev = test_env.developers[action]
                    
                    # 予測信頼度計算
                    tfidf_score = test_env.last_tfidf_scores.get(predicted_dev, 0.0)
                    feature_score = test_env.last_feature_scores.get(predicted_dev, 0.0)
                    combined_score = 0.6 * tfidf_score + 0.4 * feature_score
                    
                    predictions[task_id] = predicted_dev
                    prediction_scores[task_id] = {
                        'predicted_dev': predicted_dev,
                        'rl_action': int(action),
                        'tfidf_score': tfidf_score,
                        'feature_score': feature_score,
                        'combined_score': combined_score,
                        'method': 'reinforcement_learning'
                    }
                    
                    prediction_count += 1
                    
                    if prediction_count % 100 == 0:
                        print(f"   進捗: {prediction_count}/{len(test_data)}")
                
            except Exception as e:
                print(f"⚠️ タスク {task_id} のRL予測エラー: {e}")
                continue
        
        print(f"   RL予測完了: {len(predictions)} タスク")
        return predictions, prediction_scores
    
    def compare_with_baseline(self, test_data):
        """ベースライン手法と比較"""
        print("📊 ベースライン比較実行中...")
        
        # 1. 既存の類似度ベース予測
        print("   類似度ベース予測...")
        similarity_predictions, similarity_scores, test_assignments = self.similarity_recommender.predict_assignments(
            test_data, self.similarity_recommender.developer_profiles
        )
        
        # 2. 強化学習ベース予測
        print("   強化学習ベース予測...")
        rl_predictions, rl_scores = self.predict_with_rl(test_data)
        
        # 3. 比較評価
        comparison_results = {
            'similarity_method': {
                'predictions': similarity_predictions,
                'scores': similarity_scores,
                'metrics': self._evaluate_predictions(similarity_predictions, test_assignments)
            },
            'rl_method': {
                'predictions': rl_predictions,
                'scores': rl_scores,
                'metrics': self._evaluate_predictions(rl_predictions, test_assignments)
            },
            'test_assignments': test_assignments
        }
        
        return comparison_results
    
    def _evaluate_predictions(self, predictions, test_assignments):
        """予測の評価"""
        if not predictions or not test_assignments:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}
        
        common_tasks = set(predictions.keys()) & set(test_assignments.keys())
        if not common_tasks:
            return {'accuracy': 0.0, 'correct': 0, 'total': 0}
        
        correct = sum(1 for task_id in common_tasks if predictions[task_id] == test_assignments[task_id])
        total = len(common_tasks)
        accuracy = correct / total
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def save_rl_model(self, output_dir="models"):
        """強化学習モデルの保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RLエージェント保存
        rl_model_path = output_dir / f"rl_similarity_recommender_{timestamp}.zip"
        self.rl_agent.save(rl_model_path)
        
        # ベース類似度モデル保存
        base_model_path = self.similarity_recommender.save_model(output_dir)
        
        print(f"✅ RL類似度モデル保存: {rl_model_path}")
        return rl_model_path, base_model_path
    
    def run_full_pipeline(self, data_path, output_dir="outputs"):
        """完全パイプライン実行"""
        print("🚀 強化学習類似度推薦システム実行開始")
        print("=" * 70)
        
        # 1. ベースモデル訓練
        training_pairs, test_data = self.train_base_models(data_path)
        
        # 2. 強化学習ポリシー訓練
        self.train_rl_policy(training_pairs)
        
        # 3. 比較評価
        comparison_results = self.compare_with_baseline(test_data)
        
        # 4. 結果保存
        self.save_comparison_results(comparison_results, output_dir)
        
        # 5. モデル保存
        model_paths = self.save_rl_model()
        
        print("✅ 強化学習類似度推薦システム完了")
        return comparison_results
    
    def save_comparison_results(self, comparison_results, output_dir):
        """比較結果の保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # メトリクス比較
        comparison_metrics = {
            'similarity_accuracy': comparison_results['similarity_method']['metrics']['accuracy'],
            'rl_accuracy': comparison_results['rl_method']['metrics']['accuracy'],
            'similarity_correct': comparison_results['similarity_method']['metrics']['correct'],
            'rl_correct': comparison_results['rl_method']['metrics']['correct'],
            'total_tasks': comparison_results['similarity_method']['metrics']['total'],
            'improvement': (comparison_results['rl_method']['metrics']['accuracy'] - 
                          comparison_results['similarity_method']['metrics']['accuracy'])
        }
        
        # JSON保存
        metrics_path = output_dir / f"rl_similarity_comparison_{timestamp}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_metrics, f, indent=2, ensure_ascii=False, default=str)
        
        # 詳細結果CSV保存
        results = []
        all_task_ids = (set(comparison_results['similarity_method']['predictions'].keys()) | 
                       set(comparison_results['rl_method']['predictions'].keys()) |
                       set(comparison_results['test_assignments'].keys()))
        
        for task_id in all_task_ids:
            result = {
                'task_id': task_id,
                'actual_developer': comparison_results['test_assignments'].get(task_id),
                'similarity_prediction': comparison_results['similarity_method']['predictions'].get(task_id),
                'rl_prediction': comparison_results['rl_method']['predictions'].get(task_id),
                'similarity_correct': (comparison_results['similarity_method']['predictions'].get(task_id) == 
                                     comparison_results['test_assignments'].get(task_id)),
                'rl_correct': (comparison_results['rl_method']['predictions'].get(task_id) == 
                              comparison_results['test_assignments'].get(task_id))
            }
            
            # スコア情報追加
            if task_id in comparison_results['similarity_method']['scores']:
                sim_score = comparison_results['similarity_method']['scores'][task_id]
                result['similarity_combined_score'] = sim_score.get('combined_score', 0.0)
                result['similarity_text_score'] = sim_score.get('text_score', 0.0)
                result['similarity_feature_score'] = sim_score.get('feature_score', 0.0)
            
            if task_id in comparison_results['rl_method']['scores']:
                rl_score = comparison_results['rl_method']['scores'][task_id]
                result['rl_combined_score'] = rl_score.get('combined_score', 0.0)
                result['rl_tfidf_score'] = rl_score.get('tfidf_score', 0.0)
                result['rl_feature_score'] = rl_score.get('feature_score', 0.0)
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_path = output_dir / f"rl_similarity_detailed_{timestamp}.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        
        print(f"✅ 比較メトリクス保存: {metrics_path}")
        print(f"✅ 詳細結果保存: {results_path}")
        
        # 結果サマリー表示
        print("\n📊 比較結果サマリー:")
        print(f"   類似度ベース精度: {comparison_metrics['similarity_accuracy']:.3f}")
        print(f"   強化学習精度: {comparison_metrics['rl_accuracy']:.3f}")
        print(f"   改善度: {comparison_metrics['improvement']:+.3f}")
        print(f"   評価タスク数: {comparison_metrics['total_tasks']}")


def main():
    parser = argparse.ArgumentParser(description='強化学習類似度推薦システム')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--data', default='data/backlog.json',
                       help='統合データパス')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='強化学習の訓練ステップ数')
    
    args = parser.parse_args()
    
    # 強化学習類似度推薦システム実行
    rl_recommender = RLSimilarityRecommender(args.config)
    comparison_results = rl_recommender.run_full_pipeline(args.data, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())
