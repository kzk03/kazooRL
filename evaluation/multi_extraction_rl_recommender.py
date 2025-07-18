#!/usr/bin/env python3
"""
マルチ抽出方法対応の強化学習推薦システム

- 複数の開発者抽出方法（assignees, creators, all）をRL状態に組み込み
- 動的候補プール選択をアクション空間に含める
- 時系列ベースの報酬設計
- メタ学習による適応的抽出方法選択
"""

import argparse
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
from gymnasium import spaces
from simple_similarity_recommender import SimpleSimilarityRecommender
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class MultiExtractionRLEnvironment(gym.Env):
    """
    マルチ抽出方法対応の強化学習環境
    
    状態空間: [タスク特徴, 時系列情報, 候補プール統計, 抽出方法履歴]
    アクション空間: [抽出方法選択, 開発者選択]
    報酬: 成功報酬 + 抽出方法適切性ボーナス + 時系列一貫性ボーナス
    """
    
    def __init__(self, similarity_recommender, training_data, config):
        super().__init__()
        
        self.recommender = similarity_recommender
        self.training_data = training_data
        self.config = config
        
        # 抽出方法の定義
        self.extraction_methods = ['assignees', 'creators', 'all']
        self.method_to_id = {method: i for i, method in enumerate(self.extraction_methods)}
        
        # 状態空間の定義
        # [タスク特徴(10次元) + 時系列情報(5次元) + 候補プール統計(6次元) + 抽出履歴(3次元)]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )
        
        # アクション空間: [抽出方法(3), 開発者インデックス(最大100)]
        # マルチディスクリート: 最初の次元で抽出方法、2番目で開発者を選択
        self.action_space = spaces.MultiDiscrete([3, 100])
        
        # 環境状態
        self.current_task_idx = 0
        self.extraction_history = np.zeros(3)  # 各方法の使用回数
        self.recent_rewards = []
        self.time_step = 0
        
        print(f"🤖 マルチ抽出RL環境初期化:")
        print(f"   状態次元: {self.observation_space.shape[0]}")
        print(f"   アクション空間: 抽出方法×{len(self.extraction_methods)}, 開発者×100")
        print(f"   学習タスク数: {len(training_data)}")
    
    def reset(self, seed=None, options=None):
        """環境をリセット"""
        super().reset(seed=seed)
        
        self.current_task_idx = 0
        self.extraction_history = np.zeros(3)
        self.recent_rewards = []
        self.time_step = 0
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """1ステップ実行"""
        extraction_method_id, developer_idx = action
        extraction_method = self.extraction_methods[extraction_method_id]
        
        # 現在のタスクを取得
        if self.current_task_idx >= len(self.training_data):
            # エピソード終了
            obs = self._get_observation()
            return obs, 0.0, True, True, {}
        
        current_task = self.training_data[self.current_task_idx]
        
        # 報酬計算とタスク実行
        reward, info = self._calculate_reward(current_task, extraction_method, developer_idx)
        
        # 履歴更新
        self.extraction_history[extraction_method_id] += 1
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > 10:
            self.recent_rewards.pop(0)
        
        # 次のタスクへ
        self.current_task_idx += 1
        self.time_step += 1
        
        # エピソード終了判定
        terminated = self.current_task_idx >= len(self.training_data)
        truncated = False  # タイムアウトなし
        
        obs = self._get_observation()
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """現在の観測状態を取得"""
        if self.current_task_idx >= len(self.training_data):
            # 終了状態
            return np.zeros(24, dtype=np.float32)
        
        current_task = self.training_data[self.current_task_idx]
        
        # 1. タスク特徴量 (10次元)
        task_features = self._extract_task_features(current_task)
        
        # 2. 時系列情報 (5次元)
        temporal_features = self._extract_temporal_features(current_task)
        
        # 3. 候補プール統計 (6次元)
        pool_stats = self._extract_pool_statistics(current_task)
        
        # 4. 抽出履歴 (3次元)
        history_normalized = self.extraction_history / max(1, self.time_step)
        
        # 全特徴量を結合
        obs = np.concatenate([
            task_features,
            temporal_features, 
            pool_stats,
            history_normalized
        ]).astype(np.float32)
        
        return obs
    
    def _extract_task_features(self, task_data):
        """タスクから特徴量を抽出"""
        features = self.recommender.extract_basic_features(task_data)
        
        # 10次元の特徴ベクトル
        return np.array([
            features.get('title_length', 0) / 100.0,  # 正規化
            features.get('body_length', 0) / 1000.0,
            features.get('comments_count', 0) / 10.0,
            features.get('is_bug', 0),
            features.get('is_enhancement', 0),
            features.get('is_documentation', 0),
            features.get('is_question', 0),
            features.get('is_help_wanted', 0),
            features.get('label_count', 0) / 5.0,
            features.get('is_open', 0)
        ], dtype=np.float32)
    
    def _extract_temporal_features(self, task_data):
        """時系列特徴量を抽出"""
        created_at = task_data.get('created_at', '')
        
        try:
            if created_at:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                
                # 時系列特徴
                return np.array([
                    dt.month / 12.0,  # 月
                    dt.weekday() / 6.0,  # 曜日
                    dt.hour / 23.0,  # 時間
                    len(self.recent_rewards) / 10.0,  # 最近のエピソード数
                    np.mean(self.recent_rewards) if self.recent_rewards else 0.0  # 最近の平均報酬
                ], dtype=np.float32)
        except:
            pass
        
        return np.zeros(5, dtype=np.float32)
    
    def _extract_pool_statistics(self, task_data):
        """各抽出方法の候補プール統計"""
        stats = []
        
        for method in self.extraction_methods:
            # 各方法での開発者数を取得
            developers = self.recommender.extract_developers_from_task(task_data, method=method)
            
            # 統計計算
            dev_count = len(developers)
            avg_priority = np.mean([dev['priority'] for dev in developers]) if developers else 0
            
            stats.extend([dev_count / 100.0, avg_priority / 3.0])  # 正規化
        
        return np.array(stats, dtype=np.float32)
    
    def _calculate_reward(self, task_data, extraction_method, developer_idx):
        """報酬計算"""
        # 指定された抽出方法で開発者を抽出
        developers = self.recommender.extract_developers_from_task(task_data, method=extraction_method)
        
        if not developers:
            return -0.1, {'reason': 'no_developers', 'method': extraction_method}
        
        # 開発者インデックスの調整
        actual_dev_idx = min(developer_idx, len(developers) - 1)
        selected_dev = developers[actual_dev_idx]
        
        # 実際の正解開発者を取得
        actual_developers = self.recommender.extract_developers_from_task(task_data, method='all')
        actual_dev = actual_developers[0]['login'] if actual_developers else None
        
        # 基本報酬
        base_reward = 0.0
        success = False
        
        if actual_dev and selected_dev['login'] == actual_dev:
            # 成功時の報酬
            method_rewards = {
                'assignees': 1.0,    # 最高品質
                'creators': 0.7,     # 高モチベーション
                'all': 0.8           # バランス
            }
            base_reward = method_rewards.get(extraction_method, 0.5)
            success = True
        else:
            # 失敗時の軽微なペナルティ
            base_reward = -0.05
        
        # ボーナス報酬
        bonus_reward = 0.0
        
        # 1. 抽出方法適切性ボーナス
        if extraction_method == 'assignees' and selected_dev['priority'] == 1:
            bonus_reward += 0.1  # 高優先度開発者を選択
        elif extraction_method == 'creators' and selected_dev['source'] == 'user_creator':
            bonus_reward += 0.1  # 作成者を適切に選択
        
        # 2. 時系列一貫性ボーナス
        if len(self.recent_rewards) >= 3:
            recent_success_rate = np.mean([r > 0 for r in self.recent_rewards[-3:]])
            if recent_success_rate > 0.5:
                bonus_reward += 0.05  # 最近の成功率が高い
        
        # 3. 探索ボーナス
        method_usage = self.extraction_history / max(1, self.time_step)
        if np.std(method_usage) > 0.1:  # 均等に探索している
            bonus_reward += 0.02
        
        total_reward = base_reward + bonus_reward
        
        info = {
            'success': success,
            'method': extraction_method,
            'selected_dev': selected_dev['login'],
            'actual_dev': actual_dev,
            'base_reward': base_reward,
            'bonus_reward': bonus_reward,
            'priority': selected_dev['priority'],
            'source': selected_dev['source']
        }
        
        return total_reward, info


class MultiExtractionRLRecommender:
    """マルチ抽出方法対応の強化学習推薦システム"""
    
    def __init__(self, config_path):
        self.config_path = config_path
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ベースとなる類似度推薦システム
        self.similarity_recommender = SimpleSimilarityRecommender(config_path)
        
        print("🚀 マルチ抽出RL推薦システム初期化完了")
    
    def train_base_models(self, data_path):
        """ベースモデルを訓練"""
        print("📚 ベースモデル訓練中...")
        
        # データ読み込み
        training_data, test_data = self.similarity_recommender.load_data(data_path)
        
        # 'all'方法で学習ペア抽出（最も包括的）
        training_pairs, developer_stats = self.similarity_recommender.extract_training_pairs(
            training_data, extraction_method='all'
        )
        
        if not training_pairs:
            raise ValueError("学習ペアが見つかりませんでした")
        
        # 時系列アクティビティ構築
        self.similarity_recommender.build_developer_activity_timeline(
            training_data, extraction_method='all'
        )
        
        # 開発者プロファイル構築
        learned_profiles = self.similarity_recommender.build_developer_profiles(training_pairs)
        
        # モデル訓練
        self.similarity_recommender.train_text_similarity_model(learned_profiles)
        self.similarity_recommender.train_feature_model(training_pairs)
        
        print("✅ ベースモデル訓練完了")
        return training_pairs, test_data
    
    def train_rl_agent(self, training_pairs, total_timesteps=50000):
        """強化学習エージェントを訓練"""
        print("🎯 マルチ抽出RL訓練開始...")
        
        # 環境作成
        def make_env():
            return MultiExtractionRLEnvironment(
                self.similarity_recommender, 
                [pair['task_data'] for pair in training_pairs],
                self.config
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
            ent_coef=0.01,
            device="auto",
            seed=42,
            policy_kwargs={
                'net_arch': [256, 256, 128]  # より深いネットワーク
            }
        )
        
        # 訓練実行
        print(f"   訓練ステップ数: {total_timesteps:,}")
        self.rl_agent.learn(total_timesteps=total_timesteps)
        
        print("✅ マルチ抽出RL訓練完了")
    
    def predict_with_rl(self, test_data):
        """強化学習エージェントで予測"""
        print("🔮 マルチ抽出RL予測実行...")
        
        predictions = {}
        prediction_scores = {}
        method_usage_stats = Counter()
        
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
            
            try:
                # 環境作成（単一タスク用）
                temp_env = MultiExtractionRLEnvironment(
                    self.similarity_recommender,
                    [task_data],
                    self.config
                )
                
                # 観測取得
                obs, _ = temp_env.reset()
                
                # RLエージェントで予測
                action, _ = self.rl_agent.predict(obs, deterministic=True)
                extraction_method_id, developer_idx = action
                
                # 抽出方法とインデックスを解釈
                extraction_method = temp_env.extraction_methods[extraction_method_id]
                method_usage_stats[extraction_method] += 1
                
                # 指定された方法で開発者を抽出
                developers = self.similarity_recommender.extract_developers_from_task(
                    task_data, method=extraction_method
                )
                
                if developers:
                    actual_dev_idx = min(developer_idx, len(developers) - 1)
                    selected_dev = developers[actual_dev_idx]
                    
                    predictions[task_id] = selected_dev['login']
                    prediction_scores[task_id] = {
                        'predicted_dev': selected_dev['login'],
                        'extraction_method': extraction_method,
                        'developer_idx': actual_dev_idx,
                        'priority': selected_dev['priority'],
                        'source': selected_dev['source'],
                        'pool_size': len(developers)
                    }
                
            except Exception as e:
                print(f"⚠️ タスク {task_id} の予測エラー: {e}")
                continue
        
        print(f"   予測完了: {len(predictions)} タスク")
        print("   抽出方法使用統計:")
        for method, count in method_usage_stats.items():
            print(f"     {method}: {count} タスク ({count/len(predictions)*100:.1f}%)")
        
        return predictions, prediction_scores
    
    def evaluate_predictions(self, predictions, test_data):
        """予測結果を評価"""
        print("📊 マルチ抽出RL評価中...")
        
        # テストデータの正解を抽出
        test_assignments = {}
        for task_data in test_data:
            task_id = task_data.get('id') or task_data.get('number')
            if not task_id:
                continue
                
            # 'all'方法で正解を取得
            developers = self.similarity_recommender.extract_developers_from_task(
                task_data, method='all'
            )
            if developers:
                test_assignments[task_id] = developers[0]['login']
        
        # 評価計算
        common_tasks = set(predictions.keys()) & set(test_assignments.keys())
        correct_predictions = 0
        
        for task_id in common_tasks:
            if predictions[task_id] == test_assignments[task_id]:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(common_tasks) if common_tasks else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': len(common_tasks)
        }
        
        print(f"   マルチ抽出RL精度: {accuracy:.3f} ({correct_predictions}/{len(common_tasks)})")
        
        return metrics
    
    def run_full_pipeline(self, data_path, total_timesteps=50000, output_dir="outputs"):
        """完全パイプライン実行"""
        print("🚀 マルチ抽出RL推薦システム実行開始")
        print("=" * 70)
        
        # 1. ベースモデル訓練
        training_pairs, test_data = self.train_base_models(data_path)
        
        # 2. 強化学習エージェント訓練
        self.train_rl_agent(training_pairs, total_timesteps)
        
        # 3. 予測実行
        predictions, prediction_scores = self.predict_with_rl(test_data)
        
        # 4. 評価
        metrics = self.evaluate_predictions(predictions, test_data)
        
        # 5. 結果保存
        self.save_results(metrics, predictions, prediction_scores, output_dir)
        
        print("✅ マルチ抽出RL推薦システム完了")
        return metrics
    
    def save_results(self, metrics, predictions, prediction_scores, output_dir):
        """結果保存"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # メトリクス保存
        metrics_path = output_dir / f"multi_extraction_rl_metrics_{timestamp}.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 予測結果保存
        results = []
        for task_id, predicted_dev in predictions.items():
            result = {
                'task_id': task_id,
                'predicted_developer': predicted_dev
            }
            
            if task_id in prediction_scores:
                result.update(prediction_scores[task_id])
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_path = output_dir / f"multi_extraction_rl_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        
        # モデル保存
        model_path = output_dir / f"multi_extraction_rl_agent_{timestamp}.zip"
        self.rl_agent.save(model_path)
        
        print(f"✅ メトリクス保存: {metrics_path}")
        print(f"✅ 結果保存: {results_path}")
        print(f"✅ モデル保存: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='マルチ抽出方法対応の強化学習推薦システム')
    parser.add_argument('--config', default='configs/unified_rl.yaml',
                       help='設定ファイルパス')
    parser.add_argument('--data', default='data/backlog.json',
                       help='統合データパス')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='強化学習訓練ステップ数')
    parser.add_argument('--output', default='outputs',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # マルチ抽出RL推薦システム実行
    recommender = MultiExtractionRLRecommender(args.config)
    metrics = recommender.run_full_pipeline(
        args.data,
        total_timesteps=args.timesteps,
        output_dir=args.output
    )
    
    print("\n🎯 最終結果:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value:.3f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
